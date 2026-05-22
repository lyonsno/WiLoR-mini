from types import SimpleNamespace
import sys
import types

import pytest
import torch
from torch import nn


roma_stub = types.ModuleType("roma")
roma_stub.rotmat_to_rotvec = lambda rotmats: torch.zeros(
    *rotmats.shape[:-2], 3, device=rotmats.device, dtype=rotmats.dtype
)
sys.modules.setdefault("roma", roma_stub)

vit_stub = types.ModuleType("wilor_mini.models.vit")
vit_stub.vit = lambda **kwargs: None
vit_stub.rot6d_to_rotmat = lambda x: x.reshape(-1, 2, 3).new_zeros(
    x.shape[0], 3, 3
)
sys.modules.setdefault("wilor_mini.models.vit", vit_stub)

mano_wrapper_stub = types.ModuleType("wilor_mini.models.mano_wrapper")
mano_wrapper_stub.MANO = lambda *args, **kwargs: None
sys.modules.setdefault("wilor_mini.models.mano_wrapper", mano_wrapper_stub)

from wilor_mini.models.refinement_net import DeConvNet, DeConvNet_v2
from wilor_mini.models.wilor import WiLor


class _ContiguousBackbone(nn.Module):
    def forward(self, image_crop):
        assert image_crop.device.type == "mps"
        assert image_crop.shape == (1, 3, 256, 192)
        assert image_crop.is_contiguous(), (
            "MPS backbone input must be contiguous after the width crop"
        )

        batch_size = image_crop.shape[0]
        eye = torch.eye(3, device=image_crop.device, dtype=image_crop.dtype)
        hand_rotmats = eye.expand(batch_size, 16, 3, 3).clone()
        pred_mano_feats = {
            "hand_pose": torch.zeros(
                batch_size, 96, device=image_crop.device, dtype=image_crop.dtype
            ),
            "betas": torch.zeros(
                batch_size, 10, device=image_crop.device, dtype=image_crop.dtype
            ),
            "cam": torch.zeros(
                batch_size, 3, device=image_crop.device, dtype=image_crop.dtype
            ),
        }
        return (
            {
                "global_orient": hand_rotmats[:, :1],
                "hand_pose": hand_rotmats[:, 1:],
                "betas": torch.zeros(
                    batch_size, 10, device=image_crop.device, dtype=image_crop.dtype
                ),
            },
            torch.ones(batch_size, 3, device=image_crop.device, dtype=image_crop.dtype),
            pred_mano_feats,
            torch.zeros(
                batch_size, 1280, 16, 12, device=image_crop.device, dtype=image_crop.dtype
            ),
        )


class _RefineNet(nn.Module):
    def forward(self, img_feat, verts_3d, pred_cam, pred_mano_feats, focal_length):
        batch_size = img_feat.shape[0]
        eye = torch.eye(3, device=img_feat.device, dtype=img_feat.dtype)
        hand_rotmats = eye.expand(batch_size, 16, 3, 3).clone()
        return {
            "global_orient": hand_rotmats[:, :1],
            "hand_pose": hand_rotmats[:, 1:],
            "betas": torch.zeros(
                batch_size, 10, device=img_feat.device, dtype=img_feat.dtype
            ),
            "pred_cam": torch.ones(batch_size, 3, device=img_feat.device, dtype=img_feat.dtype),
        }


class _Mano:
    def __call__(self, **kwargs):
        batch_size = kwargs["betas"].shape[0]
        device = kwargs["betas"].device
        dtype = kwargs["betas"].dtype
        return SimpleNamespace(
            vertices=torch.zeros(batch_size, 778, 3, device=device, dtype=dtype),
            joints=torch.zeros(batch_size, 21, 3, device=device, dtype=dtype),
        )


class _AssertContiguous(nn.Module):
    def forward(self, img_feat):
        assert img_feat.is_contiguous(), (
            "MPS refinement features must be contiguous before first_conv"
        )
        return img_feat


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is required")
def test_wilor_forward_passes_contiguous_cropped_image_to_backbone_on_mps():
    model = WiLor.__new__(WiLor)
    nn.Module.__init__(model)
    model.backbone = _ContiguousBackbone()
    model.refine_net = _RefineNet()
    model.mano = _Mano()
    model.FOCAL_LENGTH = 5000
    model.IMAGE_SIZE = 256
    model.IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
    model.IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
    model.eval()

    image = torch.arange(256 * 256 * 3, device="mps", dtype=torch.float32).reshape(
        1, 256, 256, 3
    )

    with torch.no_grad():
        output = model(image)

    assert output["pred_vertices"].shape == (1, 778, 3)
    assert output["pred_keypoints_3d"].shape == (1, 21, 3)


@pytest.mark.parametrize("deconv_cls", [DeConvNet, DeConvNet_v2])
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is required")
def test_refinement_deconv_makes_image_features_contiguous_before_first_conv_on_mps(
    deconv_cls,
):
    deconv = deconv_cls(feat_dim=8)
    deconv.first_conv = _AssertContiguous()
    deconv.deconv = (
        nn.ModuleList([nn.Identity()])
        if isinstance(deconv.deconv, nn.ModuleList)
        else nn.Identity()
    )
    deconv.to("mps")

    img_feat = torch.zeros(1, 4, 4, 8, device="mps").permute(0, 3, 1, 2)
    assert not img_feat.is_contiguous()

    deconv(img_feat)
