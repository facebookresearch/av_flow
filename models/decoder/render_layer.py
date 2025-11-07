"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from drtk.interpolate import interpolate, interpolate_ref

from drtk.rasterize import rasterize
from drtk.render import render, render_ref
from drtk.screen_space_uv_derivative import screen_space_uv_derivative
from drtk.transform import transform_with_v_cam
from drtk.utils import vert_binormals


RENDERLAYER_OUTPUTS = {
    "render",
    "mask",
    "vt_img",
    "bary_img",
    "vn_img",
    "vbn_img",
    "depth_img",
    "index_img",
    "v_pix_img",
    "v_cam_img",
    "v_img",
    "v_pix",
    "v_cam",
    "vt_dxdy_img",
}


class RenderLayer(nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
        vt: Union[np.ndarray, th.Tensor],
        vi: Union[np.ndarray, th.Tensor],
        vti: Union[np.ndarray, th.Tensor],
        flip_uvs: bool = True,
        grid_sample_params: Optional[Dict[str, Any]] = None,
        use_python_renderer: bool = False,
    ) -> None:
        """Create a RenderLayer that produces w x h images."""
        super(RenderLayer, self).__init__()

        self.h = h
        self.w = w
        self.flip_uvs = flip_uvs

        self.use_python_renderer = use_python_renderer

        self.grid_sample_params: Dict[str, Any] = grid_sample_params or {
            "mode": "bilinear",
            "align_corners": False,
        }

        # This is particularly important for rendering differentiable mask
        # losses where we use textures filled with solid colors. On the border,
        # if we interpolate with zeros outside, we get a vastly different
        # result for some texels.
        self.grid_sample_params["padding_mode"] = "border"

        if not isinstance(vt, th.Tensor):
            vt = th.from_numpy(vt)
        if not isinstance(vi, th.Tensor):
            vi = th.from_numpy(vi)
        if not isinstance(vti, th.Tensor):
            vti = th.from_numpy(vti)

        self.register_buffer("vt", vt.clone().float().contiguous(), persistent=False)
        self.register_buffer("vi", vi.clone().int().contiguous(), persistent=False)
        self.register_buffer("vti", vti.clone().int().contiguous(), persistent=False)

        if flip_uvs:
            self.vt[:, 1] = 1 - self.vt[:, 1]

    def resize(self, h: int, w: int) -> None:
        self.h = h
        self.w = w

    def forward(
        self,
        v: th.Tensor,
        tex: th.Tensor,
        campos: Optional[th.Tensor] = None,
        camrot: Optional[th.Tensor] = None,
        focal: Optional[th.Tensor] = None,
        princpt: Optional[th.Tensor] = None,
        K: Optional[th.Tensor] = None,
        Rt: Optional[th.Tensor] = None,
        vt: Optional[th.Tensor] = None,
        vi: Optional[th.Tensor] = None,
        vti: Optional[th.Tensor] = None,
        distortion_mode: Optional[Sequence[str]] = None,
        distortion_coeff: Optional[th.Tensor] = None,
        fov: Optional[th.Tensor] = None,
        vn: Optional[th.Tensor] = None,
        background: Optional[th.Tensor] = None,
        output_filters: Optional[Sequence[str]] = None,
    ) -> Dict[str, th.Tensor]:
        """
        v: Tensor, N x V x 3
        Batch of vertex positions for vertices in the mesh.

        tex: Tensor, N x C x H x W
        Batch of textures to render on the mesh.

        campos: Tensor, N x 3
        Camera position.

        camrot: Tensor, N x 3 x 3
        Camera rotation matrix.

        focal: Tensor, N x 2 x 2
        Focal length [[fx, 0],
                      [0, fy]]

        princpt: Tensor, N x 2
        Principal point [cx, cy]

        K: Tensor, N x 3 x 3
        Camera intrinsic calibration matrix. Either this or both (focal,
        princpt) must be provided.

        Rt: Tensor, N x 3 x 4 or N x 4 x 4
        Camera extrinsic matrix. Either this or both (camrot, campos) must be
        provided. Camrot is the upper 3x3 of Rt, campos = -R.T @ t.

        vt: Tensor, Ntexcoords x 2
        Optional texcoords to use. If given, they override the ones
        used to construct this RenderLayer.

        vi: Tensor, Nfaces x 3
        Optional face vertex indices to use. If given, they override the ones
        used to construct this RenderLayer.

        vti: Tensor, Nfaces x 3
        Optional face texcoord indices to use. If given, they override the ones
        used to construct this RenderLayer.

        distortion_mode: Sequence[str]
        Names of the distortion modes.

        distortion_coeff: Tensor, N x 4
        Distortion coefficients.

        fov: Tensor, N x 1
        Valid field of view of the distortion model.

        depth_img: Tensor, N x H x W
        Optional pre-existing depth map. Render triangles on top of this depth
        map, discarding any triangles that lie behind the surface represented
        by the map.

        vn: Tensor, N x Nverts x 3
        Optional vertex normals. If given, they will be interpolated along the
        surface to give per-pixel interpolated normals.

        background: Tensor, N x C x H x W
        Background images on which to composite the rendered mesh.

        near: float
        Near plane.

        output_filters: Sequence[str]
        List of output names to return. Not returning unused outputs can save GPU
        memory. Valid output names:

        render:     The rendered masked image.
                    N x C x H x W

        mask:       Mask of which pixels contain valid rendered colors.
                    N x H x W

        vt_img:     Per-pixel interpolated texture coordinates.
                    N x H x W x 2

        bary_img:   Per-pixel interpolated 3D barycentric coordinates.
                    N x 3 x H x W

        vn_img:     Per-pixel interpolated vertex normals (if vn was given).
                    N x H x W x 3

        vbn_img:    Per-pixel interpolated vertex binormals (if vn was given).
                    N x H x W x 3

        index_img:  Per-pixel face indices.
                    N x H x W

        v_pix_img:  Per-pixel pixel-space vertex coordinates with preserved camera-space Z-values.
                    N x H x W x 3

        v_cam_img:  Per-pixel camera-space vertex coordinates.
                    N x H x W x 3

        v_img:      Per-pixel vertex coordinates.
                    N x H x W x 3

        v_pix:      Pixel-space vertex coordinates with preserved camera-space Z-values.
                    N x V x 3

        v_cam:      Camera-space vertex coordinates.
                    N x V x 3

        vt_dxdy_img: Per-pixel uv gradients with respect to the pixel-space position.
                     vt_dxdy_img is transposed Jacobian: (dt / dp_pix)^T, where:
                        t - uv coordinates, p_pix - pixel-space coordinates
                        vt_dxdy_img[..., i, j] = dt[j] / dp_pix[i]
                     e.i. image of 2x2 Jacobian matrices of the form: [[du/dx, dv/dx],
                                                                       [du/dy, dv/dy]]
                    N x H x W x 2 x 2

        all:        All of the above.
        """
        if output_filters is None:
            output_filters = ["render"]

        interpolate_func = interpolate_ref if self.use_python_renderer else interpolate
        render_func = render_ref if self.use_python_renderer else render

        vt = vt if vt is not None else self.vt
        vi = vi if vi is not None else self.vi
        vti = vti if vti is not None else self.vti
        assert vi.ndim == 2
        assert vti.ndim == 2
        assert vt.ndim == 2
        assert v.ndim == 3
        assert vt.shape[-1] == 2
        assert v.shape[-1] == 3
        assert vi.shape[-1] == 3
        assert vti.shape[-1] == 3
        assert vti.shape[-2] == vi.shape[-2]

        if "all" in output_filters or (
            isinstance(output_filters, str) and output_filters == "all"
        ):
            output_filters = list(RENDERLAYER_OUTPUTS)

        unknown_filters = [f for f in output_filters if f not in RENDERLAYER_OUTPUTS]
        if len(unknown_filters) > 0:
            raise ValueError(
                "RenderLayer does not produce these outputs:", ",".join(unknown_filters)
            )

        # Compute camera-space 3D coordinates and 2D pixel-space projections.
        v_pix, v_cam = transform_with_v_cam(
            v,
            campos,
            camrot,
            focal,
            princpt,
            K,
            Rt,
            distortion_mode,
            distortion_coeff,
            fov,
        )

        with th.no_grad():
            index_img = rasterize(
                v_pix,
                vi,
                self.h,
                self.w,
            )

            mask = th.ne(index_img, -1)

        depth_img, bary_img = render_func(v_pix, vi, index_img)

        vt = vt[None].expand(v_pix.shape[0], -1, -1)

        vt_img = interpolate_func(2 * vt - 1.0, vti, index_img, bary_img).permute(
            0, 2, 3, 1
        )

        render_out = {
            "depth_img": depth_img,
            "vt_img": vt_img,
            "bary_img": bary_img,
            "mask": mask,
        }
        if vn is not None:
            vn_img = interpolate_func(vn, vi, index_img, bary_img).permute(0, 2, 3, 1)
            render_out["vn_img"] = vn_img

        if "render" in output_filters:
            image = F.grid_sample(tex, vt_img, **self.grid_sample_params)

            mf = mask[:, None].float()
            image = image * mf
            if background is not None:
                image = th.addcmul(image, background, 1.0 - mf)
            render_out["render"] = image

        render_out["v_pix"] = v_pix
        render_out["v_cam"] = v_cam
        render_out["index_img"] = index_img

        if "v_pix_img" in output_filters:
            render_out["v_pix_img"] = interpolate_func(v_pix, vi, index_img, bary_img)

        if "v_cam_img" in output_filters:
            render_out["v_cam_img"] = interpolate_func(v_cam, vi, index_img, bary_img)

        if "v_img" in output_filters:
            render_out["v_img"] = interpolate_func(v, vi, index_img, bary_img)

        if "vbn_img" in output_filters:
            vbnorms = vert_binormals(v, vt, vi.long(), vti.long())
            render_out["vbn_img"] = interpolate_func(vbnorms, vi, index_img, bary_img)

        if "vt_dxdy_img" in output_filters:
            render_out["vt_dxdy_img"] = screen_space_uv_derivative(
                v,
                vt,
                vi,
                vti,
                index_img,
                bary_img,
                mask,
                campos,
                camrot,
                focal,
                distortion_mode,
                distortion_coeff,
            )

        return {k: v for k, v in render_out.items() if k in output_filters}
