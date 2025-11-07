"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Dict, Sequence

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import models.decoder.layers as la


class Decoder(nn.Module):
    def __init__(
        self,
        n_latent=256,
        n_vert_out=3 * 7306,
        tex_out_shp=(1024, 1024),
        tex_roi=((0, 0), (1024, 1024)),
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_vert_out = n_vert_out
        self.tex_roi = tex_roi
        self.tex_roi_shp = tuple([int(i) for i in np.diff(np.array(tex_roi), axis=0).squeeze()])
        self.tex_out_shp = tex_out_shp

        self.encmod = nn.Sequential(la.LinearWN(n_latent, 256), nn.LeakyReLU(0.2, inplace=True))
        self.geommod = nn.Sequential(la.LinearWN(256, n_vert_out))

        self.viewmod = nn.Sequential(la.LinearWN(3, 8), nn.LeakyReLU(0.2, inplace=True))
        self.texmod2 = nn.Sequential(
            la.LinearWN(256 + 8, 256 * 4 * 4), nn.LeakyReLU(0.2, inplace=True)
        )
        self.texmod = nn.Sequential(
            la.ConvTranspose2dWNUB(256, 256, 8, 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(256, 128, 16, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(128, 128, 32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(128, 64, 64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(64, 64, 128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(64, 32, 256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(32, 8, 512, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(8, 3, 1024, 1024, 4, 2, 1),
        )

        self.warpmod2 = nn.Sequential(
            la.LinearWN(256 + 8, 256 * 4 * 4), nn.LeakyReLU(0.2, inplace=True)
        )
        self.warpmod = nn.Sequential(
            la.ConvTranspose2dWN(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWN(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWN(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWN(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWN(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWN(64, 2, 4, 2, 1),
        )

        self.bias = nn.Parameter(th.zeros(3, self.tex_roi_shp[0], self.tex_roi_shp[1]))
        self.bias.data.zero_()

        ident_grid = identity_warp_grid(self.tex_roi_shp, align_corners=False)
        self.register_buffer("identity_warp", ident_grid[None], persistent=False)

        self.apply(lambda x: la.glorot(x, 0.2))
        la.glorot(self.texmod[-1], 1.0)
        la.glorot(self.warpmod[-1], 1.0)

        self.is_fused = False

        for l in self.warpmod:
            if hasattr(l, "weight_g"):
                with th.no_grad():
                    l.weight_g[:] /= 2

    def forward(self, inputs: Dict[str, th.Tensor], use_warp: bool = True):
        z = inputs["z"]
        view = inputs["view"]

        encout = self.encmod(z)
        geomout = self.geommod(encout)

        viewout = self.viewmod(view)
        encview = th.cat([encout, viewout], dim=1)
        texout = self.texmod(self.texmod2(encview).view(-1, 256, 4, 4))

        out = {"face_geom": geomout.view(geomout.shape[0], -1, 3)}

        if use_warp:
            warpdiff = self.warpmod(self.warpmod2(encview).view(-1, 256, 4, 4))
            warpdiff = F.interpolate(
                warpdiff, size=self.tex_roi_shp, mode="bilinear", align_corners=False
            )
            out["warpdiff"] = warpdiff

            warpdiff = warpdiff.permute(0, 2, 3, 1)
            if not self.is_fused:
                warpout = (
                    warpdiff * (1 / float(max(self.tex_out_shp[0], self.tex_out_shp[1])))
                    + self.identity_warp
                )
            else:
                warpout = warpdiff + self.identity_warp

            if self.training or not self.is_fused:
                warpout = warpout.clamp(-1e6, 1e6)
            texout = F.grid_sample(texout, warpout, align_corners=False)
            out["warpout"] = warpout
        texout = texout + self.bias[None]

        if self.tex_roi is not None:
            texout = F.pad(
                texout,
                [
                    self.tex_roi[0][1],
                    self.tex_out_shp[1] - self.tex_roi[1][1],
                    self.tex_roi[0][0],
                    self.tex_out_shp[0] - self.tex_roi[1][0],
                ],
            )

        out["face_tex"] = 255 * (texout + 0.5)
        return out

    def fuse(self):
        if self.is_fused:
            return
        self.warpmod[-1].weight.data /= max(self.tex_out_shp)
        self.warpmod[-1].bias.data /= max(self.tex_out_shp)
        self.is_fused = True

    def unfuse(self):
        if not self.is_fused:
            return
        self.warpmod[-1].weight.data *= max(self.tex_out_shp)
        self.warpmod[-1].bias.data *= max(self.tex_out_shp)
        self.is_fused = False


def identity_warp_grid(shape: Sequence[int], align_corners: bool = False) -> th.Tensor:
    """Computes the identity warp for tensors of a given shape. When used with
    `torch.nn.functional.grid_sample()` with the specified align_corners value,
    it will perform the identity warp on the input."""

    assert len(shape) in [2, 3]

    if len(shape) == 2:
        h, w = shape
        xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, w), np.linspace(-1.0, 1.0, h))
        if not align_corners:
            xgrid *= (w - 1) / w
            ygrid *= (h - 1) / h
        grid = th.from_numpy(np.stack([xgrid, ygrid], axis=-1).astype(np.float32))
    else:
        d, h, w = shape
        zgrid, ygrid, xgrid = np.meshgrid(
            np.linspace(-1.0, 1.0, d),
            np.linspace(-1.0, 1.0, h),
            np.linspace(-1.0, 1.0, w),
            indexing="ij",
        )
        if not align_corners:
            xgrid *= (w - 1) / w
            ygrid *= (h - 1) / h
            zgrid *= (d - 1) / d
        grid = th.from_numpy(np.stack((xgrid, ygrid, zgrid), axis=-1).astype(np.float32))
    return grid
