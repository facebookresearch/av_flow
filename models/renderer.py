"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import numpy as np
import torch as th

import models.decoder.face_vae as face_vae
from models.decoder.face_renderer import FaceRenderer
from utils.rotation import pad_3x4_to_4x4


DEFAULT_RT = th.Tensor(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0]]
)

DEFAULT_FOCAL = th.Tensor(
    [[10085.45,     0.00],
     [    0.00, 10082.49]]
)

DEFAULT_PRINCPT = th.Tensor(
    [2048.00, 1707.60]
)

DEFAULT_HEADPOSE = th.Tensor(
    [[1.0,  0.0, 0.0,    0.0],
     [0.0, -1.0, 0.0,    0.0],
     [0.0, 0.0, -1.0, 1000.0]]
)


def headrelative_krt(Rt, headpose):
    Rt = pad_3x4_to_4x4(Rt)
    headpose = pad_3x4_to_4x4(headpose)
    Rt = Rt @ headpose
    R = Rt[..., :3, :3]
    t = Rt[..., :3, 3]
    out = {}
    out["headrel_Rt"] = Rt
    out["headrel_cam_rot"] = R
    out["headrel_cam_pos"] = (R.transpose(-1, -2) @ -t[..., None])[..., 0]
    out["view"] = out["headrel_cam_pos"] / th.norm(out["headrel_cam_pos"], dim=-1, keepdim=True)
    return out


class Renderer(th.nn.Module):
    def __init__(
        self,
        checkpoint: str,
        vert_mean_file: str,
        vert_var_file: str,
        downsample_factor: int = 4,
    ):
        super().__init__()

        self.decoder = face_vae.Decoder()
        ckpt = th.load(checkpoint)
        self.decoder.load_state_dict(ckpt, strict=True)

        # Topology
        self.face_topology_file = os.path.join(os.path.dirname(__file__), "./face_topo_small_neck.obj")
        self.register_buffer("default_focal", DEFAULT_FOCAL)
        self.register_buffer("default_princpt", DEFAULT_PRINCPT)
        self.register_buffer("default_Rt", DEFAULT_RT)
        self.register_buffer("default_headpose", DEFAULT_HEADPOSE)

        self.downsample_factor = downsample_factor

        self.normstats = {
            "vert_mean": th.from_numpy(
                np.fromfile(vert_mean_file, dtype=np.float32).reshape(-1, 3)
            ),
            "vert_std": float(np.loadtxt(vert_var_file, dtype=np.float32) ** 0.5)
        }

        self.renderer = FaceRenderer(
            topology_obj_path=self.face_topology_file,
            normstats=self.normstats,
            height=int(4096 / downsample_factor) + int(4096 / downsample_factor) % 2,
            width=int(4096 / downsample_factor) + int(4096 / downsample_factor) % 2,
        )
 
    def render(self, z, headpose=None, Rt=None, focal=None, princpt=None):
        """
        z: B x 256 latent expression code
        headpose: B x 3 x 4 head rotation and translation
        return: B x 3 x height x width render
        """
        # Load default for optional argments that are not provided
        headpose = self.default_headpose.to(z.device) if headpose is None else headpose
        Rt = self.default_Rt.to(z.device) if Rt is None else Rt
        focal = self.default_focal.to(z.device) if focal is None else focal
        princpt = self.default_princpt.to(z.device) if princpt is None else princpt
        
        # Fix shape to have a batch dimension
        if len(Rt.shape) == 2:
            Rt = Rt.unsqueeze(0).expand(z.shape[0], -1, -1)
        if len(headpose.shape) == 2:
            headpose = headpose.unsqueeze(0).expand(z.shape[0], -1, -1)
        if len(focal.shape) == 2:
            focal = focal.unsqueeze(0).expand(z.shape[0], -1, -1)
        if len(princpt.shape) == 1:
            princpt = princpt.unsqueeze(0).expand(z.shape[0], -1)
        
        headrel_krt = headrelative_krt(Rt, headpose)

        # Decode
        dec = self.decoder({"z": z, "view": headrel_krt["view"]})
        geom, tex = dec["face_geom"], dec["face_tex"]

        # Render
        cam_params = {
            "campos": headrel_krt["headrel_cam_pos"],
            "camrot": headrel_krt["headrel_cam_rot"],
            "focal": focal / self.downsample_factor,
            "princpt": princpt / self.downsample_factor
        }
        render_out = self.renderer({"face_geom": geom, "face_tex": tex}, output_filters=["render"], **cam_params)
        render = th.clamp(render_out["render"], min=0, max=255)

        geom = geom * self.normstats["vert_std"] + self.normstats["vert_mean"].to(geom.device)

        return render, geom
    
    def forward(self, z, headpose=None, Rt=None, focal=None, princpt=None):
        return self.render(z, headpose, Rt, focal, princpt)
