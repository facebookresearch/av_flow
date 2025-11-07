"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import copy
from typing import List

import os
import sys
import numpy as np
import torch as th
from attrdict import AttrDict
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, "../audio2photoreal/")
from visualize.ca_body.utils.train import load_checkpoint, load_from_config
from visualize.ca_body.utils.image import linear2displayBatch


class BodyRenderer(th.nn.Module):
    def __init__(
        self,
        subject="PXB184",
        render_rgb: bool = True,
    ):
        super().__init__()

        config_base = os.path.join(os.path.dirname(__file__), f"../../audio2photoreal/checkpoints/ca_body/data/{subject}")
        ckpt_path = f"{config_base}/body_dec.ckpt"
        config_path = f"{config_base}/config.yml"
        assets_path = f"{config_base}/static_assets.pt"

        config = OmegaConf.load(config_path)
        gpu = config.get("gpu", 0)
        self.device = th.device(f"cuda:{gpu}")

        static_assets = AttrDict(th.load(assets_path))

        self.model = load_from_config(config.model, assets=static_assets).to(
            self.device
        )
        self.model.cal_enabled = False
        self.model.pixel_cal_enabled = False
        self.model.learn_blur_enabled = False
        self.render_rgb = render_rgb
        if not self.render_rgb:
            self.model.rendering_enabled = None

        # Load model checkpoints
        print("loading...", ckpt_path)
        load_checkpoint(
            ckpt_path,
            modules={"model": self.model},
            ignore_names={"model": ["lbs_fn.*"]},
        )
        self.model.eval()
        self.model.to(self.device)

        self.default_inputs = th.load(os.path.join(os.path.dirname(__file__), f"../../audio2photoreal/assets/render_defaults_{subject}.pth"))

    def _render_loop(self, body_pose: np.ndarray, face: np.ndarray, only_head=False) -> List[np.ndarray]:
        all_rgb = []
        default_inputs_copy = copy.deepcopy(self.default_inputs)
        for b in tqdm(range(len(body_pose))):
            B = default_inputs_copy["K"].shape[0]
            if only_head:
                default_inputs_copy["lbs_motion"][:, 12:16] = (
                    th.tensor(body_pose[b : b + 1, :], device=self.device, dtype=th.float)
                    .tile(B, 1)
                    .to(self.device)
                )
            else:
                default_inputs_copy["lbs_motion"] = (
                    th.tensor(body_pose[b : b + 1, :], device=self.device, dtype=th.float)
                    .tile(B, 1)
                    .to(self.device)
                )
            face_codes = (
                th.from_numpy(face).float().cuda() if not th.is_tensor(face) else face
            )
            curr_face = th.tile(face_codes[b : b + 1, ...], (2, 1))
            default_inputs_copy["face_embs"] = curr_face
            preds = self.model(**default_inputs_copy)
            rgb0 = linear2displayBatch(preds["rgb"])[0]
            rgb1 = linear2displayBatch(preds["rgb"])[1]
            rgb = th.cat((rgb0, rgb1), axis=-1).permute(1, 2, 0)
            rgb = rgb.clip(0, 255).to(th.uint8)
            all_rgb.append(rgb.contiguous().detach().byte().cpu().numpy())
        return all_rgb
    
    def forward(self,
        body_pose,
        face,
        only_head=False,
    ):
        return self._render_loop(body_pose, face, only_head=only_head)
