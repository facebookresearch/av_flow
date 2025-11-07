"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Any, Dict, Mapping, Optional, Union
from abc import ABC, abstractmethod

import torch as th
from models.decoder.topology import Topology
from models.decoder.utils import get_v_vt_vi_vti

from models.decoder.render_layer import RenderLayer


class IFaceRenderer(th.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        inputs: Mapping[str, th.Tensor],
        campos: th.Tensor,
        camrot: th.Tensor,
        focal: th.Tensor,
        princpt: th.Tensor,
        **kwargs: Any,
    ) -> Dict[str, th.Tensor]:
        pass



class FaceRenderer(IFaceRenderer):
    def __init__(
        self,
        topology_obj_path: Union[str, Topology],
        normstats: Dict[str, th.Tensor],
        height: int,
        width: int,
        renderlayer_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            topology_obj_path: Path to the obj file containing the face mesh, or the Topology object
                directly
            normstats: A dictionary containing the mean and standard deviation of the neutral face
                geometry in order to recover the full scale geometry with:
                `geom = geom * self.vert_std + self.vert_mean`
            height: Height of the output image
            width: Width of the output image
            renderlayer_args: Optional dictionary of arguments to pass to the RenderLayer module
        """
        super().__init__()

        if renderlayer_args is None:
            renderlayer_args = {}

        _, vt, vi, vti = get_v_vt_vi_vti(topology_obj_path)

        self.vert_std: th.Tensor = normstats["vert_std"]
        self.register_buffer(
            "vert_mean", th.as_tensor(normstats["vert_mean"][None]).float(), persistent=False
        )
        self.register_buffer("vi", th.from_numpy(vi), persistent=False)

        self.rl = RenderLayer(h=height, w=width, vi=th.from_numpy(vi), vt=th.from_numpy(vt), vti=th.from_numpy(vti), **renderlayer_args)

    def forward(
        self,
        inputs: Mapping[str, th.Tensor],
        campos: th.Tensor,
        camrot: th.Tensor,
        focal: th.Tensor,
        princpt: th.Tensor,
        **kwargs: Any,
    ) -> Dict[str, th.Tensor]:
        geom = inputs["face_geom"]
        tex = inputs["face_tex"]

        geom = geom * self.vert_std + self.vert_mean

        return self.rl(geom, tex, campos, camrot, focal, princpt, **kwargs)
