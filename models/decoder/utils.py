"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

import torch as th
import numpy as np

from models.decoder.topology import Topology

ObjectType = Dict[str, Union[List[np.typing.NDArray], np.typing.NDArray]]
ArrayType = Union[List[Union[List, np.typing.NDArray]], np.typing.NDArray]


def get_v_vt_vi_vti(
    topology_or_path: Union[str, Topology]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A util function to build a mesh from either a Topology or a path to an obj file.

    Parameters:
        topology_or_path (Union[str, Topology]): The input can be either a string representing the
            path to an obj file or a Topology object.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following:
            - v: A numpy array of shape (n_vertices, 3) representing the 3D coordinates of the
                vertices.
            - vt: A numpy array of shape (n_texture_coordinates, 2) representing the 2D coordinates
                of the texture coordinates.
            - vi: A numpy array of shape (n_faces, 3) representing the indices of the vertices for
                each face.
            - vti: A numpy array of shape (n_faces, 3) representing the indices of the texture
                coordinates for each face.
    """
    if isinstance(topology_or_path, str):
        obj = load_obj(topology_or_path)
        return obj["v"][:, :3], obj["vt"][:, :2], obj["vi"][:, :3], obj["vti"][:, :3]
    elif isinstance(topology_or_path, Topology):
        return topology_or_path.v, topology_or_path.vt, topology_or_path.vi, topology_or_path.vti
    else:
        raise ValueError(
            f"topology_or_path must be a str or Topology, got {type(topology_or_path)}"
        )


def load_obj(path: Union[str, TextIO], return_vn: bool = False) -> ObjectType:
    """Load wavefront OBJ from file."""

    if isinstance(path, str):
        with open(path, "r") as f:
            lines: List[str] = f.readlines()
    else:
        lines: List[str] = path.readlines()

    v = []
    vt = []
    vindices = []
    vtindices = []
    vn = []

    for line in lines:
        if line == "":
            break

        if line[:2] == "v ":
            v.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "vt":
            vt.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "vn":
            vn.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "f ":
            vindices.append([int(entry.split("/")[0]) - 1 for entry in line.split()[1:]])
            if line.find("/") != -1:
                vtindices.append([int(entry.split("/")[1]) - 1 for entry in line.split()[1:]])

    if len(vt) == 0:
        assert len(vtindices) == 0, "Tried to load an OBJ with texcoord indices but no texcoords!"
        vt = [[0.5, 0.5]]
        vtindices = [[0, 0, 0]] * len(vindices)

    # If we have mixed face types (tris/quads/etc...), we can't create a
    # non-ragged array for vi / vti.
    mixed_faces = False
    for vi in vindices:
        if len(vi) != len(vindices[0]):
            mixed_faces = True
            break
    
    if mixed_faces:
        vi = [np.array(vi, dtype=np.int32) for vi in vindices]
        vti = [np.array(vti, dtype=np.int32) for vti in vtindices]
    else:
        vi = np.array(vindices, dtype=np.int32)
        vti = np.array(vtindices, dtype=np.int32)

    out = {
        "v": np.array(v, dtype=np.float32),
        "vn": np.array(vn, dtype=np.float32),
        "vt": np.array(vt, dtype=np.float32),
        "vi": vi,
        "vti": vti,
    }

    if return_vn:
        assert len(out["vn"]) > 0
        return out
    else:
        out.pop("vn")
        return out


def write_obj(path: str, *args: Any, **kwargs: Any) -> None:
    with open(path, "wb") as f:
        f.write(serialize_obj(*args, **kwargs))


def serialize_obj(
    v: ArrayType, vt: ArrayType, vi: ArrayType, vti: ArrayType, vn: Optional[ArrayType] = None
) -> bytes:
    """Write a Wavefront OBJ to a file.
    v: vertex list
    vt: UV list
    vi: face vertex index list
    vti: texture face UV index list
    vn: vertex normals (optional)
    """

    outstr = ""
    for xyz in v:
        outstr += " ".join(["v"] + [str(c) for c in xyz]) + "\n"
    if vn is not None:
        for nxyz in vn:
            outstr += " ".join(["vn"] + [str(c) for c in nxyz]) + "\n"
    for uv in vt:
        outstr += "vt {} {}\n".format(uv[0], uv[1])

    # With texture coordinate indices
    if vti is not None:
        for i, ti in zip(vi, vti):
            outstr += "f "
            for vind, vtind in zip(i, ti):
                outstr += "{}/{} ".format(vind + 1, vtind + 1)
            outstr += "\n"
    else:
        for i in vi:
            outstr += "f "
            for vind in i:
                outstr += "{} ".format(vind + 1)
            outstr += "\n"

    return outstr.encode()
