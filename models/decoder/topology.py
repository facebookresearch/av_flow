"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations
from typing import cast, Dict, Optional, Tuple, Union

import igl
import numpy as np

__all__ = ["Topology"]


class Topology:
    def __init__(self, other: Optional[Topology] = None) -> None:
        """Default construct. Also copy constructor if other is not None."""
        self.v: np.ndarray = np.ndarray((0,), dtype=np.float32)
        self.vt: np.ndarray = np.ndarray((0,), dtype=np.float32)
        self.vi: np.ndarray = np.ndarray((0,), dtype=np.compat.long)
        self.vti: np.ndarray = np.ndarray((0,), dtype=np.compat.long)
        self.color: Optional[np.ndarray] = None
        self.new_to_old_vi: Optional[np.ndarray] = None
        self.new_to_old_vti: Optional[np.ndarray] = None
        if other is not None:
            self.v = other.v.copy()
            self.vt = other.vt.copy()
            self.vi = other.vi.copy()
            self.vti = other.vti.copy()
            if other.color is not None:
                self.color = other.color.copy()
            if other.new_to_old_vi is not None:
                self.new_to_old_vi = other.new_to_old_vi.copy()
            if other.new_to_old_vti is not None:
                self.new_to_old_vti = other.new_to_old_vti.copy()

    @classmethod
    def load_from_dict(
        cls, topology: Dict[str, np.ndarray], flip_uv: bool = False, normalize_uv: bool = False
    ) -> Topology:
        """Loads topology from dictionary."""
        instance = cls()
        instance.v, instance.vt, instance.vi, instance.vti = (
            topology["v"].copy(),
            topology["vt"].copy(),
            topology["vi"].copy(),
            topology["vti"].copy(),
        )

        if flip_uv:
            instance.vt[:, 1] = 1 - instance.vt[:, 1]

        if normalize_uv:
            instance.vt = instance.vt - instance.vt.min(axis=0, keepdims=True)
            instance.vt = instance.vt / instance.vt.max(axis=0, keepdims=True)

        if instance.v.shape[1] > 3:
            instance.color = instance.v[:, 3:]
            instance.v = instance.v[:, :3]
        else:
            instance.color = None

        instance.new_to_old_vi = None
        instance.new_to_old_vti = None
        return instance

    def get_dict(self) -> Dict[str, np.ndarray]:
        return {"v": self.v, "vi": self.vi, "vt": self.vt, "vti": self.vti}

    def correct_uvs_for_align_corners_false(self, uv_shape: Union[Tuple[int, int], int]) -> None:
        if isinstance(uv_shape, int):
            uv_shape = (uv_shape, uv_shape)
        uv_shape = np.asarray(uv_shape)

        b: np.ndarray = 0.5 / uv_shape
        a = (uv_shape - 1.0) / uv_shape
        self.vt = self.vt * a[None, ...] + b[None, ...]

    def get_vti_to_vi_map(self) -> np.ndarray:
        vlist = [set() for _ in self.vt]
        for (i, j, k), (ti, tj, tk) in zip(self.vi, self.vti):
            vlist[ti].add(i)
            vlist[tj].add(j)
            vlist[tk].add(k)
        vlist = [list(li) for li in vlist]
        max_uvs_per_vertex = max(len(li) for li in vlist)
        assert (
            max_uvs_per_vertex == 1
        ), "Two vertives have the same uv index. This should not happen. The opposite is possible"
        vlist = [li[0] for li in vlist]
        vti_to_vi = np.zeros((self.vt.shape[0]), dtype=np.int32)
        for e, item in enumerate(vlist):
            vti_to_vi[e] = item
        return vti_to_vi

    def fixup_hrgeo(self, size: int = 256) -> None:
        vt = self.vt * (size - 1)
        merge_vt_list = []

        def process_edge(axis, at):
            top_vertices = np.where(np.abs(vt[:, axis] - at) < 1e-6)[0]
            vt_top = vt[top_vertices]
            other_axis = 1 if axis == 0 else 0
            s = np.argsort(vt_top[:, other_axis])
            top_vertices = top_vertices[s]
            vt_top = vt[top_vertices]
            vt_top_rounded = np.round(vt_top[:, other_axis]).astype(dtype=np.int32)
            self.vt[top_vertices, other_axis] = vt_top_rounded / (size - 1)
            counts = np.bincount(vt_top_rounded)
            edges = np.cumsum(counts)
            edges = np.concatenate([[0], edges])
            for e0, e1 in zip(edges[:-1], edges[1:]):
                if e0 + 1 != e1:
                    merge_vt_list.append(np.sort(top_vertices[e0:e1]))

        process_edge(0, 0)
        process_edge(1, 0)
        process_edge(0, size - 1)
        process_edge(1, size - 1)

        def reduce_merge_list(merge_list):
            """
            Collapses overlapping merge lines
            """
            mem = {}
            new_list = []
            for item in merge_list:
                skip = False
                for e in item:
                    if e in mem:
                        k = mem[e]
                        new_list[k] = np.unique(np.concatenate([new_list[k], item]))
                        mem.update({j: k for j in item})
                        skip = True
                        break
                    mem[e] = len(new_list)
                if not skip:
                    new_list.append(item)
            return new_list

        merge_vt_list = reduce_merge_list(merge_vt_list)

        # map vti to vi. This assumes that no two uv share the same vertex
        vti_to_vi_map = {}
        for (i, j, k), (ti, tj, tk) in zip(self.vi, self.vti):
            vti_to_vi_map[ti] = i
            vti_to_vi_map[tj] = j
            vti_to_vi_map[tk] = k

        vec_vti_to_vi_map = np.vectorize(lambda i: vti_to_vi_map[i], otypes=[np.int32])
        merge_v_list = [vec_vti_to_vi_map(m) for m in merge_vt_list]

        merge_v_list = reduce_merge_list(merge_v_list)

        for item in merge_v_list:
            self.v[item[0]] = self.v[item].mean(axis=0, keepdims=True)

        if self.color is not None:
            color = self.color  # Helps beat the typing
            for item in merge_v_list:
                color[item[0]] = color[item].mean(axis=0, keepdims=True)

        merge_vt_list_m = {}
        for item in merge_vt_list:
            for e in item[1:]:
                merge_vt_list_m[e] = item[0]
        map_indices = np.vectorize(
            lambda i: merge_vt_list_m[i] if i in merge_vt_list_m else i,
            otypes=[np.int32],
        )
        self.vti = map_indices(self.vti)

        merge_v_list_m = {}
        for item in merge_v_list:
            for e in item[1:]:
                merge_v_list_m[e] = item[0]

        map_indices = np.vectorize(
            lambda i: merge_v_list_m[i] if i in merge_v_list_m else i, otypes=[np.int32]
        )
        self.vi = map_indices(self.vi)

        self.cleanup()

        assert len(self.vt) == size * size

    def cleanup(self) -> None:
        # remove degenerated triangles
        mask = np.logical_not(
            np.logical_or(
                np.logical_or(self.vi[:, 0] == self.vi[:, 1], self.vi[:, 0] == self.vi[:, 2]),
                self.vi[:, 1] == self.vi[:, 2],
            )
        )
        self.vi = self.vi[mask]
        self.vti = self.vti[mask]

        # shrink vt space (removes unused vt's)
        unigue_vti = np.unique(self.vti.flatten())
        self.vt = self.vt[unigue_vti]
        unigue_vti_inv = np.zeros(self.vti.max() + 1)
        unigue_vti_inv[unigue_vti] = np.arange(len(unigue_vti))

        map_indices = np.vectorize(lambda i: unigue_vti_inv[i], otypes=[np.int32])
        self.vti = map_indices(self.vti)

        # shrink v space (removes unused v's)
        unigue_vi = np.unique(self.vi.flatten())
        self.v = self.v[unigue_vi]
        if self.color is not None:
            self.color = self.color[unigue_vi]
        unigue_vi_inv = np.zeros(self.vi.max() + 1)
        unigue_vi_inv[unigue_vi] = np.arange(len(unigue_vi))

        map_indices = np.vectorize(lambda i: unigue_vi_inv[i], otypes=[np.int32])
        self.vi = map_indices(self.vi)

    def break_uv_seams(self) -> Optional[Topology]:
        """
        Returns new topology with where vertices with different uv's are split. Makes vi == vti
        """
        # find unique vertices
        m = set()
        for (i, j, k), (ti, tj, tk) in zip(self.vi, self.vti):
            m.add((i, ti))
            m.add((j, tj))
            m.add((k, tk))
        m = list(m)

        if len(m) == self.v.shape[0] == self.vt.shape[0]:
            # Nothing to do here. Return None to indicate that
            return None

        # forward mappings
        new_to_old_vi = np.asarray([x[0] for x in m], dtype=np.int32)
        new_to_old_vti = np.asarray([x[1] for x in m], dtype=np.int32)

        # reverse mappings
        reverse = {(i, it): e for e, (i, it) in enumerate(m)}

        map_indices = np.vectorize(lambda i, it: reverse[(i, it)], otypes=[np.int32])

        new = Topology(self)

        new.v = self.v[new_to_old_vi]
        new.vt = self.vt[new_to_old_vti]
        if new.color is not None:
            color = self.color
            assert color is not None
            new.color = color[new_to_old_vi]

        new.vti = new.vi = map_indices(self.vi, self.vti)

        # save mappings for future reference
        new.new_to_old_vi = new_to_old_vi
        new.new_to_old_vti = new_to_old_vti
        return new

    def compute_uv_map(self) -> np.ndarray:
        """
        Attempts to compute map vertex -> uv.
        Vertex may several uv coordinates. If that is the case, this will check if those uv's are the same.
        If they are not the same, and map vertex -> uv is not possible, this will throw an exception
        :return: map vertex index -> uv
        """
        if np.all(self.vi == self.vti):
            # Easy case!
            return self.vt
        elif self.v.shape[0] == self.vt.shape[0]:
            # try to see if vi <-> vti are bijective
            id_vi_vti = np.zeros(len(self.v), dtype=np.int32)
            id_vti_vi = np.zeros(len(self.v), dtype=np.int32)
            id_vi_vti[self.vi] = self.vti
            id_vti_vi[self.vti] = self.vi
            if np.all(id_vi_vti[id_vti_vi] == id_vti_vi) and np.all(
                id_vti_vi[id_vi_vti] == id_vi_vti
            ):
                raise NotImplementedError

        # try to do more complex stuff to check if uv's coincide and map is still possible.
        vlist = [set() for _ in self.v]
        for (i, j, k), (ti, tj, tk) in zip(self.vi, self.vti):
            vlist[i].add(ti)
            vlist[j].add(tj)
            vlist[k].add(tk)
        if all(len(x) == 1 for x in vlist):
            # Great! Each vertex has single uv
            vlist = np.asarray([x.pop() for x in vlist])
            return self.vt[vlist]
        vt = []
        for v_i in vlist:
            v_i = np.asarray(list(v_i), dtype=np.int32)
            if v_i.shape[0] == 0:
                print("Mesh is poisoned, contains unused vertices")
                continue
            vt_i = self.vt[v_i]
            if np.all(np.abs(vt_i.mean(axis=0)[None, ...] - vt_i) < 1e-6):
                vt.append(vt_i[0])
            else:
                # give up
                raise RuntimeError("Multiple different uv's per vertex")
        return np.asarray(vt, dtype=np.float32)

    def get_weight(self, threshold: Optional[float] = None) -> Optional[np.ndarray]:
        if self.color is None:
            return None
        weight = self.color.mean(1)
        if threshold is not None:
            weight[weight < threshold] = 0

        return weight

    def subdivide(self, n: int = 1) -> Optional[Topology]:
        try:
            # We have to cleanup, otherwise igl crashes. This will change topology.
            subdivided = Topology(self)
            subdivided.cleanup()
            # pyre-ignore
            v, vi = igl.loop(subdivided.v, subdivided.vi, n)
            if subdivided.color is not None:
                color = subdivided.color
                # pyre-ignore
                subdivided.color, _ = igl.upsample(color, subdivided.vi, n)
            subdivided.v = cast(np.ndarray, v)
            subdivided.vi = cast(np.ndarray, vi)
            # pyre-ignore
            subdivided.vt, subdivided.vti = igl.upsample(subdivided.vt, subdivided.vti, n)
            return subdivided
        except ImportError:
            print("Could not import igl. Can not subdivide mesh")
            return None
