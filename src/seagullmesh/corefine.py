from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Tuple, List, Literal, Sequence

import numpy as np
from numpy import arange, unique
from seagullmesh._seagullmesh import corefine
from typing_extensions import Self

from seagullmesh import Mesh3, PropertyMap, Edge, Face, sgm, Faces


@dataclass
class _TrackSpec:
    idx: int
    mesh: Mesh3
    edge_constrained: str | PropertyMap[Edge, bool] = 'edge_is_constrained'
    face_origin: str | PropertyMap[Face, int] = 'face_origin'
    face_idx: str | PropertyMap[Face, int] = 'orig_face_idx'

    def realize(self):
        ecm = self.mesh.edge_data.get(self.edge_constrained, default=False)
        # Face origin defaults to -1 so original faces don't need to be updated
        face_origin = self.mesh.face_data.get(
            self.face_origin, default=-1, dtype='int64')

        face_idx = self.mesh.face_data.get(
            self.face_idx, default=-1, dtype='int64')

        # Todo: PMap[Indices<T>, int] could have a assign-index method
        face_idx[self.mesh.faces] = arange(self.mesh.n_faces)

        return _Tracked(self.idx, self.mesh, self, ecm, face_origin, face_idx)


@dataclass
class _Tracked:
    idx: int
    mesh: Mesh3
    spec: _TrackSpec
    edge_constrained: PropertyMap[Edge, bool]
    face_origin: PropertyMap[Face, int]
    face_idx: PropertyMap[Face, int]

    def to_tracker(self, tracker: corefine.CorefineTracker):
        tracker.track(self.mesh.mesh, self.idx, self.face_origin.pmap, self.face_idx.pmap)
        return self.mesh.mesh, self.edge_constrained.pmap

    @cached_property
    def faces(self) -> Faces:
        return self.mesh.faces

    @cached_property
    def face_origin_(self) -> np.ndarray:
        return self.face_origin[self.faces]

    @cached_property
    def face_idx_(self) -> np.ndarray:
        return self.face_idx[self.faces]


class Corefiner:
    def __init__(self, mesh0: Mesh3, mesh1: Mesh3, **kwargs):
        self._spec: list[_TrackSpec] = [_TrackSpec(0, mesh0), _TrackSpec(1, mesh1)]
        if kwargs:
            self.track(**kwargs)

    def track(self, mesh_idx: int | None = None, **kwargs) -> Self:
        mesh_idxs = range(2) if mesh_idx is None else (mesh_idx,)
        for i in mesh_idxs:
            self._spec[i] = replace(self._spec[i], **kwargs)
        return self

    def _apply(self, fn, *args):
        tracked0 = self._spec[0].realize()
        tracked1 = self._spec[1].realize()
        tracker = corefine.CorefineTracker()
        mesh0, ecm0 = tracked0.to_tracker(tracker)
        mesh1, ecm1 = tracked1.to_tracker(tracker)
        fn(mesh0, mesh1, ecm0, ecm1, tracker, *args)
        return Corefined([tracked0, tracked1])

    def corefine(self):
        return self._apply(corefine.corefine)

    def union(self):
        # Also needs to specify output
        return self._apply(corefine.union, self._spec[0].mesh.mesh)


class Corefined:
    def __init__(self, tracked: List[_Tracked]):
        self.tracked = tracked

    # def update_face_properties(self, mesh_idx: int, property_names: Sequence['str']):
    #     # TODO - can't default to face_data.keys() bc that would include face_idx and face_origin
    #     dest_tracked = self.tracked[mesh_idx]
    #
    #     for k in property_names:
    #         dest_pmap = dest_tracked.mesh.face_data[k]
    #
    #         for src_mesh_idx in unique(dest_tracked.face_origin_):
    #             i = dest_tracked.face_origin_ == mesh_idx
    #             dest_faces = dest_tracked.faces[]
    #
    #             if src_mesh_idx == mesh_idx:
    #                 dest_pmap.copy_values(dest_tracked.face_origin_[i], des)
    #             src_tracked = self.tracked[mesh_idx]
    #             src_pmap = src_tracked.mesh.face_data[k]
    #             pmap[tracked.faces[i]] = self.tracked[mesh_idx].mesh.
