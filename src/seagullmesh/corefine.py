from __future__ import annotations

from dataclasses import dataclass, replace, field
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
    edge_is_constrained: str | PropertyMap[Edge, bool] = 'edge_is_constrained'
    face_mesh_map: str | PropertyMap[Face, int] = 'face_mesh_map'
    face_face_map: str | PropertyMap[Face, Face] = 'face_face_map'
    orig_face_properties: List[str] = field(init=False)

    def __post_init__(self):
        self.orig_face_properties = list(self.mesh.face_data.keys())

    def realize(self):
        ecm = self.mesh.edge_data.get(self.edge_is_constrained, default=False)
        # Face mesh map defaults to -1 so original faces don't need to be updated
        face_mesh_map = self.mesh.face_data.get(
            self.face_mesh_map, default=-1, dtype='int32')

        face_face_map = self.mesh.face_data.get(
            self.face_face_map, default=Mesh3.null_face)

        faces = self.mesh.faces
        face_face_map[faces] = faces

        return _Tracked(self.idx, self.mesh, self, ecm, face_mesh_map, face_face_map)


@dataclass
class _Tracked:
    idx: int
    mesh: Mesh3
    spec: _TrackSpec
    edge_is_constrained: PropertyMap[Edge, bool]
    face_mesh_map: PropertyMap[Face, int]
    face_face_map: PropertyMap[Face, Face]

    def to_tracker(self, tracker: corefine.CorefineTracker):
        tracker.track(self.mesh.mesh, self.idx, self.face_mesh_map.pmap, self.face_face_map.pmap)
        return self.mesh.mesh, self.edge_is_constrained.pmap

    @cached_property
    def faces(self) -> Faces:
        return self.mesh.faces

    @cached_property
    def face_mesh_idx(self) -> np.ndarray:
        return self.face_mesh_map[self.faces]


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

    def update_face_properties(self, mesh_idx: int, property_names: Sequence['str'] | None):
        dest = self.tracked[mesh_idx]
        prop_names = property_names or dest.spec.orig_face_properties

        for src_mesh_idx in unique(dest.face_mesh_idx):
            if src_mesh_idx == -1:  # unoriginal, untouched faces
                continue

            src = self.tracked[src_mesh_idx]
            i = dest.face_mesh_idx == src_mesh_idx

            new_faces = dest.faces[i]
            orig_faces = dest.face_face_map[new_faces]
            for k in prop_names:
                dest.mesh.face_data[k][new_faces] = src.mesh.face_data[k][orig_faces]
