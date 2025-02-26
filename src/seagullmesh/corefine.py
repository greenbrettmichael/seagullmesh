from __future__ import annotations

from dataclasses import dataclass, replace, field
from functools import cached_property
from typing import Tuple, List, Literal, Sequence

import numpy as np
from numpy import arange, unique
from seagullmesh._seagullmesh import corefine
from typing_extensions import Self

from seagullmesh import Mesh3, PropertyMap, Edge, Face, sgm, Faces, Vertex, Vertices


@dataclass
class _TrackSpec:
    idx: int
    mesh: Mesh3
    edge_is_constrained: str | PropertyMap[Edge, bool] | None = None
    face_mesh_map: str | PropertyMap[Face, int] | None = None
    face_face_map: str | PropertyMap[Face, Face] = None
    vert_mesh_map: str | PropertyMap[Vertex, int] | None = None
    orig_face_properties: List[str] = field(init=False)
    orig_edge_properties: List[str] = field(init=False)

    def __post_init__(self):
        self.orig_face_properties = list(self.mesh.face_data.keys())
        self.orig_face_properties = list(self.mesh.edge_data.keys())

    def realize(self, tracker: corefine.CorefineTracker):
        ecm = self.mesh.edge_data.get(
            self.edge_is_constrained or '_temp_edge_is_constrained', default=False)

        # Face mesh map defaults to -1 so original faces don't need to be updated
        face_mesh_map = self.mesh.face_data.get(
            self.face_mesh_map or '_temp_face_mesh_map', default=-1, dtype='int32')

        face_face_map = self.mesh.face_data.get(
            self.face_face_map or '_temp_face_face_map', default=Mesh3.null_face)

        # Default to -1 for original verts don't need to be updated
        vert_mesh_map = self.mesh.vertex_data.get(
            self.vert_mesh_map or '_temp_vert_mesh_map', default=-1, dtype='int32')

        # Initialize the face-face map to the identity function
        faces = self.mesh.faces
        face_face_map[faces] = faces

        # Register references with the C++ tracker
        tracker.track(
            self.mesh.mesh, self.idx, face_mesh_map.pmap,
            face_face_map.pmap, vert_mesh_map.pmap, ecm.pmap,
        )

        return _Tracked(self.idx, self.mesh, self, ecm, face_mesh_map, face_face_map, vert_mesh_map)


@dataclass
class _Tracked:
    idx: int
    mesh: Mesh3
    spec: _TrackSpec
    edge_is_constrained: PropertyMap[Edge, bool]
    face_mesh_map: PropertyMap[Face, int]
    face_face_map: PropertyMap[Face, Face]
    vert_mesh_map: PropertyMap[Vertex, int]

    @cached_property
    def faces(self) -> Faces:
        return self.mesh.faces

    @cached_property
    def vertices(self) -> Vertices:
        return self.mesh.vertices

    @cached_property
    def face_mesh_idx(self) -> np.ndarray:
        return self.face_mesh_map[self.faces]


class Corefiner:
    def __init__(self, mesh0: Mesh3, mesh1: Mesh3, **kwargs):
        self._spec: list[_TrackSpec] = [_TrackSpec(0, mesh0), _TrackSpec(1, mesh1)]
        if kwargs:
            self.track(**kwargs)

    def track(self, mesh_idx: int | Sequence[int] | None, **kwargs) -> Self:
        if mesh_idx is None:
            mesh_idxs = range(2)
        elif isinstance(mesh_idx, int):
            mesh_idxs = mesh_idx,
        else:
            mesh_idxs = mesh_idx

        for i in mesh_idxs:
            self._spec[i] = replace(self._spec[i], **kwargs)

        return self

    def _track(self) -> Tuple[corefine.CorefineTracker, list[_Tracked]]:
        tracker = corefine.CorefineTracker()
        tracked0 = self._spec[0].realize(tracker)
        tracked1 = self._spec[1].realize(tracker)
        return tracker, [tracked0, tracked1]

    def corefine(self):
        tracker, (t0, t1) = self._track()
        corefine.corefine(
            t0.mesh.mesh, t1.mesh.mesh,
            t0.edge_is_constrained.pmap,
            t1.edge_is_constrained.pmap,
            tracker
        )
        return Corefined([t0, t1])

    def _compute(self, fn, out: Mesh3 | None, **kwargs) -> Corefined:
        tracker, tracked = self._track()

        if out is None:
            out = tracked[0].mesh
        elif out not in (tracked[0].mesh, tracked[1].mesh):
            tracked_output = _TrackSpec(idx=2, mesh=out, **kwargs).realize(tracker=tracker)
            tracked.append(tracked_output)

        fn(
            tracked[0].mesh.mesh,
            tracked[1].mesh.mesh,
            tracked[0].edge_is_constrained.pmap,
            tracked[1].edge_is_constrained.pmap,
            tracker,
            out.mesh  # Also needs to specify output
        )
        return Corefined(tracked)

    def union(self, out: Mesh3 | None = None, **kwargs):
        return self._compute(corefine.union, out=out, **kwargs)

    def intersection(self, out: Mesh3 | None = None, **kwargs):
        return self._compute(corefine.intersection, out=out, **kwargs)

    def difference(self, out: Mesh3 | None = None, **kwargs):
        return self._compute(corefine.intersection, out=out, **kwargs)

    def clip(self, clip_volume: bool = False, use_compact_clipper: bool = False):
        # TODO doesn't need the ecms...
        tracker, tracked = self._track()
        corefine.clip(tracked[0].mesh.mesh, tracked[1].mesh.mesh, tracker, clip_volume, use_compact_clipper)
        return Corefined(tracked)


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

    def mark_new_vertices(self, mesh_idx: int, new_vertices: str | PropertyMap[Vertex, bool]):
        dest = self.tracked[mesh_idx]
        new_vertices = dest.mesh.vertex_data.get(new_vertices, default=False)
        is_new = dest.vert_mesh_map[dest.vertices] != -1
        new_vertices[dest.vertices[is_new]] = True

    def remove_temporary_properties(self, mesh_idx: int):
        spec = self.tracked[mesh_idx].spec
        if spec.edge_is_constrained is None:
            spec.mesh.edge_data.remove('_temp_edge_is_constrained')
        if spec.face_mesh_map is None:
            spec.mesh.face_data.remove('_temp_face_mesh_map')
        if spec.face_face_map is None:
            spec.mesh.face_data.remove('_temp_face_face_map')
        if spec.vert_mesh_map is None:
            spec.mesh.vertex_data.remove('_temp_vert_mesh_map')
