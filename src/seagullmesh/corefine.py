from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence, Tuple

from numpy import arange
from pyvista.examples.cells import Vertex
from seagullmesh._seagullmesh import corefine
from typing_extensions import Self

from seagullmesh import Mesh3, PropertyMap, Edge, Face, sgm, Vertices

from src.seagullmesh import Faces


class Corefiner:
    def __init__(
            self,
            mesh0: Mesh3,
            mesh1: Mesh3,
            edge_constrained0: str | PropertyMap[Edge, bool] = 'edge_is_constrained',
            edge_constrained1: str | PropertyMap[Edge, bool] = 'edge_is_constrained',
            face_idx0: str | PropertyMap[Face, int] = 'orig_face_idx',
            face_idx1: str | PropertyMap[Face, int] = 'orig_face_idx',
    ):
        self.sources: Tuple[Mesh3, Mesh3] = (mesh0, mesh1)
        self.edge_constrained = (edge_constrained0, edge_constrained1)
        self.face_idx = (face_idx0, face_idx1)

    def _get_inputs(self, i: int, tracker: corefine.CorefineTracker):
        mesh = self.sources[i]
        ecm = mesh.edge_data.get_or_create_property(self.edge_constrained[0], default=False)
        face_idx = mesh.face_data.get_or_create_property(self.face_idx[0], default=-1, dtype='int64')
        face_idx[mesh.faces] = arange(mesh.n_faces)
        tracker.track_mesh(mesh.mesh, face_idx.pmap)
        return mesh, ecm, face_idx

    def corefine(self) -> Corefined:
        tracker = corefine.CorefineTracker()
        mesh0, ecm0, face_idx0 = self._get_inputs(0, tracker)
        mesh1, ecm1, face_idx0 = self._get_inputs(1, tracker)
        sgm.corefine.corefine(mesh0, mesh1, ecm0, ecm1, tracker)
        return Corefined(self)  # TODO store pmap references

    def union(self) -> Corefined:
        """Corefines the two meshes and returns their boolean union"""
        with self._inputs() as (mesh0, mesh1, ecm0, ecm1):
            tracker = corefine.CorefineTracker(mesh0, mesh1)
            sgm.corefine.union(mesh0, mesh1, mesh0, ecm0, ecm1, tracker)
            return Corefined(self, tracker)


class Corefined:
    def __init__(self, corefiner: Corefiner):
        self.corefiner = corefiner

    def get_new_vertices(self, i: int) -> Vertices:
        mesh = self.corefiner.sources[i]
        return Vertices(mesh, self.tracker.get_new_vertices(i))

    def label_new_vertices(self, i: int, pmap: str | PropertyMap[Vertex, bool]) -> Self:
        pmap = self.corefiner.sources[i].vertex_data.get_or_create_property(pmap, default=False)
        new_verts = self.tracker.get_new_vertices(i)
        pmap[new_verts] = True
        return self

    def get_split_faces(self, i: int) -> Tuple[Faces, Faces]:
        mesh = self.corefiner.sources[i]
        old_faces, new_faces = self.tracker.get_split_faces(i, mesh.mesh)
        return Faces(mesh, old_faces), Faces(mesh, new_faces)

    def update_split_faces(self, i: int, property_names: Sequence[str] | None = None) -> Self:
        mesh = self.corefiner.sources[i]
        property_names = mesh.face_data.keys() if property_names is None else property_names
        old_faces, new_faces = self.tracker.get_split_faces(i, mesh.mesh)
        for k in property_names:
            mesh.face_data[k].pmap.copy_values(old_faces, new_faces)
        return self

    def update_copied_faces(self, property_names: Sequence[str] | None = None) -> Self:
        dest, src = self.corefiner.sources
        if property_names is None:
            property_names = set(src.face_data.keys()) & set(dest.face_data.keys())
        old_faces, new_faces = self.tracker.get_copied_faces(1)

        for k in property_names:
            dest.face_data[k][new_faces] = src.face_data[k][old_faces]
