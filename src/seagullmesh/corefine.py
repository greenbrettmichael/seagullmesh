from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence, Tuple

from pyvista.examples.cells import Vertex
from seagullmesh._seagullmesh import corefine
from typing_extensions import Self

from seagullmesh import Mesh3, PropertyMap, Edge, sgm, Vertices


class Corefiner:
    def __init__(
            self,
            mesh0: Mesh3,
            mesh1: Mesh3,
            edge_constrained0: str | PropertyMap[Edge, bool] | None = None,
            edge_constrained1: str | PropertyMap[Edge, bool] | None = None,
    ):
        self.sources: Tuple[Mesh3, Mesh3] = (mesh0, mesh1)
        self.edge_constrained = (edge_constrained0, edge_constrained1)

    @contextmanager
    def _inputs(self):
        mesh0, mesh1 = self.sources
        ecm0, ecm1 = self.edge_constrained
        with (
            mesh0.edge_data.get_or_temp(ecm0, '_temp_edge_is_constrained', default=False) as ecm0,
            mesh1.edge_data.get_or_temp(ecm1, '_temp_edge_is_constrained', default=False) as ecm1
        ):
            yield mesh0.mesh, mesh1.mesh, ecm0.pmap, ecm1.pmap

    def corefine(self) -> Corefined:
        with self._inputs() as (mesh0, mesh1, ecm0, ecm1):
            tracker = sgm.corefine.corefine(mesh0, mesh1, ecm0, ecm1)
            return Corefined(self, tracker)

    def union(self) -> Corefined:
        """Corefines the two meshes and returns their boolean union"""
        with self._inputs() as (mesh0, mesh1, ecm0, ecm1):
            tracker = sgm.corefine.union(mesh0, mesh1, mesh0, ecm0, ecm1)
            return Corefined(self, tracker)


class Corefined:
    def __init__(
            self,
            corefiner: Corefiner,
            tracker: corefine.CorefineTracker,
    ):
        self.corefiner = corefiner
        self.tracker = tracker

    def get_new_vertices(self, i: int) -> Vertices:
        mesh = self.corefiner.sources[i]
        return Vertices(mesh, self.tracker.get_new_vertices(i))

    def label_new_vertices(self, i: int, pmap: str | PropertyMap[Vertex, bool]) -> Self:
        pmap = self.corefiner.sources[i].vertex_data.get_or_create_property(pmap, default=False)
        new_verts = self.tracker.get_new_vertices(i)
        pmap[new_verts] = True
        return self

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
