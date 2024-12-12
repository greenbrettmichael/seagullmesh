from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

from seagullmesh._seagullmesh import corefine

from seagullmesh import Mesh3, PropertyMap, Edge, sgm


class Corefiner:
    def __init__(
            self,
            mesh0: Mesh3,
            mesh1: Mesh3,
            edge_constrained0: str | PropertyMap[Edge, bool] | None = None,
            edge_constrained1: str | PropertyMap[Edge, bool] | None = None,
    ):
        self.sources = (mesh0, mesh1)
        self.edge_constrained = (edge_constrained0, edge_constrained1)

    @contextmanager
    def _mesh_and_ecm(self, i: int):
        mesh = self.sources[i]
        ecm = self.edge_constrained[0]
        with mesh.edge_data.get_or_temp(ecm, '_temp_edge_is_constrained', default=False) as ecm:
            yield mesh.mesh, ecm

    def union(self) -> Corefined:
        """Corefines the two meshes and returns their boolean union"""
        output = self.sources[0]  # todo: non-inplace

        with (
            self._mesh_and_ecm(0) as (mesh0, ecm0),
            self._mesh_and_ecm(1) as (mesh1, ecm1)
        ):
            success, tracker = sgm.corefine.union(mesh0, mesh1, output.mesh, ecm0, ecm1)
            return Corefined(self, tracker, output)


class Corefined:
    def __init__(self, corefiner: Corefiner, tracker: corefine.CorefineTracker, output: Mesh3):
        self.corefiner = corefiner
        self.tracker = tracker
        self.output = output

    def update_corefined_faces(
            self,
            property_names: Sequence[str] | None = None,
        ):

        for i, mesh in enumerate(self.corefiner.sources):
            # Update source mesh
            prop_names = mesh.face_data.keys() if property_names is None else property_names
            old_faces, new_faces = self.tracker.get_split_faces(i, mesh.mesh)
            for k in prop_names:
                mesh.face_data[k].copy_values(old_faces, new_faces)

            # Update output mesh
            old_faces, new_faces = self.tracker.get_copied_faces(i)
            if mesh is self.output:
                mesh.face_data[k].copy_values(old_faces, new_faces)
            else:
                self.output.face_data[k][old_faces] = mesh.face_data[k][new_faces]
