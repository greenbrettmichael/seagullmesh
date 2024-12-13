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
            face_origin0: str | PropertyMap[Face, int] = 'face_origin',
            face_origin1: str | PropertyMap[Face, int] = 'face_origin',
            face_idx0: str | PropertyMap[Face, int] = 'orig_face_idx',
            face_idx1: str | PropertyMap[Face, int] = 'orig_face_idx',
    ):
        self.sources: Tuple[Mesh3, Mesh3] = (mesh0, mesh1)
        self.edge_constrained = (edge_constrained0, edge_constrained1)
        self.face_origin = (face_origin0, face_origin1)
        self.face_idx = (face_idx0, face_idx1)

    def _get_inputs(self, i: int, tracker: corefine.CorefineTracker):
        mesh = self.sources[i]
        assert isinstance(self.edge_constrained[i], str)
        ecm = mesh.edge_data.get_or_create_property(self.edge_constrained[i], default=False)
        face_origin = mesh.face_data.get_or_create_property(self.face_origin[i], default=i, dtype='int64')
        face_idx = mesh.face_data.get_or_create_property(self.face_idx[i], default=-1, dtype='int64')
        face_idx[mesh.faces] = arange(mesh.n_faces)
        tracker.track(mesh.mesh, i, face_origin.pmap, face_idx.pmap)
        return mesh, ecm, face_origin, face_idx

    def corefine(self):
        tracker = corefine.CorefineTracker()
        mesh0, ecm0, face_origin0, face_idx0 = self._get_inputs(0, tracker)
        mesh1, ecm1, face_origin1, face_idx1 = self._get_inputs(1, tracker)
        sgm.corefine.corefine(mesh0.mesh, mesh1.mesh, ecm0.pmap, ecm1.pmap, tracker)

    def union(self):
        tracker = corefine.CorefineTracker()
        mesh0, ecm0, face_origin0, face_idx0 = self._get_inputs(0, tracker)
        mesh1, ecm1, face_origin1, face_idx1 = self._get_inputs(1, tracker)
        output = mesh0
        sgm.corefine.union(mesh0.mesh, mesh1.mesh, ecm0.pmap, ecm1.pmap, tracker, output.mesh)
