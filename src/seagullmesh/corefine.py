from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Tuple

from typing_extensions import Self

from seagullmesh import Mesh3, PropertyMap, Vertex, Edge, sgm, Face
from seagullmesh._seagullmesh import corefine


_PMaps = Tuple[
    PropertyMap[Edge, bool],
    PropertyMap[Vertex, Vertex],
    PropertyMap[Face ,Face],
]


@dataclass
class InputSpec:
    mesh: Mesh3
    edge_constrained: str | PropertyMap[Edge, bool] | None = None
    vertex_index: str | PropertyMap[Vertex, Vertex] | None = None
    face_index: str | PropertyMap[Face, Face] | None = None

    @contextmanager
    def get_or_temp_properties(self) -> Iterator[_PMaps]:
        m = self.mesh
        with (
            m.edge_data.get_or_temp(self.edge_constrained, '_temp_ecm', default=False) as ecm,
            m.vertex_data.get_or_temp(self.vertex_index, '_temp_vidx', default=m.null_vertex) as vidx,
            m.face_data.get_or_temp(self.face_index, '_temp_fidx', default=m.null_face) as fidx
        ):
            yield ecm, vidx, fidx


class Corefiner:
    def __init__(
            self,
            mesh0: Mesh3,
            mesh1: Mesh3,
            inplace=False,
            edge_constrained: str | None = None,
            vertex_index
    ):
        self.specs = InputSpec(mesh=mesh0), InputSpec(mesh=mesh1)
        self.output = mesh0 if inplace else Mesh3()



    def corefine(self, other: Mesh3) -> None:
        """Corefines the two meshes in place"""
        sgm.corefine.corefine(self._mesh, other._mesh)


    def union(self, other: Mesh3, inplace=False) -> Mesh3:
        """Corefines the two meshes and returns their boolean union"""

        sgm.corefine.union(self._mesh, other._mesh, out._mesh)
        return out


    def difference(self, other: Mesh3, inplace=False) -> Mesh3:
        """Corefines the two meshes and returns their boolean difference"""
        out = self if inplace else Mesh3(_Mesh3())
        sgm.corefine.difference(self._mesh, other._mesh, out._mesh)
        return out


    def intersection(self, other: Mesh3, inplace=False) -> Mesh3:
        """Corefines the two meshes and returns their boolean intersection"""
        out = self if inplace else Mesh3(_Mesh3())
        sgm.corefine.intersection(self._mesh, other._mesh, out._mesh)
        return out
