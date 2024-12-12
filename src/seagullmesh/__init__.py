from __future__ import annotations

from abc import ABC
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING, Union, Sequence, TypeVar, overload, Tuple, \
    Generic, Iterator, Type, Dict, Literal, Callable

import numpy as np
from seagullmesh._seagullmesh.mesh import (
    Mesh3 as _Mesh3,
    Point2, Point3, Vector2, Vector3,
    Vertex, Face, Edge, Halfedge,
)
from typing_extensions import Self

from seagullmesh import _seagullmesh as sgm
from ._version import version_info, __version__  # noqa
from .skeleton import Skeleton

if TYPE_CHECKING:
    try:
        import pyvista as pv  # noqa
    except ImportError:
        pv = None

_IndexTypes = (Vertex, Face, Edge, Halfedge)
_IndexUnion = Vertex | Face | Edge | Halfedge
_IndicesTypes = (sgm.mesh.Vertices, sgm.mesh.Faces, sgm.mesh.Edges, sgm.mesh.Halfedges)
_CppIndicesUnion = sgm.mesh.Vertices | sgm.mesh.Faces | sgm.mesh.Edges | sgm.mesh.Halfedges
TIndex = TypeVar('TIndex', Vertex, Face, Edge, Halfedge)
TIndices = TypeVar('TIndices', sgm.mesh.Vertices, sgm.mesh.Faces, sgm.mesh.Edges, sgm.mesh.Halfedges)


class Indices(Generic[TIndex, TIndices], Sequence[TIndex]):
    index_type: Type[TIndex]  # Set by subclass
    indices_type: Type[TIndices]  # The C++ indices class, set by subclass

    _indices_types = sgm.mesh.Vertices | sgm.mesh.Faces | sgm.mesh.Edges | sgm.mesh.Halfedges

    # Updated in __init__subclass__
    _cpp_indices_to_py_indices: Dict[Type[_indices_types], Type[Indices]] = {}
    # _cpp_index_to_

    def __init__(self, mesh: Mesh3, indices: TIndices | Sequence[TIndex]):
        self.mesh = mesh

        if isinstance(indices, Indices._indices_types):
            if isinstance(indices, self.indices_type):
                self.indices = indices
            else:
                msg = f"Can't construct {type(self).__name__} from {type(indices.__name__)}"
                raise TypeError(msg)
        else:
            # A sequence e.g. list[Face] we can coerce into Indices<Face>?
            # Should raise a pybind11 type error otherwise
            self.indices = self.indices_type(indices)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        Indices._cpp_indices_to_py_indices[cls.indices_type] = cls

    @staticmethod
    def from_indices(mesh: Mesh3, indices: TIndices) -> Indices:
        return Indices._cpp_indices_to_py_indices[type(indices)](mesh, indices)

    @property
    def _array(self) -> np.ndarray:  # array of ints
        return self.indices.indices  # mapped to Indices.get_indices() on the C++ side

    def _with_array(self, arr: np.NDArray[np.uint32]) -> Self:
        indices = self.indices_type(arr)
        return type(self)(self.mesh, indices)

    def __repr__(self) -> str:
        return f'{self.indices_type.__name__}(n={len(self)})'

    def __len__(self) -> int:
        return len(self._array)

    def __eq__(self, other: TIndex | Indices[TIndex]) -> np.ndarray:
        # e.g. (these_faces == that_face) -> array[bool]
        if isinstance(other, self.index_type):
            return self._array == other.to_int()

        # e.g. (these_faces == those_faces) -> array[bool]
        elif isinstance(other, Indices) and (other.index_type is self.index_type):
            return self._array == other._array
        else:
            msg = f"Can only compare indices of the same type, got {self=} and {other=}"
            raise TypeError(msg)

    def __ne__(self, other: TIndex | Indices[TIndex]) -> np.ndarray:
        return np.logical_not((self == other))

    def __iter__(self) -> Iterator[TIndex]:
        for i in self._array:
            yield self.index_type(i)

    def copy(self) -> Self:
        return self._with_array(self._array.copy())

    @overload
    def __getitem__(self, item: int) -> TIndex: ...  # Vertex, Face, Edge, Halfedge

    @overload
    def __getitem__(self, item: slice | Sequence[int] | np.ndarray) -> Self: ...  # slice self to get another Self

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.index_type(self._array[item])  # Convert int to descriptor
        else:
            # As long as it can slice an array we're happy
            return self._with_array(self._array[item])

    @overload
    def __setitem__(self, item: int, value: TIndex): ...  # self.idxs[8] = some_vertex

    @overload
    def __setitem__(self, item: Sequence, value: Self): ...  # self.idxs[1:]] = self.idxs[1:][::-1]

    def __setitem__(self, item, value):
        if isinstance(item, int) and isinstance(value, self.index_type):
            # faces[0] = some_face
            self._array[item] = value.to_int()
        elif isinstance(value, type(self)):
            if value.mesh is self.mesh:
                self._array[item] = value._array
            else:
                raise ValueError("Assigning indices between different meshes, this is probably a mistake")
        else:
            raise TypeError("Can only assign indices of the same type")

    def unique(self) -> Self:
        return self._with_array(np.unique(self._array))


class Vertices(Indices[Vertex, sgm.mesh.Vertices]):
    index_type = Vertex
    indices_type = sgm.mesh.Vertices

    def adjacent_faces(self) -> Faces:
        return Faces(self.mesh, sgm.connected.vertices_to_faces(self.mesh.mesh, self.indices))

    def adjacent_edges(self) -> Edges:
        return Edges(self.mesh, sgm.connected.edges_to_faces(self.mesh.mesh, self.indices))

    def degrees(self) -> np.ndarray:
        return sgm.connected.vertex_degrees(self.mesh, self.indices)

    def points(self) -> np.ndarray:
        return self.mesh.vertex_point_map[self]

    def normals(self) -> np.ndarray:
        return sgm.geometry.vertex_normals(self.mesh.mesh, self.indices)


class Faces(Indices[Face, sgm.mesh.Faces]):
    index_type = Face
    indices_type = sgm.mesh.Faces

    def construct_points(
            self,
            bary_coords: np.ndarray,
            vert_points: str | PropertyMap[Vertex, Point2 | Point3] | None = None,
    ) -> np.ndarray:
        """Construct a set of points from face barycentric coordinates

        `bary_coords` must be of shape (len(faces), 3)

        By default, points are constructed using the default mesh vertex points. An optional vertex point map of value
        Point2 or Point3 can also be supplied. The returned array if of shape (len(faces), 2 or 3) as appropriate.
        """
        pmap = self.mesh.get_vertex_point_map(vert_points)
        return sgm.locate.construct_points(self.mesh, self.indices, bary_coords, pmap)

    def triangle_soup(self) -> np.ndarray:
        return sgm.io.triangle_soup(self.mesh.mesh, self.indices)  # TODO index, index_map

    def normals(self) -> np.ndarray:
        """(nf, 3) array of face normal vectors"""
        return sgm.geometry.face_normals(self.mesh.mesh, self.indices)

    def areas(self) -> np.ndarray:
        """(nf,) array of face areas"""
        return sgm.geometry.face_areas(self.mesh.mesh, self.indices)

    def adjacent_edges(self) -> Edges:
        return Edges(self.mesh, sgm.connected.faces_to_edges(self.mesh.mesh, self.indices))

    def adjacent_vertices(self) -> Vertices:
        return Vertices(self.mesh, sgm.connected.faces_to_vertices(self.mesh.mesh, self.indices))


class Edges(Indices[Edge, sgm.mesh.Edges]):
    index_type = Edge
    indices_type = sgm.mesh.Edges

    def edge_soup(self) -> np.ndarray:
        return sgm.io.edge_soup(self.mesh.mesh, self.indices)  # TODO index, index_map

    def lengths(self) -> np.ndarray:
        return sgm.geometry.edge_lengths(self.mesh.mesh, self.indices)


class Halfedges(Indices[Halfedge, sgm.mesh.Halfedges]):
    index_type = Halfedge
    indices_type = sgm.mesh.Halfedges


class Mesh3:
    null_vertex: Vertex = _Mesh3.null_vertex
    null_face: Face = _Mesh3.null_face
    null_edge: Edge = _Mesh3.null_edge
    # null_halfedge: Halfedge = _Mesh3.null_halfedge

    def __init__(self, mesh: _Mesh3 | None = None):
        """Construct a python-wrapped mesh"""

        """The C++ mesh object being wrapped"""
        self.mesh = mesh if mesh else _Mesh3()

        if hasattr(sgm, 'properties'):
            self.vertex_data = MeshData(self, Vertex, sgm.mesh.Vertices)
            self.face_data = MeshData(self, Face, sgm.mesh.Faces)
            self.edge_data = MeshData(self, Edge, sgm.mesh.Edges)
            self.halfedge_data = MeshData(self, Halfedge, sgm.mesh.Halfedges)

    n_vertices = property(lambda self: self.mesh.n_vertices)
    n_faces = property(lambda self: self.mesh.n_faces)
    n_edges = property(lambda self: self.mesh.n_edges)
    n_halfedges = property(lambda self: self.mesh.n_halfedges)
    is_valid = property(lambda self: self.mesh.is_valid)

    @property
    def vertices(self) -> Vertices:
        """Vector of vertex indices"""
        return Vertices(self, self.mesh.vertices)

    @property
    def faces(self) -> Faces:
        """Vector of face indices"""
        return Faces(self, self.mesh.faces)

    @property
    def edges(self) -> Edges:
        """Vector of edge indices"""
        return Edges(self, self.mesh.edges)

    @property
    def halfedges(self) -> Halfedges:
        """Vector of halfedge indices"""
        return Halfedges(self, self.mesh.halfedges)

    @property
    def has_garbage(self) -> bool:
        return self.mesh.has_garbage

    def collect_garbage(self) -> None:
        self.mesh.collect_garbage()

    @cached_property
    def vertex_point_map(self) -> PropertyMap[Vertex, Point3]:
        return self.vertex_data.find_property_map(
            sgm.properties.V_Point3_PropertyMap,
            name='v:point',
        )

    def iter_meshdata(self) -> Iterator[MeshData]:
        yield self.vertex_data
        yield self.face_data
        yield self.edge_data
        yield self.halfedge_data

    def copy(self) -> Mesh3:
        """Deep-copy the mesh and all its properties"""
        out = Mesh3(sgm.mesh.Mesh3(self.mesh))

        # The properties have been copied, just need to create new pmap references
        for src_data, dest_data in zip(self.iter_meshdata(), out.iter_meshdata()):
            for name, src_wrapper in src_data.items():
                wrapped = dest_data.find_property_map(
                    pmap_cls=type(src_wrapper.pmap),
                    name=name,
                    wrapper_cls=type(src_wrapper),
                    dtype_name=src_wrapper.dtype_name,
                )
                assert isinstance(wrapped, PropertyMap)
                dest_data[name] = wrapped

        return out

    def add(self, other: Mesh3, check_properties=False, inplace=False) -> Mesh3:
        if check_properties:
            for d_self, d_other in zip(self.iter_meshdata(), other.iter_meshdata()):
                d_self.check_has_same_properties(d_other)

        out = self if inplace else self.copy()
        out.mesh += other.mesh
        return out

    @staticmethod
    def from_polygon_soup(verts: np.ndarray, faces: np.ndarray, orient=True) -> Mesh3:
        """Constructs a surface mesh from vertices (nv * 3) and faces (nf * 3) arrays

        If `orient` is True (default), the faces are reindexed to represent a consistent manifold surface.
        """
        mesh = sgm.io.polygon_soup_to_mesh3(verts, faces, orient)
        return Mesh3(mesh)

    def to_polygon_soup(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns vertices (nv * 3) and faces (nf * 3) array"""
        return sgm.io.mesh3_to_polygon_soup(self.mesh)

    @staticmethod
    def from_file(filename: str) -> Mesh3:
        mesh = sgm.io.load_mesh_from_file(filename)
        return Mesh3(mesh)

    def to_file(self, filename: str):
        ext = Path(filename).suffix
        if ext == '.ply':
            sgm.io.write_ply(self.mesh, filename)
        elif ext == '.off':
            sgm.io.write_off(self.mesh, filename)
        else:
            raise ValueError(f"Unsupported format '{ext}'")

    @staticmethod
    def icosahedron(center: np.ndarray | Sequence[float] = (0, 0, 0), radius: float = 1.0) -> Mesh3:
        out = Mesh3()
        sgm.mesh.add_icosahedron(out.mesh, *center, radius)
        return out

    @staticmethod
    def pyramid(
            base_center: np.ndarray | Sequence[float] | Point3 = (0, 0, 0),
            n_base_pts: int = 4,
            height: float = 1.0,
            radius: float = 1.0,
            closed: bool = True,
    ):
        out = Mesh3()
        base_center = Point3(*base_center) if not isinstance(base_center, Point3) else base_center
        sgm.mesh.add_pyramid(out.mesh, n_base_pts, base_center, height, radius, closed)
        return out

    @staticmethod
    def grid(
            ni: int,
            nj: int,
            calculator: Callable[[int, int], Point3] = lambda i, j: Point3(i, j, 0),
            triangulated: bool = False,
    ):
        out = Mesh3()
        sgm.mesh.add_grid(out.mesh, ni, nj, calculator, triangulated)
        return out

    def estimate_geodesic_distances(
            self,
            src: Union[Vertex, Vertices],
            distance_prop: str | PropertyMap[Vertex, float],
    ) -> PropertyMap[Vertex, float]:
        """Estimates the geodesic distance from the source vertex/vertices to all vertices in the mesh

        Estimated distances are stored in the supplied vertex property map.
        """
        distances = self.vertex_data.get_or_create_property(distance_prop, default=0.0)
        self.mesh.estimate_geodesic_distances(distances.pmap, src)
        return distances

    def transform(self, transform: np.ndarray, inplace=False) -> Mesh3:
        out = self if inplace else self.copy()
        sgm.geometry.transform(out.mesh, transform)
        return out

    def volume(self) -> float:
        return sgm.geometry.volume(self.mesh)

    def area(self) -> float:
        return sgm.geometry.area(self.mesh)

    def bounding_box(self) -> sgm.mesh.BoundingBox3:
        return sgm.geometry.bounding_box(self.mesh)

    def does_bound_a_volume(self) -> bool:
        return sgm.geometry.does_bound_a_volume(self.mesh)

    def is_outward_oriented(self) -> bool:
        return sgm.geometry.is_outward_oriented(self.mesh)

    @staticmethod
    def from_pyvista(
            polydata: pv.PolyData,
            orient=True,
            vertex_data: Sequence[str] | Literal['all'] = (),
            face_data: Sequence[str] | Literal['all'] = (),
    ) -> Mesh3:
        """Converts a `pyvista.PolyData` object to a surface mesh.

        All point/cell data is ignored unless property names to copy are specified.
        """
        from vtkmodules.util.numpy_support import vtk_to_numpy
        cells = vtk_to_numpy(polydata.GetPolys().GetConnectivityArray())
        faces = cells.reshape(-1, 3)
        out = Mesh3.from_polygon_soup(polydata.points, faces, orient=orient)

        if vertex_data:
            keys = polydata.point_data.keys() if vertex_data == 'all' else vertex_data
            for k in keys:
                out.vertex_data[k] = polydata.point_data[k]

        if face_data:
            keys = polydata.cell_data.keys() if face_data == 'all' else face_data
            for k in keys:
                out.face_data[k] = polydata.cell_data[k]

        return out

    def to_pyvista(
            self,
            vertex_data: Literal['all'] | Sequence[str] = (),
            face_data: Literal['all'] | Sequence[str] = (),
    ) -> pv.PolyData:
        """Returns the mesh as a `pyvista.PolyData` object.

        By default, vertex and cell data is ignored -- specify vertex_data and cell_data as a list of keys
        naming property maps to copy, or the string 'all' for all of them.
        """
        import pyvista as pv
        verts, faces = sgm.io.mesh3_to_polygon_soup(self.mesh)
        mesh = pv.PolyData.from_regular_faces(verts, faces)

        if vertex_data:
            keys = self.vertex_data.keys() if vertex_data == 'all' else vertex_data
            vertices = self.vertices
            for k in keys:
                mesh.point_data[k] = self.vertex_data[k][vertices]

        if face_data:
            keys = self.face_data.keys() if face_data == 'all' else face_data
            faces = self.faces
            for k in keys:
                mesh.cell_data[k] = self.face_data[k][faces]

        return mesh

    def remesh(
            self,
            target_edge_length: float,
            n_iter: int = 1,
            protect_constraints=False,
            vertex_constrained: str | PropertyMap[Vertex, bool] = '_vcm',
            edge_constrained: str | PropertyMap[Edge, bool] = '_ecm',
            touched: str | PropertyMap[Vertex, bool] = '_touched',
            faces: Optional[Faces] = None,
            face_patch_map: Optional[PropertyMap, int] = None,
    ) -> None:
        """Perform isotropic remeshing on the specified faces (default: all faces).

        Vertices and edges can be constrained by setting their corresponding values in the
        vertex_constrained and edge_constrained maps to True. Constrained vertices cannot be
        modified during remeshing. Constrained edges *can* be split or collapsed, but not flipped,
        nor its endpoints moved. (If protect_constraints=True, constrained edges cannot be split or
        collapsed.

        If an optional `touched_map: PropertyMap[Vertex, bool]` mapping vertices to bools is
        specified, vertices that were created or moved during the remeshing are flagged as True.
        """
        faces = self.faces if faces is None else faces
        with (
            self.vertex_data.get_or_temp(vertex_constrained, temp_name='_vcm', default=False) as vcm,
            self.edge_data.get_or_temp(edge_constrained, temp_name='_ecm', default=False) as ecm,
            self.vertex_data.get_or_temp(touched, temp_name='_touched', default=False) as touched,
        ):
            args = [
                self.mesh, faces, target_edge_length, n_iter, protect_constraints, vcm.pmap, ecm.pmap, touched.pmap]

            if face_patch_map:
                sgm.meshing.uniform_isotropic_remeshing2(*args, face_patch_map.pmap)
            else:
                sgm.meshing.uniform_isotropic_remeshing(*args)

    def remesh_adaptive(
            self,
            edge_len_min_max: Tuple[float, float],
            tolerance: float,
            n_iter: int = 1,
            ball_radius: float = -1.0,
            protect_constraints=False,
            vertex_constrained: str | PropertyMap[Vertex, bool] = '_vcm',
            edge_constrained: str | PropertyMap[Edge, bool] = '_ecm',
            touched: str | PropertyMap[Vertex, bool] = '_touched',
            faces: Optional[Faces] = None,
            face_patch_map: Optional[PropertyMap, int] = None,
    ) -> None:
        """Isotropic remeshing with a sizing field adaptive to the local curvature.

        Smaller tolerance values lead to shorter edges.

        Ball radius is the radius over which to calculate the local curvature. If ball_radius == -1,
        the 1-ring of each vertex is used.

        For other parameter descriptions see `remesh`.
        """
        faces = self.faces if faces is None else faces
        with (
            self.vertex_data.get_or_temp(vertex_constrained, temp_name='_vcm', default=False) as vcm,
            self.edge_data.get_or_temp(edge_constrained, temp_name='_ecm', default=False) as ecm,
            self.vertex_data.get_or_temp(touched, temp_name='_touched', default=False) as touched,
        ):
            args = [self.mesh, faces, tolerance, ball_radius, edge_len_min_max,
               n_iter, protect_constraints, vcm.pmap, ecm.pmap, touched.pmap]
            if face_patch_map:
                sgm.meshing.adaptive_isotropic_remeshing2(*args, face_patch_map.pmap)
            else:
                sgm.meshing.adaptive_isotropic_remeshing(*args)

    def fair(self, verts: Vertices, continuity=0) -> None:
        """Fair the specified mesh vertices"""
        sgm.meshing.fair(self.mesh, verts, continuity)

    def refine(self, faces: Faces, density=np.sqrt(3)) -> Tuple[Vertices, Faces]:
        """Refine the specified mesh faces

        The number of faces is increased by a factor of `density`.
        Returns indices to the newly created vertices and faces.
        """
        return sgm.meshing.refine(self.mesh, faces, density)

    def smooth_angle_and_area(
            self,
            faces: Faces,
            n_iter: int,
            use_area_smoothing=True,
            use_angle_smoothing=True,
            use_safety_constraints=False,
            do_project=True,
            vertex_constrained: str | PropertyMap[Vertex, bool] = '_vcm',
            edge_constrained: str | PropertyMap[Edge, bool] = '_ecm',
    ) -> None:
        """Smooths a triangulated region of a polygon mesh

        This function attempts to make the triangle angle and area distributions as uniform as possible by moving
        (non-constrained) vertices.
        """
        with (
            self.vertex_data.get_or_temp(vertex_constrained, temp_name='_vcm', default=False) as vcm,
            self.edge_data.get_or_temp(edge_constrained, temp_name='_ecm', default=False) as ecm,
        ):
            sgm.meshing.smooth_angle_and_area(
                self.mesh, faces, n_iter, use_area_smoothing, use_angle_smoothing,
                use_safety_constraints, do_project, vcm.pmap, ecm.pmap)

    def tangential_relaxation(
            self,
            verts: Vertices,
            n_iter: int,
            relax_constraints=False,
            vertex_constrained: str | PropertyMap[Vertex, bool] = '_vcm',
            edge_constrained: str | PropertyMap[Edge, bool] = '_ecm',
    ) -> None:
        with (
            self.vertex_data.get_or_temp(vertex_constrained, temp_name='_vcm', default=False) as vcm,
            self.edge_data.get_or_temp(edge_constrained, temp_name='_ecm', default=False) as ecm,
        ):
            sgm.meshing.tangential_relaxation(
                self.mesh, verts, n_iter, relax_constraints, vcm.pmap, ecm.pmap)

    def smooth_shape(
            self,
            faces: Faces,
            time: float,
            n_iter: int,
            vertex_constrained: str | PropertyMap[Vertex, bool] = '_vcm',
    ) -> None:
        """Smooth the mesh shape by mean curvature flow

        A larger time step results in faster convergence but details may be distorted to a larger extent compared to
         more iterations with a smaller step. Typical values scale in the interval (1e-6, 1]
        """
        with self.vertex_data.get_or_temp(vertex_constrained, temp_name='_vcm', default=False) as vcm:
            sgm.meshing.smooth_shape(self.mesh, faces, time, n_iter, vcm.pmap)

    def does_self_intersect(self) -> bool:
        """Returns True if the mesh self-intersects"""
        return sgm.meshing.does_self_intersect(self.mesh)

    def self_intersections(self) -> Tuple[Faces, Faces]:
        """Returns pairs of intersecting faces"""
        return sgm.meshing.self_intersections(self.mesh)

    def remove_self_intersections(self) -> None:
        return sgm.meshing.remove_self_intersections(self.mesh)

    def get_vertex_point_map(self, vert_points: str | PropertyMap[Vertex, Point2 | Point3] | None = None):
        if vert_points is None:
            return self.mesh.points
        else:
            return self.vertex_data.get_property_map(vert_points).pmap

    def aabb_tree(
            self,
            vert_points: str | PropertyMap[Vertex, Point2 | Point3] | None = None,
    ) -> sgm.locate.AABB_Tree2 | sgm.locate.AABB_Tree3:
        """Construct an axis-aligned bounding box tree for accelerated point location by `Mesh3.locate_points

        By default, the AABB tree is constructed for the default mesh vertex locations, but also accepts a vertex
        property map storing Point2 or Point3 locations.
        """
        return sgm.locate.aabb_tree(self.mesh, self.get_vertex_point_map(vert_points))

    def locate_points(
            self,
            points: np.ndarray,
            aabb_tree=None,
            vert_points: str | PropertyMap[Vertex, Point2 | Point3] | None = None,
    ) -> Tuple[Faces, np.ndarray]:
        """Given an array of points, locate the nearest corresponding points on the mesh

        `aabb_tree` is an optional axis-aligned bounding box from Mesh3.aabb_tree. If the tree was constructed with a
        Point2 vertex property map, `points` is of shape (np, 2), otherwise (np, 3).

        Returns a list of face indices of length np, and an array (np, 3) of barycentric coordinates within those faces.
        """
        tree = aabb_tree or self.aabb_tree()
        pmap = self.get_vertex_point_map(vert_points)
        surface_points = sgm.locate.locate_points(self.mesh, tree, points, pmap)
        return surface_points.faces, surface_points.bary_coords

    def first_ray_intersections(
            self,
            aabb_tree: sgm.locate.AABB_Tree3,
            points: np.ndarray,
            directions: np.ndarray,
    ) -> Tuple[Faces, np.ndarray]:
        """Find the first intersections of the rays with the mesh

        The (n, 3) arrays `points` and `directions` define `n` rays.
        Returns a `n` list of faces and a (n, 3) array of barycentric coordinates in the
        corresponding faces.

        If a ray doesn't intersect the mesh, the face is equal to `self.null_face` and the
        barycentric coordinates are all zero.
        """
        surf_pts = sgm.locate.first_ray_intersections(self.mesh, aabb_tree, points, directions)
        return surf_pts.faces, surf_pts.bary_coords

    def shortest_path(
            self,
            src_face: Face,
            src_bc: np.ndarray,
            tgt_face: Face,
            tgt_bc: np.ndarray,
    ):
        """Constructs the shortest path between the source and target locations

        locations are specified as a face and barycentric coordinates
        """
        return sgm.locate.shortest_path(self.mesh, src_face, src_bc, tgt_face, tgt_bc)

    def lscm(self, uv_map: str | PropertyMap[Vertex, Point2], initial_verts: Tuple[Vertex, Vertex] = None) -> None:
        """Performs least-squares conformal mapping

        `initial_verts` are indices into the UV map whose coordinates have been fixed.
        Raises a seagullmesh.ParametrizationError if parametrization fails.

        """
        if isinstance(uv_map, str):
            uv_map = self.vertex_data.get_or_create_property(uv_map, default=Point2(0, 0))
        if initial_verts is not None:
            msg = sgm.parametrize.lscm(self.mesh, uv_map.pmap, *initial_verts)
        else:
            msg = sgm.parametrize.lscm(self.mesh, uv_map.pmap)

        if msg != "Success":
            raise ParametrizationError(msg)

    def arap(self, uv_map: str | PropertyMap[Vertex, Point2]) -> None:
        """Performs as-rigid-as-possible parameterization

        Raises a seagullmesh.ParametrizationError if parametrization fails.
        """
        uv_map = self.vertex_data.get_or_create_property(uv_map, default=Point2(0, 0))
        msg = sgm.parametrize.arap(self.mesh, uv_map.pmap)
        if msg != "Success":
            raise ParametrizationError(msg)

    def label_border_vertices(self, is_border: str | PropertyMap[Vertex, bool]):
        is_border = self.vertex_data.get_or_create_property(is_border, default=False)
        sgm.border.label_border_vertices(self.mesh, is_border.pmap)
        return is_border

    def label_border_edges(self, is_border: str | PropertyMap[Edge, bool]):
        is_border = self.edge_data.get_or_create_property(is_border, default=False)
        sgm.border.label_border_edges(self.mesh, is_border.pmap)
        return is_border

    def extract_boundary_cycles(self) -> Halfedges:
        return Halfedges(self, sgm.border.extract_boundary_cycles(self.mesh))

    def has_boundary(self) -> bool:
        return sgm.border.has_boundary(self.mesh)

    def trace_boundary_from_vertex(self, vertex: Vertex) -> Vertices:
        return sgm.border.trace_boundary_from_vertex(self.mesh, vertex)

    def remesh_planar_patches(
            self,
            edge_constrained: str | PropertyMap[Edge, bool] = '_ecm',
            # face_patch_map: FaceMap = '_face_map',
            cosine_of_maximum_angle: float = 1.0,
    ) -> Mesh3:
        with self.edge_data.get_or_temp(edge_constrained, temp_name='_ecm', default=False) as ecm:
            # fpm = self.face_data.get_or_create_property(face_patch_map, default=-1)
            # TODO see comments in c++ remesh_planar_patches regarding face_patch_map
            out = sgm.meshing.remesh_planar_patches(
                self.mesh, ecm.pmap, cosine_of_maximum_angle)

        return Mesh3(out)

    def edge_collapse(
            self,
            stop_policy_mode: Literal["face", "edge", "edge_length"],
            stop_policy_thresh: float | int,
            edge_constrained: str | PropertyMap[Edge, bool] = '_ecm',
    ) -> int:
        """Mesh simplification by edge collapse

        See https://doc.cgal.org/latest/Surface_mesh_simplification/index.html for details.

        If `stop_policy_mode` is 'face' or 'edge', 'stop_policy_thresh' is either an int or float.
        If an int, stops after the number of edges/faces drops below that threshold. If a float, the
        threshold indicates a ration in [0, 1], stopping when the number of edges/faces is below
        that ratio of the original number.

        If `stop_policy_mode` is 'edge_length', stop_policy_thresh must be a thresh, indicating
        the minimum edge length.

        Constrained edges can be indicated in a boolean edge property map `edge_constrained`.
        """
        if stop_policy_mode == 'face':
            if isinstance(stop_policy_thresh, float):
                fn = sgm.simplification.edge_collapse_face_count_ratio
            elif isinstance(stop_policy_thresh, int):
                fn = sgm.simplification.edge_collapse_face_count
            else:
                raise ValueError(f"Unsupported threshold type {type(stop_policy_thresh)}")
        elif stop_policy_mode == 'edge':
            if isinstance(stop_policy_thresh, float):
                fn = sgm.simplification.edge_collapse_edge_count_ratio
            elif isinstance(stop_policy_thresh, int):
                fn = sgm.simplification.edge_collapse_edge_count
            else:
                raise ValueError(f"Unsupported threshold type {type(stop_policy_thresh)}")
        elif stop_policy_mode == 'edge_length':
            fn = sgm.simplification.edge_collapse_edge_length
        else:
            raise ValueError(f"Unsupported stop policy mode {stop_policy_mode}")

        with self.edge_data.get_or_temp(edge_constrained, temp_name='_ecm', default=False) as ecm:
            out = fn(self.mesh, stop_policy_thresh, ecm.pmap)

        return out

    def skeletonize(self):
        """Construct the medial axis skeleton

        From [Triangulated Surface Mesh Skeletonization](
            https://doc.cgal.org/latest/Surface_mesh_skeletonization/index.html).

        Returns a `Skeleton` object with properties
          points : (n, 3) array of medial axis vertex positions
          edges : (m, 2) array of vertex indices
          vertex_map : dict[int, list[mesh vertex]] mapping skeleton vertices to mesh vertices
        """
        from seagullmesh.skeleton import Skeleton
        skel = sgm.skeletonization.extract_mean_curvature_flow_skeleton(self.mesh)
        return Skeleton(mesh=self, skeleton=skel)

    def interpolated_corrected_curvatures(
            self,
            ball_radius: float = -1,
            mean_curvature_map: str | PropertyMap[Vertex, float] = 'mean_curvature',
            gaussian_curvature_map: str | PropertyMap[Vertex, float] = 'gaussian_curvature',
            principal_curvature_map: str | PropertyMap[Vertex, sgm.properties.PrincipalCurvaturesAndDirections] = 'principal_curvature',
    ):
        mcm = self.vertex_data.get_or_create_property(mean_curvature_map, 0.0)
        gcm = self.vertex_data.get_or_create_property(gaussian_curvature_map, 0.0)
        pcm = self.vertex_data.get_or_create_property(
            principal_curvature_map, sgm.properties.PrincipalCurvaturesAndDirections())

        sgm.meshing.interpolated_corrected_curvatures(
            self.mesh, mcm.pmap, gcm.pmap, pcm.pmap, ball_radius)

    @staticmethod
    def from_poisson_surface_reconstruction(
            points: np.ndarray,
            normals: np.ndarray,
            spacing: float,
    ) -> Mesh3:
        mesh = sgm.poisson_reconstruct.reconstruct_surface(points, normals, spacing)
        return Mesh3(mesh)

    def alpha_wrapping(
            self,
            alpha: float | None,
            offset: float | None,
            relative_alpha: float = 20,
            relative_offset: float = 600,
    ) -> Mesh3:
        if alpha is None or offset is None:
            diagonal = self.bounding_box().diagonal()
            alpha = diagonal / relative_alpha if alpha is None else alpha
            offset = diagonal / relative_offset if offset is None else offset

        mesh = sgm.alpha_wrapping.wrap_mesh(self.mesh, alpha, offset)
        return Mesh3(mesh)

    @staticmethod
    def from_alpha_wrapping(
            points: np.ndarray,
            alpha: float | None,
            offset: float | None,
            relative_alpha: float = 20,
            relative_offset: float = 600,
    ):
        if alpha is None or offset is None:
            diagonal = _bbox_diagonal(points)
            alpha = diagonal / relative_alpha if alpha is None else alpha
            offset = diagonal / relative_offset if offset is None else offset

        mesh = sgm.alpha_wrapping.wrap_points(points, alpha, offset)
        return Mesh3(mesh)

    def triangulate_faces(self, faces: Faces | None = None):
        faces = self.faces if faces is None else faces
        sgm.triangulate.triangulate_faces(self.mesh, faces)

    def reverse_face_orientation(self, faces: Faces | None = None):
        faces = self.faces if faces is None else faces
        sgm.triangulate.reverse_face_orientations(self.mesh, faces)

    def regularize_face_selection_borders(
            self,
            is_selected: PropertyMap[Face, bool],
            weight: float,
            prevent_unselection: bool = False,
    ):
        sgm.border.regularize_face_selection_borders(self.mesh, is_selected, weight, prevent_unselection)

    def label_selected_face_patches(self, faces: Faces, face_patch_idx: PropertyMap[Face, int] | str):
        # faces not in faces are labeled face_patch_idx=0, otherwise 1 + the index of the patch of selected regions
        face_patch_idx = self.face_data.get_or_create_property(face_patch_idx, default=0, is_index=True)
        sgm.connected.label_selected_face_patches(self.mesh, faces, face_patch_idx.pmap)
        return face_patch_idx

    def label_connected_components(self, face_patches: PropertyMap[Face, int], edge_is_constrained: PropertyMap[Edge, bool]) -> int:
        return sgm.connected.label_connected_components(self.mesh, face_patches.pmap, edge_is_constrained.pmap)

    def remove_connected_face_patches(
            self,
            to_remove: Sequence[int | bool],
            face_patches: PropertyMap[Face, int | bool],
    ):
        sgm.connected.remove_connected_face_patches(self.mesh, to_remove, face_patches.pmap)

    def connected_component(
            self, seed_face: Face, edge_is_constrained: PropertyMap[Edge, bool] | str = '_ecm') -> Faces:
        with self.face_data.get_or_temp(edge_is_constrained, tempname='_ecm', default=False) as ecm:
            return sgm.connected.connected_component(self.mesh, seed_face, ecm.pmap)


class ParametrizationError(RuntimeError):
    pass


def _bbox_diagonal(points: np.ndarray):
    x0, y0, z0 = points.min(axis=0)
    x1, y1, z1 = points.max(axis=0)
    return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)


Key = TypeVar('Key', Vertex, Face, Edge, Halfedge)
Val = TypeVar('Val', int, bool, float, Point2, Point3, Vector2, Vector3)

"""Something that can vector-index a property map"""
IntoIndices = np.ndarray | slice | Sequence[Key]


class PropertyMap(Generic[Key, Val], ABC):
    def __init__(self, pmap, data: MeshData[Key], dtype: str):
        self._pmap = pmap  # the C++ object
        self._data = data
        self._dtype_name = dtype

    def __str__(self) -> str:
        return f'PropertyMap[{self._data.key_type.__name__}, {self.dtype_name}]'

    @property
    def pmap(self):
        return self._pmap

    @property
    def data(self) -> MeshData[Key]:
        return self._data

    @property
    def dtype_name(self) -> str:
        return self._dtype_name

    @property
    def key_t(self) -> Type[TIndex]:
        return self._data.key_type

    @property
    def indices_t(self) -> type:
        return self._data.indices_t

    def _to_cpp_indices(self, key: IntoIndices[Key]) -> _CppIndicesUnion:
        # Returns the C++ indexer -- sgm.mesh.Vertices, sgm.mesh.Faces, etc
        if isinstance(key, Indices):
            if key.index_type is self.key_t:
                return key.indices
            else:
                msg = f'Tried to index a {self.key_t} property map with {key.index_type} indices'
                raise TypeError(msg)

        # Two possibilities:
        try:
            # 1) list[key]
            return sgm.mesh.make_indices(key)
        except TypeError:
            # 2) index into indices, like a ndarray[bool]
            return self._data.all_indices[key].indices

    @overload
    def __getitem__(self, key: int | Key) -> Val: ...

    @overload
    def __getitem__(self, key: IntoIndices[Key]) -> np.ndarray: ...

    def __getitem__(self, key):
        if isinstance(key, int):  # e.g. pmap[0] -> value for the first face
            return self.pmap[self._data.all_indices[key]]
        elif isinstance(key, _IndexTypes):  # e.g. pmap[Face] -> scalar value
            if isinstance(key, self.key_t):
                return self.pmap[key]
            else:
                msg = f'Tried to index a {self.key_t} property map with {type(key)} index'
                raise TypeError(msg)
        else:
            return self.pmap[self._to_cpp_indices(key)]

    @overload
    def __setitem__(self, key: int | Key, val: Val) -> None: ...

    @overload
    def __setitem__(self, key: IntoIndices[Key], val: np.ndarray | Sequence[Val]) -> None: ...

    def __setitem__(self, key, val) -> None:
        if isinstance(key, int):  # pmap[0] -> value for the first face
            self.pmap[self._data.all_indices[key]] = val
        elif isinstance(key, _IndexTypes):
            if isinstance(key, self.key_t):
                self.pmap[key] = val
            else:
                msg = f'Tried to index a {self.key_t} property map with {type(key)} index'
                raise TypeError(msg)
        else:
            self.pmap[self._to_cpp_indices(key)] = val

    for dunder in (
            '__add__',
            '__eq__',
            '__ge__',
            '__gt__',
            '__le__',
            '__lt__',
            '__mul__',
            '__ne__',
            '__neg__',
            '__pos__',
            '__pow__',
            '__mod__',
    ):
        def _dunder_impl(self, other, _dunder=dunder):
            if isinstance(other, PropertyMap):
                other = other[:]
            fn = getattr(self[:], _dunder)
            return fn(other)

        locals()[dunder] = _dunder_impl


class ScalarPropertyMap(PropertyMap[Key, Val]):
    pass


class ArrayPropertyMap(PropertyMap[Key, Val]):
    def get_objects(self, key: IntoIndices[Key]) -> Sequence[Val]:
        # __getitem__ defaults to returning array(nk, ndim), also allow returning list[Point2]
        return self.pmap.get_vector(self._to_cpp_indices(key))

    def set_objects(self, key: IntoIndices[Key], val: Sequence[Val]) -> None:
        # handled by the general case
        self[key] = val


_PMapDType = str | np.dtype | type


class MeshData(Generic[Key]):
    def __init__(
            self,
            mesh: Mesh3,
            key_type: Type[TIndex],
            indices_t: type,
    ):
        self._data: Dict[str, PropertyMap[Key]] = {}
        self.mesh = mesh  # python wrapped mesh
        self._key_name = indices_t.__name__.lower()  # 'vertices', 'faces', 'edges', 'halfedges'
        self._prefix = self._key_name[0].upper()  # 'V', 'F', 'E', 'H'
        self.indices_t = indices_t
        self.key_type = key_type

    _dtype_mappings: dict[type, str] = {
        float: 'double',
        int: 'int64',
    }

    def _dtype_name(self, dtype: _PMapDType) -> str:
        if isinstance(dtype, str):
            return dtype
        elif isinstance(dtype, np.dtype):
            if dtype.name == 'float64':
                return 'double'
            return dtype.name
        elif mapped := self._dtype_mappings.get(dtype):
            return mapped
        else:
            return dtype.__name__

    def _pmap_class_name(self, dtype_name: str) -> str:
        return f'{self._key_name[0].upper()}_{dtype_name}_PropertyMap'

    @property
    def all_indices(self) -> Indices[Key]:
        # e.g. mesh.faces
        return getattr(self.mesh, self._key_name)

    @property
    def n_mesh_keys(self) -> int:
        # e.g. mesh.n_faces
        return getattr(self.mesh, f'n_{self._key_name}')

    @contextmanager
    def temp(
            self,
            name: str,
            default: Val,
            dtype: _PMapDType | None = None,
    ) -> Iterator[PropertyMap[Key, Val]]:
        """Create a temporary property map and remove it after exiting the contextmanager"""
        pmap = self.add_property(name=name, default=default, dtype=dtype)
        yield pmap
        self.remove_property(name)

    @contextmanager
    def get_or_temp(
            self,
            pmap: str | PropertyMap[Key, Val] | None,
            temp_name: str,
            default: Val,
            dtype: _PMapDType | None = None,
    ) -> Iterator[PropertyMap[Key, Val]]:
        """Construct a placeholder property map

        If pmap: str == temp_name, the map is removed after use. If pmap is another str or a
        pre-existing property map, it's not removed.
        """
        remove_after = isinstance(pmap, str) and pmap == temp_name
        pmap = self.get_or_create_property(pmap, default=default, dtype=dtype)
        yield pmap
        if remove_after:
            self.remove_property(temp_name)

    def add_property(
            self,
            name: str,
            default: Val,
            dtype: str | np.dtype | None = None,
    ) -> PropertyMap[Key, Val]:
        """Add a property map

        The type of the map's value is inferred from the default value. E.g.
        `mesh.vertex_data.add_property('foo', 0.0)` constructs a property map named 'foo'
        storing doubles on vertices and `mesh.face_data.add_property('bar', Point2(0, 0))` stores
        2d points on mesh faces.

        Mapping int-values to C++ types is inherently ambiguous, so it's preferred to explicitly
        specify a dtype, either as a string like 'uint32' or a numpy dtype.
        """
        dtype_name = self._dtype_name(dtype or type(default))
        cls_name = self._pmap_class_name(dtype_name)

        try:
            pmap_class: type = getattr(sgm.properties, cls_name)
        except AttributeError:
            msg = (
                f"Property map class {cls_name} does not exist for default = "
                f"{type(default)}({default}) with supplied {dtype=} and inferred {dtype_name=} "
                f"during attempted construction of property name {name=}"
            )
            raise TypeError(msg)

        pmap = pmap_class(self.mesh.mesh, name, default)
        return self.assign_property_map(name=name, pmap=pmap, dtype_name=dtype_name)  # The wrapped map

    def remove_property(self, key: str):
        pmap = self._data.pop(key)
        sgm.properties.remove_property_map(self.mesh.mesh, pmap.pmap)

    def find_property_map(
            self,
            pmap_cls: type,
            name: str,
            wrapper_cls: Type[PropertyMap] | None = None,
            dtype_name: str = 'unknown',
    ) -> PropertyMap[Key]:
        # Finds a pre-existing pmap, wraps it, and returns it without adding it to self
        pmap = pmap_cls.get_property_map(self.mesh.mesh, name)  # noqa
        if pmap is None:
            raise KeyError(f"Property map {pmap_cls} {name} doesn't exist")
        return self.wrap_property_map(pmap, wrapper_cls=wrapper_cls, dtype_name=dtype_name)

    def wrap_property_map(
            self,
            pmap,  # the c++ class,
            wrapper_cls: Type[PropertyMap] | None = None,
            dtype_name: str = 'unknown',
    ) -> PropertyMap[Key]:
        if wrapper_cls is None:
            wrapper_cls = ScalarPropertyMap if pmap.is_scalar else ArrayPropertyMap
        return wrapper_cls(pmap=pmap, data=self, dtype=dtype_name)

    def assign_property_map(
            self,
            name: str,
            pmap,  # The C++ property map
            wrapper_cls: Type[PropertyMap] | None = None,
            dtype_name: str = 'unknown',
    ) -> PropertyMap:
        wrapped_pmap = self._data[name] = self.wrap_property_map(
            pmap, wrapper_cls, dtype_name=dtype_name)
        return wrapped_pmap

    def get_property_map(self, key: str | PropertyMap[Key, Val]) -> PropertyMap[Key, Val]:
        if isinstance(key, PropertyMap):
            return key

        return self._data[key]

    def get_or_create_property(
            self,
            key: str | PropertyMap[Key, Val],
            default: Val,
            dtype: _PMapDType | None = None,
    ) -> PropertyMap[Key, Val]:
        if isinstance(key, PropertyMap):
            return key

        if key in self._data:
            return self._data[key]
        else:
            return self.add_property(name=key, default=default, dtype=dtype)

    def __getitem__(self, item: str) -> PropertyMap[Key, Any]:
        return self._data[item]

    def __delitem__(self, item: str):
        self.remove_property(item)

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, PropertyMap):  # An already wrapped pmap
            if value.data is not self:
                raise TypeError("Property map does not belong to this mesh")
            self._data[key] = value
            return

        if hasattr(value, "is_sgm_property_map"):
            # Assigning the bare C++ property map
            self.assign_property_map(name=key, pmap=value)
            return

        # Implicit construction of a new property map with initial value(s) `value`
        default = np.zeros_like(value, shape=()).item()
        pmap = self.get_or_create_property(key, default)
        pmap[self.all_indices] = value

    def items(self) -> Iterator[Tuple[str, PropertyMap[Key, Any]]]:
        yield from self._data.items()

    def values(self) -> Iterator[PropertyMap[Key, Any]]:
        yield from self._data.values()

    def keys(self) -> Iterator[str]:
        yield from self._data.keys()

    def __iter__(self) -> Iterator[str]:
        yield from self._data.__iter__()

    def check_has_same_properties(self, other: Self) -> None:
        if missing_keys := set(self.keys()).symmetric_difference(other.keys()):
            msg = f"{self.key_type} properties {missing_keys} are not present in both meshes."
            raise ValueError(msg)

        for k, pmap in self.items():
            t0, t1 = type(pmap.pmap), type(other[k].pmap)
            if t0 is not t1:
                raise TypeError(f"Property {k} has two different types: {t0} and {t1}")
