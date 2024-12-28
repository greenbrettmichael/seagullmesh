from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, Sequence, NamedTuple

import numpy as np

from seagullmesh import (
    sgm, Mesh3, Vertex, Point2, Point3, Face, Faces, VertexPointMap
)


@dataclass
class SurfacePoints:
    faces: Faces
    bary_coords: np.ndarray

    def __getitem__(self, item) -> SurfacePoints | SurfacePoint:
        if isinstance(item, (int, np.integer)):
            return SurfacePoint(
                face=self.faces[item], bary_coords=self.bary_coords[item], faces=self.faces[item:(item + 1)])
        else:
            return SurfacePoints(face=self.faces[item], bary_coords=self.bary_coords[item])

    @cached_property
    def points(self) -> np.ndarray:
        if self.faces.is_null().any():
            raise ValueError("Null faces!")
        return self.faces.construct_points(bary_coords=self.bary_coords)

    def non_null(self) -> SurfacePoints:
        i = ~self.faces.is_null()
        return SurfacePoints(faces=self.faces[i], bary_coords=self.bary_coords[i])


@dataclass
class SurfacePoint:
    faces: Faces
    face: Face
    bary_coords: np.ndarray

    @cached_property
    def point(self) -> np.ndarray:
        if self.face == Mesh3.null_face:
            raise ValueError("Null face!")
        return self.faces.construct_points(np.array([self.bary_coords]))[0]


class AabbTree:
    def __init__(self, mesh: Mesh3, aabb_tree: sgm.locate.aabb_treem):
        self.mesh = mesh
        self.aabb_tree = aabb_tree  # The c++ tree

    def locate_points(
            self,
            points: np.ndarray,
            vpm: str | VertexPointMap | None = None,
    ) -> SurfacePoints:
        """Given an array of points, locate the nearest corresponding points on the mesh
        Returns a list of face indices of length np, and an array (np, 3) of barycentric coordinates within those faces.
        """
        vpm = self.mesh.get_vertex_point_map(vpm)
        faces, bary_coords = sgm.locate.locate_points(
            self.mesh.mesh, self.aabb_tree, points, vpm.pmap)
        return SurfacePoints(faces=Faces(self.mesh, faces), bary_coords=bary_coords)

    def locate_point(self, point: Sequence[float], vpm: str | VertexPointMap | None = None) -> SurfacePoint:
        surf_pts = self.locate_points(np.array([point]), vpm=vpm)
        return surf_pts[0]

    def first_ray_intersections(self, points: np.ndarray, directions: np.ndarray) -> SurfacePoints:
        """Find the first intersections of the rays with the mesh

        The (n, 3) arrays `points` and `directions` define `n` rays.
        Returns a `n` list of faces and a (n, 3) array of barycentric coordinates in the
        corresponding faces.

        If a ray doesn't intersect the mesh, the face is equal to `self.null_face` and the
        barycentric coordinates are all zero.
        """
        faces, bary_coords = sgm.locate.first_ray_intersections(
            self.mesh.mesh, self.aabb_tree, points, directions)
        return SurfacePoints(faces=Faces(self.mesh, faces), bary_coords=bary_coords)

    def first_ray_intersection(self, point: np.ndarray, direction: np.ndarray) -> SurfacePoint:
        surf_pts = self.first_ray_intersections(np.array([point]), np.array([direction]))
        return surf_pts[0]
