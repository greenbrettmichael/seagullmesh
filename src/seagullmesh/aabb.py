from __future__ import annotations

from typing import Tuple, Sequence

import numpy as np

from seagullmesh import (
    sgm, Mesh3, Vertex, Point2, Point3, Face, Faces, VertexPointMap
)

SurfacePoint = Tuple[Face, np.ndarray]
SurfacePoints = Tuple[Faces, np.ndarray]


class AabbTree:
    def __init__(self, mesh: Mesh3, aabb_tree: sgm.locate.aabb_treem):
        self.mesh = mesh
        self.aabb_tree = aabb_tree  # The c++ tree

    def locate_points(
            self, points: np.ndarray, vpm: str | VertexPointMap | None = None) -> SurfacePoints:
        """Given an array of points, locate the nearest corresponding points on the mesh
        Returns a list of face indices of length np, and an array (np, 3) of barycentric coordinates within those faces.
        """
        vpm = self.mesh.get_vertex_point_map(vpm)
        faces, bary_coords = sgm.locate.locate_points(
            self.mesh.mesh, self.aabb_tree, points, vpm.pmap)
        return Faces(self.mesh, faces), bary_coords

    def locate_point(self, point: Sequence[float], vpm: str | VertexPointMap | None = None) -> SurfacePoint:
        faces, bary_coords = self.locate_points(np.array([point]), vpm=vpm)
        return faces[0], bary_coords[0]

    def first_ray_intersections(self, points: np.ndarray, directions: np.ndarray) -> SurfacePoints:
        """Find the first intersections of the rays with the mesh

        The (n, 3) arrays `points` and `directions` define `n` rays.
        Returns a `n` list of faces and a (n, 3) array of barycentric coordinates in the
        corresponding faces.

        If a ray doesn't intersect the mesh, the face is equal to `self.null_face` and the
        barycentric coordinates are all zero.
        """
        faces, bary_coords = sgm.locate.first_ray_intersections(
            self.mesh, self.aabb_tree, points, directions)
        return Faces(self.mesh, faces), bary_coords

    def first_ray_intersection(self, point: np.ndarray, direction: np.ndarray) -> SurfacePoints:
        faces, bary_coords = self.first_ray_intersections(np.array([point], np.array([direction])))
        return faces[0], bary_coords[0]
