from __future__ import annotations

from typing import Tuple

import numpy as np

from seagullmesh import (
    sgm, Mesh3, Vertex, Point2, Point3, Faces, VertexPointMap
)

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
        surface_points = sgm.locate.locate_points(self.mesh.mesh, self.aabb_tree, points, vpm.pmap)
        return surface_points.faces, surface_points.bary_coords

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
        return faces, bary_coords
