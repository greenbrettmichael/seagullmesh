import pytest
import numpy as np

from seagullmesh import Mesh3

from .util import mesh

locate = pytest.importorskip('seagullmesh._seagullmesh.locate')


def test_locate(mesh: Mesh3):
    aabb_tree = mesh.aabb_tree()
    surf_pts = aabb_tree.first_ray_intersections(
        points=np.random.uniform(-1, 1, (10, 3)), directions=np.random.uniform(-1, 1, (10, 3)))
    pts = surf_pts.non_null().points
