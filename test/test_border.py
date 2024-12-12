import pytest
from pyvista import Cylinder, PolyData

from seagullmesh import Mesh3, sgm

border = pytest.importorskip('seagullmesh._seagullmesh.border')


@pytest.fixture
def grid():
    return Mesh3.grid(3, 3)


def test_label_border_vertices(grid):
    assert grid.has_boundary()
