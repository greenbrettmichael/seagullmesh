import pytest
from pyvista import Cylinder, PolyData

from seagullmesh import Mesh3, sgm

border = pytest.importorskip('seagullmesh._seagullmesh.border')


@pytest.fixture
def grid():
    return Mesh3.grid(3, 3)


def test_has_boundary(grid):
    assert grid.has_boundary()


def test_extract_boundary_cycles(grid):
    assert len(grid.extract_boundary_cycles()) == 1


def test_label_border_vertices(grid):
    assert set(grid.label_border_vertices('is_border')) == {False, True}


def test_label_border_edges(grid):
    assert set(grid.label_border_edges('is_border')) == {False, True}
