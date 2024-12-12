import pytest

from seagullmesh import Mesh3

corefine = pytest.importorskip("seagullmesh._seagullmesh.geometry")

from .util import mesh


def test_volume(mesh: Mesh3):
    assert mesh.volume() > 0


def test_area(mesh: Mesh3):
    assert mesh.area() > 0


def test_bbox(mesh: Mesh3):
    _ = mesh.bounding_box()


def test_does_bound_a_volume(mesh: Mesh3):
    assert mesh.does_bound_a_volume()


def test_is_outward_oriented(mesh: Mesh3):
    assert mesh.is_outward_oriented()


