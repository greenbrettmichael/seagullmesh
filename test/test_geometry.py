import pytest
import numpy as np
from seagullmesh import Mesh3
from seagullmesh.util import axis_angle
from .util import mesh


corefine = pytest.importorskip("seagullmesh._seagullmesh.geometry")


def test_volume(mesh: Mesh3):
    assert mesh.volume() > 0


def test_area(mesh: Mesh3):
    assert mesh.area() > 0


def test_bbox(mesh: Mesh3):
    _ = mesh.bounding_box()


def test_does_bound_a_volume(mesh: Mesh3):
    assert mesh.does_bound_a_volume is True


def test_is_outward_oriented(mesh: Mesh3):
    assert mesh.is_outward_oriented is True


@pytest.mark.parametrize('inplace', (False, True))
def test_transform(inplace: bool):
    orig = Mesh3.icosahedron()
    transform = axis_angle(axis=[.1, .2, .3], angle=np.pi/3, translate=[4, 5, 6])
    transformed = orig.transform(transform, inplace=inplace)
    if inplace:
        assert transformed is orig
    else:
        assert transformed is not orig


def test_face_normals(mesh: Mesh3):
    assert mesh.faces.normals().shape == (mesh.n_faces, 3)


def test_face_areas(mesh: Mesh3):
    assert mesh.faces.areas().shape == (mesh.n_faces,)


def test_vertex_normals(mesh: Mesh3):
    assert mesh.vertices.normals().shape == (mesh.n_vertices, 3)


def test_edge_lengths(mesh: Mesh3):
    assert mesh.edges.lengths().shape == (mesh.n_edges,)


def test_reverse_face_orientations(mesh: Mesh3):
    assert mesh.is_outward_oriented
    mesh.reverse_face_orientations()
    assert not mesh.is_outward_oriented
