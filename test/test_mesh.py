import numpy as np
import pytest

from seagullmesh import Mesh3, Vertex, Vertices
from .util import mesh


def test_indices_from_vector(mesh):
    vs = mesh.vertices
    v0, v1 = vs[0], vs[1]
    idxs = Vertices(mesh, [v0, v1])
    assert np.all(idxs == vs[:2])


def test_index_indexing(mesh):
    idxs = mesh.vertices
    assert len(idxs) == mesh.n_vertices
    n = len(idxs)
    i = idxs[0]
    assert isinstance(i, Vertex)

    i_np = idxs[np.integer(0)]
    assert isinstance(i_np, Vertex) and i == i_np

    assert i.to_int() != mesh.null_vertex.to_int()
    assert i != mesh.null_vertex
    assert i < mesh.null_vertex
    assert i == i
    assert not (i != i)

    assert np.all(idxs == idxs)
    assert not np.any(idxs != idxs)

    assert np.sum(i == idxs) == 1
    assert np.sum(i != idxs) == (n - 1)
    assert np.all(idxs == idxs[np.arange(n)])
    assert np.all(idxs == idxs[np.ones(n, dtype=bool)])

    i0_repeated = idxs[np.zeros(n, dtype=int)]
    assert len(i0_repeated) == n
    assert np.all(i == i0_repeated)

    set_ = set(idxs)
    assert len(set_) == n
    assert i in set_


def test_vertex_points(mesh: Mesh3):
    pts = mesh.vertices.points()
    assert pts.shape == (mesh.n_vertices, 3)


def test_is_closed():
    assert Mesh3.icosahedron().is_closed is True
    assert Mesh3.grid(ni=2, nj=2).is_closed is False
    assert Mesh3.pyramid(closed=True).is_closed is True
    assert Mesh3.pyramid(closed=False).is_closed is False


def test_copy(mesh):
    copied = mesh.copy()
    assert copied is not mesh
    assert copied.mesh is not mesh.mesh
