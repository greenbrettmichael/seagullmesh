from numpy import arange, ones

from seagullmesh import sgm, Mesh3, Indices, Vertex
import numpy as np


def test_indices_from_vector():
    mesh = Mesh3.icosahedron()
    vs = mesh.vertices
    v0, v1 = vs[0], vs[1]
    idxs = Indices.collect(Vertex, [v0, v1])
    assert (idxs == vs[:2]).all()


def test_index_indexing():
    mesh = Mesh3.icosahedron()
    idxs = mesh.vertices
    assert len(idxs) == mesh.n_vertices
    n = len(idxs)
    i = idxs[0]

    assert i.to_int() != mesh.null_vertex.to_int()
    assert i != mesh.null_vertex
    assert i < mesh.null_vertex
    assert i == i
    assert not (i != i)

    assert (idxs == idxs).all()
    assert not (idxs != idxs).any()

    assert (i == idxs).sum() == 1
    assert (i != idxs).sum() == (n - 1)
    assert (idxs == idxs[arange(n)]).all()
    assert (idxs == idxs[ones(n, dtype=bool)]).all()

    i0_repeated = idxs[np.zeros(n, dtype=int)]
    assert len(i0_repeated) == n
    assert (i == i0_repeated).all()

    set_ = set(idxs)
    assert len(set_) == n
    assert i in set_
