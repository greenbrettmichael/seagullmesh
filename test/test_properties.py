import numpy as np
import pytest
from numpy import full, arange, zeros, ones
from numpy.testing import assert_array_equal

from seagullmesh import Mesh3, Point2, MeshData, Point3, Vector2, Vector3, sgm, Indices, Vertex
from test.test_mesh import props
from test.util import tetrahedron_mesh


_halfedge = pytest.param(
    'halfedge',
    marks=pytest.mark.skip(reason="halfedges are flaky"),
)


def test_indices_from_vector():
    mesh = Mesh3.icosahedron()
    vs = mesh.vertices
    v0, v1 = vs[0], vs[1]
    _idxs = sgm.mesh.Vertices.from_vector([v0, v1])
    idxs = Indices(_idxs, Vertex)
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

    i0_repeated = idxs[zeros(n, dtype=int)]
    assert len(i0_repeated) == n
    assert (i == i0_repeated).all()

    set_ = set(idxs)
    assert len(set_) == n
    assert i in set_


@pytest.mark.parametrize(
    ['data_name', 'cls', 'default'],
    [
        ('vertex_data', props.V_uint32_PropertyMap, 0),
        ('face_data', props.F_int64_PropertyMap, 0),
    ]
)
def test_explicit_property_map_construction(data_name, cls, default):
    mesh = Mesh3.icosahedron()
    d = getattr(mesh, data_name)
    d['foo'] = cls(mesh.mesh, 'foo', default)
    assert (d['foo'][:] == default).all()


@pytest.mark.parametrize(
    ['data_name', 'cls', 'default', 'dtype'],
    [
        ('vertex_data', props.V_uint32_PropertyMap, 0, 'uint32'),
        ('face_data', props.F_int64_PropertyMap, 0, 'int64'),
    ]
)
def test_add_property_map_typed(data_name, cls, default, dtype):
    mesh = Mesh3.icosahedron()
    data = getattr(mesh, data_name)
    pmap = data.add_property('foo', default=default, dtype=dtype)
    assert isinstance(pmap.pmap, cls)
    assert pmap[data.mesh_keys].dtype.name == dtype


# @pytest.mark.parametrize('key_type', ['vertex', 'face', 'edge', _halfedge])
# @pytest.mark.parametrize('val_type', [int, bool, float])
# def test_scalar_properties(key_type, val_type):
#     mesh = tetrahedron_mesh()
#     data: MeshData = getattr(mesh, key_type + '_data')
#
#     data['foo'] = full(data.n_mesh_keys, val_type(0))
#
#     keys = data.mesh_keys
#     key = keys[0]
#     data['foo'][key] = val_type(1)
#
#     assert 'foo' in data
#     assert 'bar' not in data
#
#     val = data['foo'][key]
#     assert val == val_type(1)
#
#     data['foo'][keys[:2]] = [val_type(1), val_type(1)]
#     assert data['foo'][keys[0]] == val_type(1) and data['foo'][keys[1]] == val_type(1)
#
#     data.remove_property('foo')
#     assert 'foo' not in data.keys()
#     assert 'foo' not in data


# @pytest.mark.parametrize('key_type', ['vertex', 'face', 'edge', _halfedge])
# @pytest.mark.parametrize('val_type', [Point2, Point3, Vector2, Vector3])
# def test_array_properties(key_type, val_type):
#     mesh = tetrahedron_mesh()
#     d: MeshData = getattr(mesh, key_type + '_data')
#
#     ndims = int(val_type.__name__[-1])
#     default = val_type(*[0.0 for _ in range(ndims)])
#     d.add_property('foo', default=default)
#
#     nkeys = d.n_mesh_keys
#     data = np.random.uniform(-1, 1, (nkeys, ndims))
#     d['foo'] = data
#
#     assert_array_equal(data, d['foo'][:])
#
#     data2 = data * 2
#     objs = [val_type(*val) for val in data2]
#     d['foo'].set_objects(d.mesh_keys, objs)
#     objs2 = d['foo'].get_objects(d.mesh_keys)
#
#     for (o1, o2) in zip(objs, objs2):
#         assert o1 == o2


# def test_copy_mesh_copies_properties():
#     mesh1 = Mesh3.icosahedron()
#     foo1 = mesh1.vertex_data.add_property('foo', default=0)
#     foo1[0] = 1
#
#     mesh2 = mesh1.copy()
#     foo2 = mesh2.vertex_data['foo']
#     assert foo1[0] == 1
#     assert foo2[0] == 1  # Should have been copied
#
#     foo2[0] = 2
#     assert foo1[0] == 1  # Should be unchanged
#     assert foo2[0] == 2

