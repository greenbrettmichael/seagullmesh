import numpy as np
import pytest
from numpy import full, arange, zeros, ones
from numpy.testing import assert_array_equal

from seagullmesh import Mesh3, Point2, MeshData, Point3, Vector2, Vector3, sgm, Indices, Vertex

props = pytest.importorskip("seagullmesh._seagullmesh.properties")


@pytest.fixture
def mesh():
    return Mesh3.icosahedron()


@pytest.mark.parametrize(
    ['data_name', 'cls', 'default'],
    [
        ('vertex_data', props.V_uint32_PropertyMap, 0),
        ('face_data', props.F_int64_PropertyMap, 0),
    ]
)
def test_explicit_property_map_construction(mesh, data_name, cls, default):
    d = getattr(mesh, data_name)
    d['foo'] = cls(mesh.mesh, 'foo', default)
    assert (d['foo'][:] == default).all()


@pytest.mark.parametrize(
    ['data_name', 'cls', 'default', 'dtype'],
    [
        ('vertex_data', props.V_uint32_PropertyMap, 0, 'uint32'),
        ('face_data', props.F_int64_PropertyMap, 0, 'int64'),
        ('edge_data', props.E_bool_PropertyMap, False, 'bool'),
        ('halfedge_data', props.H_double_PropertyMap, -1.0, 'double'),
    ]
)
def test_add_property_map_typed(mesh, data_name, cls, default, dtype):
    data = getattr(mesh, data_name)
    pmap = data.add_property('foo', default=default, dtype=dtype)
    assert isinstance(pmap.pmap, cls)
    dtype_name = pmap[data.all_indices].dtype.name
    if dtype_name == 'float64':
        dtype_name = 'double'
    assert dtype_name == dtype


@pytest.mark.parametrize('key_type', ['vertex', 'face', 'edge', 'halfedge'])
@pytest.mark.parametrize('val_type', [int, bool, float])
def test_scalar_properties(mesh, key_type, val_type):
    data: MeshData = getattr(mesh, key_type + '_data')

    data['foo'] = full(data.n_mesh_keys, val_type(0))

    keys = data.all_indices
    key = keys[0]
    data['foo'][key] = val_type(1)

    assert 'foo' in data
    assert 'bar' not in data

    val = data['foo'][key]
    assert val == val_type(1)

    val = data['foo'][0]
    assert val == val_type(1)

    data['foo'][keys[:2]] = [val_type(1), val_type(1)]
    assert data['foo'][keys[0]] == val_type(1) and data['foo'][keys[1]] == val_type(1)

    data.remove_property('foo')
    assert 'foo' not in data.keys()
    assert 'foo' not in data


@pytest.mark.parametrize('key_type', ['vertex', 'face', 'edge', 'halfedge'])
@pytest.mark.parametrize('val_type', [Point2, Point3, Vector2, Vector3])
def test_array_properties(mesh, key_type, val_type):
    d: MeshData = getattr(mesh, key_type + '_data')

    ndims = int(val_type.__name__[-1])
    default = val_type(*[0.0 for _ in range(ndims)])
    d.add_property('foo', default=default)

    nkeys = d.n_mesh_keys
    data = np.random.uniform(-1, 1, (nkeys, ndims))
    d['foo'] = data

    assert_array_equal(data, d['foo'][:])

    data2 = data * 2
    objs = [val_type(*val) for val in data2]
    d['foo'].set_objects(d.all_indices, objs)
    objs2 = d['foo'].get_objects(d.all_indices)

    for (o1, o2) in zip(objs, objs2):
        assert o1 == o2


def test_copy_mesh_copies_properties(mesh):
    foo = mesh.vertex_data.add_property('foo', default=0)
    foo[0] = 1

    mesh1 = mesh.copy()
    foo1 = mesh1.vertex_data['foo']
    assert foo[0] == 1
    assert foo1[0] == 1  # Should have been copied

    foo1[0] = 2
    assert foo[0] == 1  # Should be unchanged
    assert foo1[0] == 2


@pytest.mark.parametrize('inplace', (False, True))
def test_add_mesh_adds_properties(inplace: bool):
    orig = Mesh3.icosahedron()
    nv0 = orig.n_vertices

    orig.vertex_data['foo'] = 1
    assert set(orig.vertex_data['foo'][:]) == {1}

    other = Mesh3.pyramid()
    other.vertex_data['foo'] = 2
    assert set(other.vertex_data['foo'][:]) == {2}
    assert type(orig.vertex_data['foo'].pmap) is type(other.vertex_data['foo'].pmap)

    added = orig.add(other, inplace=inplace)
    assert added.n_vertices == (nv0 + other.n_vertices)

    if inplace:
        assert added is orig
    else:
        assert added is not orig

    assert set(added.vertex_data['foo']) == {1, 2}

