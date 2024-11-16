import numpy as np
import pytest
from numpy import full
from numpy.testing import assert_array_equal

from seagullmesh import Point2, Mesh3, MeshData, Point3, Vector2, Vector3
from test.test_mesh import props
from test.util import tetrahedron_mesh, tetrahedron


@pytest.mark.parametrize(
    ['data_name', 'cls', 'default'],
    [
        ('vertex_data', props.VertIntPropertyMap, 0),
        ('vertex_data', props.VertUIntPropertyMap, 0),
        ('face_data', props.FaceIntPropertyMap, 0),
        ('face_data', props.FaceUIntPropertyMap, 0),

    ]
)
def test_explicit_property_map_construction(data_name, cls, default):
    mesh = tetrahedron_mesh()
    d = getattr(mesh, data_name)
    d['foo'] = cls(mesh.mesh, 'foo', default)
    assert (d['foo'][:] == default).all()


@pytest.mark.parametrize(
    ['data_name', 'cls', 'default', 'signed'],
    [
        ('vertex_data', props.VertBoolPropertyMap, False, None),
        ('vertex_data', props.VertIntPropertyMap, 0, True),
        ('vertex_data', props.VertUIntPropertyMap, 0, False),
        ('vertex_data', props.VertDoublePropertyMap, 0.0, None),
        ('vertex_data', props.VertPoint2PropertyMap, Point2(0, 0), None),
        ('face_data', props.FaceBoolPropertyMap, False, None),
        ('face_data', props.FaceIntPropertyMap, 0, True),
        ('face_data', props.FaceUIntPropertyMap, 0, False),
        ('face_data', props.FaceDoublePropertyMap, 0.0, None),
        ('face_data', props.FacePoint2PropertyMap, Point2(0, 0), None),
    ]
)
def test_add_property_map(data_name, cls, default, signed):
    mesh = Mesh3.from_polygon_soup(*tetrahedron())
    data = getattr(mesh, data_name)
    pmap = data.add_property('foo', default=default, signed=signed)
    assert isinstance(pmap.pmap, cls)


@pytest.mark.parametrize('key_type', ['vertex', 'face', 'edge', 'halfedge'])
@pytest.mark.parametrize('val_type', [int, bool, float])
def test_scalar_properties(key_type, val_type):
    mesh = Mesh3.from_polygon_soup(*tetrahedron())
    data: MeshData = getattr(mesh, key_type + '_data')

    data['foo'] = full(data.n_mesh_keys, val_type(0))

    keys = data.mesh_keys
    key = keys[0]
    data['foo'][key] = val_type(1)

    assert 'foo' in data
    assert 'bar' not in data

    val = data['foo'][key]
    assert val == val_type(1)

    data['foo'][keys[:2]] = [val_type(1), val_type(1)]
    assert data['foo'][keys[0]] == val_type(1) and data['foo'][keys[1]] == val_type(1)

    data.remove_property('foo')
    assert 'foo' not in data.keys()
    assert 'foo' not in data


@pytest.mark.parametrize('key_type', ['vertex', 'face', 'edge', 'halfedge'])
@pytest.mark.parametrize('val_type', [Point2, Point3, Vector2, Vector3])
def test_array_properties(key_type, val_type):
    mesh = Mesh3.from_polygon_soup(*tetrahedron())
    d: MeshData = getattr(mesh, key_type + '_data')

    ndims = int(val_type.__name__[-1])
    default = val_type(*[0.0 for _ in range(ndims)])
    d.add_property('foo', default=default)

    nkeys = d.n_mesh_keys
    data = np.random.uniform(-1, 1, (nkeys, ndims))
    d['foo'] = data

    assert_array_equal(data, d['foo'][:])
