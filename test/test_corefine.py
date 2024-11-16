import pytest
from numpy import pi

from seagullmesh import Mesh3, _seagullmesh as sgm
from test.util import tetrahedron


def corefine_meshes():
    m1 = Mesh3.from_polygon_soup(*tetrahedron())
    m2 = Mesh3.from_polygon_soup(*tetrahedron(scale=0.9, rot_z=pi/3))
    return m1, m2


skip_no_corefine = pytest.mark.skipif(
    not hasattr(sgm, 'corefine'), reason='corefine submodule not installed')


@skip_no_corefine
def test_corefine():
    m1, m2 = corefine_meshes()
    nv1_orig, nv2_orig = m1.n_vertices, m2.n_vertices
    m1.corefine(m2)

    nv1, nv2 = m1.n_vertices, m2.n_vertices
    assert nv1 > nv1_orig and nv2 > nv2_orig


@skip_no_corefine
@pytest.mark.parametrize('op', ['union', 'intersection', 'difference'])
@pytest.mark.parametrize('inplace', [False, True])
def test_boolean_ops(op, inplace):
    m1, m2 = corefine_meshes()
    _m3 = getattr(m1, op)(m2, inplace=inplace)
