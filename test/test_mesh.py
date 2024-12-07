import pytest

from tempfile import TemporaryDirectory
from pathlib import Path

from numpy import arange, ones

from seagullmesh import sgm, Mesh3
from test.util import tetrahedron, tetrahedron_mesh

props = sgm.properties


try:
    import pyvista
except ImportError:
    pyvista = None


def test_from_polygon_soup():
    verts, faces = tetrahedron()
    mesh = Mesh3.from_polygon_soup(verts, faces)
    assert mesh.n_vertices == 4 and mesh.n_faces == 4


@pytest.mark.parametrize('file', ['armadillo.off', 'sphere.ply'])
def test_from_file(file):
    file = Path(__file__).parent / 'assets' / file
    assert file.exists()
    _mesh = Mesh3.from_file(str(file))


@pytest.mark.parametrize('ext', ['ply', 'off'])
def test_to_file(ext):
    mesh = Mesh3.from_polygon_soup(*tetrahedron())
    with TemporaryDirectory() as d:
        file = str(Path(d) / f'mesh.{ext}')
        mesh.to_file(file)


@pytest.mark.skipif(pyvista is None, reason="pyvista not installed")
def test_pyvista_roundtrip():
    pvmesh0 = pyvista.Sphere().clean().triangulate()
    mesh = Mesh3.from_pyvista(pvmesh0)
    _pvmesh1 = mesh.to_pyvista()


def test_indices_indexing():
    mesh = Mesh3.icosahedron()
    idxs = mesh.vertices
    assert len(idxs) == mesh.n_vertices
    n = len(idxs)

    assert idxs[0].to_int() != mesh.null_vertex.to_int()
    assert idxs[0] == idxs[0]
    assert (idxs == idxs).all()
    assert not (idxs != idxs).any()

    assert (idxs == idxs[arange(n)]).all()
    assert (idxs == idxs[ones(n, dtype=bool)]).all()
    assert (idxs == idxs[ones(n, dtype=int)]).sum() == 1

    assert (idxs == sgm.mesh.Vertices(list(idxs))).all()



# @pytest.fixture()
# def armadillo():
#     file = Path(__file__).parent / 'assets' / 'armadillo.off'
#     assert file.exists()
#     return Mesh3.from_file(str(file))
#
#
# def test_estimate_geodesic_distance_source_vert(armadillo):
#     armadillo.estimate_geodesic_distances(armadillo.vertices[0], 'distances')
#     assert (armadillo.vertex_data['distances'] > 0).any()
#
#
# def test_estimate_geodesic_distance_source_verts(armadillo):
#     armadillo.estimate_geodesic_distances(armadillo.vertices[:3], 'distances')
#     assert (armadillo.vertex_data['distances'] > 0).any()
