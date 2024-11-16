import pytest

from tempfile import TemporaryDirectory
from pathlib import Path

from seagullmesh import sgm, Mesh3
from test.util import tetrahedron

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
    pass


@pytest.fixture()
def armadillo():
    file = Path(__file__).parent / 'assets' / 'armadillo.off'
    assert file.exists()
    return Mesh3.from_file(str(file))


def test_estimate_geodesic_distance_source_vert(armadillo):
    armadillo.estimate_geodesic_distances(armadillo.vertices[0], 'distances')
    assert (armadillo.vertex_data['distances'] > 0).any()


def test_estimate_geodesic_distance_source_verts(armadillo):
    armadillo.estimate_geodesic_distances(armadillo.vertices[:3], 'distances')
    assert (armadillo.vertex_data['distances'] > 0).any()
