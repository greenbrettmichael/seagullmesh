from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

try:
    import pyvista
except ImportError:
    pyvista = None

from seagullmesh import Mesh3


# def test_from_polygon_soup():
#     verts, faces = Mesh3.icosahedron()
#     mesh = Mesh3.from_polygon_soup(verts, faces)
#     assert mesh.n_vertices == 4 and mesh.n_faces == 4


@pytest.mark.parametrize('file', ['armadillo.off', 'sphere.ply'])
def test_from_file(file):
    file = Path(__file__).parent / 'assets' / file
    assert file.exists()
    _mesh = Mesh3.from_file(str(file))


@pytest.mark.parametrize('ext', ['ply', 'off'])
def test_to_file(ext):
    mesh = Mesh3.icosahedron()
    with TemporaryDirectory() as d:
        file = str(Path(d) / f'mesh.{ext}')
        mesh.to_file(file)


@pytest.mark.skipif(pyvista is None, reason="pyvista not installed")
def test_pyvista_roundtrip():
    pvmesh0 = pyvista.Sphere().clean().triangulate()
    mesh = Mesh3.from_pyvista(pvmesh0)
    _pvmesh1 = mesh.to_pyvista()

