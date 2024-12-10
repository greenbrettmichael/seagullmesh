from pathlib import Path

import pytest

from seagullmesh import Mesh3, sgm
_ = pytest.importorskip("seagullmesh.geodesic_distances")


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
