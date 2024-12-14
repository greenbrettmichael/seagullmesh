import pytest

_ = pytest.importorskip('seagullmesh._seagullmesh.tube_mesher')
_ = pytest.importorskip('seagullmesh._seagullmesh.geometry')
from seagullmesh.tube_mesher import TubeMesher


def test_cylinder():
    cyl = TubeMesher.cylinder()
    assert cyl.is_valid
    assert cyl.does_bound_a_volume()
