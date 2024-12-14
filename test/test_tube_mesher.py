import pytest

_ = pytest.importorskip('seagullmesh._seagullmesh.tube_mesher')
_ = pytest.importorskip('seagullmesh._seagullmesh.geometry')
from seagullmesh.tube_mesher import TubeMesher


@pytest.mark.parametrize('closed', (False, True))
def test_cylinder(closed: bool):
    cyl = TubeMesher.cylinder(closed=closed, flip_faces=True)
    assert cyl.is_valid
    assert cyl.is_closed is closed
    # assert cyl.does_bound_a_volume()
