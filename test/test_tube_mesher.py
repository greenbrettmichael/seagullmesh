import pytest

_ = pytest.importorskip('seagullmesh._seagullmesh.tube_mesher')
# _ = pytest.importorskip('seagullmesh._seagullmesh.geometry')
from seagullmesh.tube_mesher import TubeMesher


@pytest.mark.parametrize('closed', (False,))
def test_cylinder(closed: bool):
    cyl = TubeMesher.cylinder(closed=closed, flip_faces=False, n_axial=2, n_radial=3)
    assert cyl.is_valid
    # assert cyl.is_closed is closed
    # assert cyl.does_bound_a_volume()
