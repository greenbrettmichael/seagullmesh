import pytest
from pyvista import Sphere

from seagullmesh import Mesh3
from seagullmesh import _seagullmesh as sgm


skip_no_meshing = pytest.mark.skipif(
    not hasattr(sgm, 'meshing'), reason='meshing submodule not installed')


@pytest.fixture()
def sphere():
    return Mesh3.from_pyvista(Sphere().triangulate())


@skip_no_meshing
@pytest.mark.parametrize(('angle', 'area'), [(True, False), (False, True), (True, True)])
def test_angle_and_area_smoothing(sphere, area, angle):
    sphere.smooth_angle_and_area(
        sphere.faces,
        n_iter=1,
        use_area_smoothing=area,
        use_angle_smoothing=angle,
    )


@skip_no_meshing
def test_tangential_relaxation(sphere):
    sphere.tangential_relaxation(sphere.vertices, n_iter=1)


@skip_no_meshing
def test_smooth_shape(sphere):
    sphere.smooth_shape(sphere.faces, n_iter=1, time=.1)
