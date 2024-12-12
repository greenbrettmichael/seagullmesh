import pytest
import numpy as np

from seagullmesh import Mesh3


def transform(axis, angle, translate):
    x, y, z = np.array(axis) / np.linalg.norm(axis)
    c, s, t = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    out = np.eye(4)
    out[:3, :3] = [
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c, ],]
    out[:3, 3] = translate
    return out


@pytest.fixture
def mesh():
    return Mesh3.icosahedron()
