import pytest
import numpy as np

from seagullmesh import Mesh3


@pytest.fixture
def mesh():
    return Mesh3.icosahedron()
