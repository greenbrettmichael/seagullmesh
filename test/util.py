import pytest
from numpy import array, cos, sin

from seagullmesh import sgm


def requires(module_name: str):
    return pytest.mark.skipif(
        not hasattr(sgm, module_name),
        reason=f'{module_name} module not installed')

