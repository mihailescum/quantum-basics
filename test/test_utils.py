import pytest

from quantum.utils import *


@pytest.mark.parametrize(
    "x, y, dim_y, expected_result",
    [
        (0, 0, 5, 0),
        (1, 1, 1, 3),
        (1, 1, 3, 9),
        (1, 2, 4, 33),
        (0, 1, 6, 64),
    ],
)
def test_combine_basis_state(x, y, dim_y, expected_result):
    result = combine_basis_state(x, y, dim_y)
    assert result == expected_result


@pytest.mark.parametrize(
    "a, dim_second, expected_result",
    [
        (0, 5, (0, 0)),
        (3, 1, (1, 1)),
        (9, 3, (1, 1)),
        (33, 4, (1, 2)),
        (64, 6, (0, 1)),
    ],
)
def test_reduce_basis_state(a, dim_second, expected_result):
    result = reduce_basis_state(a, dim_second)
    assert result == expected_result
