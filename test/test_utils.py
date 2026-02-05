import pytest
import numpy.testing as npt

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


@pytest.mark.parametrize(
    "input, expected_output",
    [
        (
            np.array(
                [
                    [0, 1, 1, 1],
                    [0, 1, 0, 1],
                    [1, 1, 0, 1],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [1, 1, 0, 1],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=np.float64,
            ),
        ),
        (
            np.array(
                [
                    [0, 1, 1, 1],
                    [0, 1, 0, 1],
                    [1, 1, 0, 1],
                    [0, 1, 0, 1],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [1, 1, 0, 1],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.float64,
            ),
        ),
        (
            np.array(
                [
                    [0, 3, 1],
                    [3, 5, 1],
                    [5, 3, 1],
                    [6, 7, 4],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [3, 5, 1],
                    [0, 3, 1],
                    [0, 0, 1.1111111],
                    [0, 0, 0],
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_lower_triangular_form(input, expected_output):
    output = lower_triangular_form(input)
    npt.assert_allclose(output, expected_output)


@pytest.mark.parametrize(
    "x, width, big_endian, expected_output",
    [
        (3, 3, True, [1, 1, 0]),
        (3, 3, False, [0, 1, 1]),
        (7, 5, True, [1, 1, 1, 0, 0]),
        (7, 5, False, [0, 0, 1, 1, 1]),
    ],
)
def test_get_bitmask(x, width, big_endian, expected_output):
    output = get_bitmask(x, width, big_endian)
    npt.assert_equal(output, expected_output)
