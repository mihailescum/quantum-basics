import pytest

from qiskit_aer import AerSimulator

from quantum.algorithms import DeutschJozsa
from quantum.oracles import AutoOracle

NUM_SHOTS = 5


def f_1(x):
    if x == 0:
        return 0
    else:
        return 1


def g_1(x):
    if x == 0:
        return 1
    else:
        return 1


def f_2(x):
    if x == 0 or x == 2:
        return 0
    else:
        return 1


def g_2(x):
    if x == 0 or x == 2:
        return 1
    else:
        return 1


def f_3(x):
    if x == 0 or x == 3 or x == 6 or x == 7:
        return 0
    else:
        return 1


def g_3(x):
    if x == 0 or x == 3 or x == 6 or x == 7:
        return 1
    else:
        return 1


@pytest.mark.parametrize(
    "f, n, expected_output",
    [
        (f_1, 1, "balanced"),
        (g_1, 1, "constant"),
        (f_2, 2, "balanced"),
        (g_2, 2, "constant"),
        (f_3, 3, "balanced"),
        (g_3, 3, "constant"),
    ],
)
def test_deutschjozsa(f, n, expected_output):
    oracle = AutoOracle(f, n, 1)
    algorithm = DeutschJozsa(oracle.gate, n)
    qc = algorithm.qc

    result = AerSimulator().run(qc, shots=NUM_SHOTS, memory=False).result()
    counts = result.get_counts()
    label_zero = "0" * n
    if label_zero not in counts:
        counts[label_zero] = 0

    output = "error"
    if counts[label_zero] == NUM_SHOTS:
        output = "constant"
    elif counts[label_zero] == 0:
        output = "balanced"

    assert output == expected_output
