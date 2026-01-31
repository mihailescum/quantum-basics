import pytest

from qiskit import transpile
from qiskit_aer import AerSimulator

from quantum.algorithms import DeutschJozsa
from quantum.gates import AutoOracleGate

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
    "f, n, expected_result",
    [
        (f_1, 1, DeutschJozsa.Result.Balanced),
        (g_1, 1, DeutschJozsa.Result.Constant),
        (f_2, 2, DeutschJozsa.Result.Balanced),
        (g_2, 2, DeutschJozsa.Result.Constant),
        (f_3, 3, DeutschJozsa.Result.Balanced),
        (g_3, 3, DeutschJozsa.Result.Constant),
    ],
)
def test_deutschjozsa(f, n, expected_result):
    def simulate_qc(qc):
        simulator = AerSimulator()
        qct = transpile(qc, backend=simulator)
        return simulator.run(qct, shots=NUM_SHOTS, memory=False).result().get_counts()

    oracle = AutoOracleGate(f, n, 1)
    algorithm = DeutschJozsa(oracle.gate, n)

    result = algorithm.run(simulate_qc)

    assert result is expected_result
