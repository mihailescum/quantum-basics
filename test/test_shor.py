import pytest
import math

from qiskit import transpile
from qiskit_aer import AerSimulator

from quantum.algorithms import Shor

NUM_SHOTS = 5


@pytest.mark.parametrize("N", [15, 12])
def test_deutschjozsa(N):
    def simulate_qc(qc):
        simulator = AerSimulator()
        qct = transpile(qc, backend=simulator)
        return (
            simulator.run(qct, shots=4, memory=False)
            .result()
            .get_counts()
            .int_outcomes()
        )

    algorithm = Shor(N)
    result = algorithm.run(simulate_qc)

    assert math.prod(result) == N
    for factor in result:
        assert 1 < factor < N
