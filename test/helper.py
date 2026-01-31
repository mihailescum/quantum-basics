from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import numpy as np

from collections.abc import Callable


def nottest(obj):
    obj.__test__ = False
    return obj


@nottest
def test_gate_using_basis(
    unitary: np.ndarray, validation: Callable[[int], int], prob_tol=1e-7
):
    dimension = unitary.shape[0]

    for input in range(dimension):
        input_statevector = Statevector.from_int(input, dimension)
        result_probabilities = Statevector(
            unitary @ input_statevector.data
        ).probabilities()
        result_label = np.argmax(result_probabilities)

        expected_result = validation(input)
        assert (
            result_label == expected_result
        ), f"Result: {result_label}, Expected: {expected_result}"
        assert (
            np.abs(1 - result_probabilities[result_label]) < prob_tol
        ), f"Probability of result was {result_probabilities[result_label]}"


def fast_unitary_of_circuit(qc: QuantumCircuit):
    qc.save_unitary()

    simulator = AerSimulator()
    qct = transpile(qc, backend=simulator)
    results = simulator.run(qct, shots=0).result()
    unitary = results.get_unitary().data
    return unitary
