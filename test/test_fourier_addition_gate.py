import pytest

import qiskit as qk
import qiskit_aer as qk_aer

import numpy as np

from helper import test_matrix_using_basis, get_matrix_representation

from quantum.gates import FourierAdditionGate, ModularFourierAdditionGate
from quantum.utils import get_bitmask


@pytest.mark.parametrize(
    "a, b, width",
    [
        (0, 0, 3),
        (0, 1, 3),
        (1, 0, 3),
        (1, 1, 1),
        (3, 3, 2),
        (4, 7, 4),
    ],
)
def test_fourier_addition_gate(a, b, width):
    qc = qk.QuantumCircuit(width + 1, width + 1)

    # Load `b` into the quantum register
    b_bit_mask = get_bitmask(b, width + 1, big_endian=True)
    b_flip_bits = np.where(b_bit_mask == 1)[0].tolist()
    if len(b_flip_bits) > 0:
        qc.x(b_flip_bits)

    gate = FourierAdditionGate(a, width, apply_QFT=True)
    native = gate.get_native()
    qc.append(native, range(qc.num_qubits))

    # Measure `b`
    qc.measure(range(qc.num_qubits), range(qc.num_clbits))

    simulator = qk_aer.AerSimulator()
    qct = qk.transpile(qc, backend=simulator)
    outcomes = (
        simulator.run(qct, shots=1, memory=False).result().get_counts().int_outcomes()
    )

    expected_result = a + b
    assert outcomes.get(
        expected_result
    ), f"Outcomes were {outcomes}, expected {expected_result}"


@pytest.mark.parametrize(
    "a, b, N, width",
    [
        (1, 1, 2, 2),
        (3, 3, 4, 3),
        (4, 7, 11, 4),
    ],
)
def test__modular_fourier_addition_gate(a, b, N, width):
    qc = qk.QuantumCircuit(width + 4, width + 1)

    # Activate the two control bits
    qc.x(range(2))

    # Load `b` into the quantum register
    b_bit_mask = get_bitmask(b, width + 1, big_endian=True)
    b_flip_bits = (2 + np.where(b_bit_mask == 1)[0]).tolist()
    if len(b_flip_bits) > 0:
        qc.x(b_flip_bits)

    gate = ModularFourierAdditionGate(a, N, width, apply_QFT=True)
    native = gate.get_native()
    qc.append(native, range(qc.num_qubits))

    # Measure `b`
    qc.measure(range(2, width + 3), range(qc.num_clbits))

    simulator = qk_aer.AerSimulator()
    qct = qk.transpile(qc, backend=simulator)
    outcomes = (
        simulator.run(qct, shots=1, memory=False).result().get_counts().int_outcomes()
    )

    expected_result = (a + b) % N
    assert outcomes.get(
        expected_result
    ), f"Outcomes were {outcomes}, expected {expected_result}"
