import pytest

import qiskit as qk
import qiskit_aer as qk_aer

import numpy as np

from helper import test_matrix_using_basis, get_matrix_representation

from quantum.gates import (
    AdditionGate,
    ModularAdditionGate,
    ModularMultiplicationGate,
)
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
def test_addition_gate(a, b, width):
    b_reg = qk.circuit.QuantumRegister(width + 1, "b")
    measurement_reg = qk.circuit.ClassicalRegister(b_reg.size)
    qc = qk.QuantumCircuit(b_reg, measurement_reg)

    # Load `b` into the quantum register
    b_bit_mask = get_bitmask(b, width + 1, big_endian=True)
    b_flip_bits = np.where(b_bit_mask == 1)[0].tolist()
    if len(b_flip_bits) > 0:
        qc.x(b_reg[b_flip_bits])

    gate = AdditionGate(a, width, apply_QFT=True)
    native = gate.get_native()
    qc.append(native, range(qc.num_qubits))

    # Measure `b`
    qc.measure(b_reg, measurement_reg)

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
def test_modular_addition_gate(a, b, N, width):
    b_reg = qk.circuit.QuantumRegister(width + 1, "b")
    ancilla_reg = qk.circuit.AncillaRegister(1, "ancilla")
    measurement_reg = qk.circuit.ClassicalRegister(b_reg.size)
    qc = qk.QuantumCircuit(b_reg, ancilla_reg, measurement_reg)

    # Load `b` into the quantum register
    b_bit_mask = get_bitmask(b, width + 1, big_endian=True)
    b_flip_bits = np.where(b_bit_mask == 1)[0].tolist()
    if len(b_flip_bits) > 0:
        qc.x(b_reg[b_flip_bits])

    gate = ModularAdditionGate(a, N, width, apply_QFT=True)
    native = gate.get_native()
    qc.append(native, range(qc.num_qubits))

    # Measure `b`
    qc.measure(b_reg, measurement_reg)

    simulator = qk_aer.AerSimulator()
    qct = qk.transpile(qc, backend=simulator)
    outcomes = (
        simulator.run(qct, shots=1, memory=False).result().get_counts().int_outcomes()
    )

    expected_result = (a + b) % N
    assert outcomes.get(
        expected_result
    ), f"Outcomes were {outcomes}, expected {expected_result}"


@pytest.mark.parametrize(
    "x, a, b, N, width",
    [
        (3, 1, 0, 4, 3),
        (3, 6, 0, 7, 3),
        (3, 6, 2, 21, 5),
    ],
)
def test_modular_multiplication_gate(x, a, b, N, width):
    x_reg = qk.circuit.QuantumRegister(width, "x")
    b_reg = qk.circuit.QuantumRegister(width + 1, "b")
    ancilla_reg = qk.circuit.AncillaRegister(1, "ancilla")
    measurement_reg = qk.circuit.ClassicalRegister(b_reg.size)
    qc = qk.QuantumCircuit(x_reg, b_reg, ancilla_reg, measurement_reg)

    # Load `x` into the quantum register
    x_bit_mask = get_bitmask(x, width, big_endian=True)
    x_flip_bits = np.where(x_bit_mask == 1)[0].tolist()
    if len(x_flip_bits) > 0:
        qc.x(x_reg[x_flip_bits])

    # Load `b` into the quantum register
    b_bit_mask = get_bitmask(b, width + 1, big_endian=True)
    b_flip_bits = np.where(b_bit_mask == 1)[0].tolist()
    if len(b_flip_bits) > 0:
        qc.x(b_reg[b_flip_bits])

    gate = ModularMultiplicationGate(a, N, width)
    native = gate.get_native()
    qc.append(native, range(qc.num_qubits))

    # Measure `b`
    qc.measure(b_reg, measurement_reg)

    simulator = qk_aer.AerSimulator()
    qct = qk.transpile(qc, backend=simulator)
    outcomes = (
        simulator.run(qct, shots=1, memory=False).result().get_counts().int_outcomes()
    )

    expected_result = (b + a * x) % N
    assert outcomes.get(
        expected_result
    ), f"Outcomes were {outcomes}, expected {expected_result}"
