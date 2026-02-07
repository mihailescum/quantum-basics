import pytest

import qiskit as qk
import qiskit_aer as qk_aer

import numpy as np

from helper import test_matrix_using_basis, get_matrix_representation

from quantum.gates import (
    AdditionGate,
    CCModularAdditionGate,
    CModularMultiplicationGate,
    CModularInplaceMultiplicationGate,
)
from quantum.utils import get_bitmask, combine_basis_state, split_state


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
    ), f"Outcomes were {outcomes.keys()}, expected {expected_result}"


@pytest.mark.parametrize(
    "c, a, b, N, width",
    [
        (3, 1, 1, 2, 2),
        (3, 3, 3, 4, 3),
        (3, 4, 7, 11, 4),
        (0, 1, 1, 2, 2),
        (1, 3, 3, 4, 3),
        (2, 4, 7, 11, 4),
    ],
)
def test_modular_addition_gate(c, a, b, N, width):
    controls = qk.circuit.QuantumRegister(2, "controls")
    b_reg = qk.circuit.QuantumRegister(width + 1, "b")
    ancillas = qk.circuit.AncillaRegister(1, "ancilla")

    qc = qk.QuantumCircuit(
        controls,
        b_reg,
        ancillas,
    )

    # Activate controls
    c_bit_mask = get_bitmask(c, width, big_endian=True)
    c_flip_bits = np.where(c_bit_mask == 1)[0].tolist()
    if len(c_flip_bits) > 0:
        qc.x(controls[c_flip_bits])

    # Load `b` into the quantum register
    b_bit_mask = get_bitmask(b, width + 1, big_endian=True)
    b_flip_bits = np.where(b_bit_mask == 1)[0].tolist()
    if len(b_flip_bits) > 0:
        qc.x(b_reg[b_flip_bits])

    gate = CCModularAdditionGate(a, N, width, apply_QFT=True)
    native = gate.get_native()
    qc.append(native, range(qc.num_qubits))

    qc.measure_all()

    simulator = qk_aer.AerSimulator()
    qct = qk.transpile(qc, backend=simulator)
    outcomes = simulator.run(qct, shots=1, memory=False).result().get_counts()
    result = split_state(
        list(outcomes.keys())[0],
        controls.size,
        b_reg.size,
        ancillas.size,
    )

    if c == int("1" * controls.size, 2):
        expected_result = (c, (a + b) % N, 0)
    else:
        expected_result = (c, b, 0)
    assert result == expected_result, f"Result was {result}, expected {expected_result}"


@pytest.mark.parametrize(
    "c, x, a, b, N, width",
    [
        (1, 3, 1, 0, 4, 3),
        (1, 3, 6, 0, 7, 3),
        (1, 3, 6, 2, 21, 5),
        (0, 3, 1, 0, 4, 3),
        (0, 3, 6, 0, 7, 3),
        (0, 3, 6, 2, 21, 5),
    ],
)
def test_modular_multiplication_gate(c, x, a, b, N, width):
    controls = qk.circuit.QuantumRegister(1, "controls")
    x_reg = qk.circuit.QuantumRegister(width, "x")
    b_reg = qk.circuit.QuantumRegister(width + 1, "b")
    ancillas = qk.circuit.AncillaRegister(1, "ancilla")
    qc = qk.QuantumCircuit(controls, x_reg, b_reg, ancillas)

    # Activate controls
    c_bit_mask = get_bitmask(c, width, big_endian=True)
    c_flip_bits = np.where(c_bit_mask == 1)[0].tolist()
    if len(c_flip_bits) > 0:
        qc.x(controls[c_flip_bits])

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

    gate = CModularMultiplicationGate(a, N, width)
    native = gate.get_native()
    qc.append(native, range(qc.num_qubits))

    # Measure
    qc.measure_all()

    simulator = qk_aer.AerSimulator()
    qct = qk.transpile(qc, backend=simulator)

    outcomes = simulator.run(qct, shots=1, memory=False).result().get_counts()
    result = split_state(
        list(outcomes.keys())[0],
        controls.size,
        x_reg.size,
        b_reg.size,
        ancillas.size,
    )

    if c == int("1" * controls.size, 2):
        expected_result = (c, x, (b + a * x) % N, 0)
    else:
        expected_result = (c, x, b, 0)

    assert result == expected_result, f"Result was {result}, expected {expected_result}"


@pytest.mark.parametrize(
    "c, x, a, N, width",
    [
        (1, 3, 1, 4, 3),
        (1, 3, 6, 7, 3),
        (1, 3, 7, 20, 5),
        (0, 3, 1, 4, 3),
        (0, 3, 6, 7, 3),
        (0, 3, 7, 20, 5),
    ],
)
def test_modular_inplace_multiplication_gate(c, x, a, N, width):
    controls = qk.circuit.QuantumRegister(1, "controls")
    x_reg = qk.circuit.QuantumRegister(width, "x")
    ancillas = qk.circuit.AncillaRegister(width + 2, "ancilla")
    qc = qk.QuantumCircuit(controls, x_reg, ancillas)

    # Activate controls
    c_bit_mask = get_bitmask(c, width, big_endian=True)
    c_flip_bits = np.where(c_bit_mask == 1)[0].tolist()
    if len(c_flip_bits) > 0:
        qc.x(controls[c_flip_bits])

    # Load `x` into the quantum register
    x_bit_mask = get_bitmask(x, width, big_endian=True)
    x_flip_bits = np.where(x_bit_mask == 1)[0].tolist()
    if len(x_flip_bits) > 0:
        qc.x(x_reg[x_flip_bits])

    gate = CModularInplaceMultiplicationGate(a, N, width)
    native = gate.get_native()
    qc.append(native, range(qc.num_qubits))

    # Measure
    qc.measure_all()

    simulator = qk_aer.AerSimulator()
    qct = qk.transpile(qc, backend=simulator)

    expected_result = (a * x) % N

    outcomes = simulator.run(qct, shots=1, memory=False).result().get_counts()
    result = split_state(
        list(outcomes.keys())[0],
        controls.size,
        x_reg.size,
        ancillas.size,
    )

    if c == int("1" * controls.size, 2):
        expected_result = (c, (a * x) % N, 0)
    else:
        expected_result = (c, x, 0)

    assert result == expected_result, f"Result was {result}, expected {expected_result}"
