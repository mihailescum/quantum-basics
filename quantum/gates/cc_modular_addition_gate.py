import qiskit as qk
import numpy as np

from quantum.utils import get_bitmask
from quantum.gates import AdditionGate


class CCModularAdditionGate:
    """Implements the PhiADD(a)MOD(N) gate from 'Circuit for Shor's algorithm using 2n+3 qubits'
    by Stephane Beauregard. See https://arxiv.org/abs/quant-ph/0205095

    Given classical n-bit values `a`, `N` and a `(n+1)`-qubit state `|b>_n ⊗ |0>_1`,
    computes `|(a+b)>_{n+1}`. It is assumed that a,b < N.

    The circuit is controlled on the first two qubits.
    """

    def __init__(self, a, N, width, apply_QFT) -> None:
        self.a = a
        self.N = N
        self.width = width
        self.apply_QFT = apply_QFT
        self.num_qubits = None
        self._gate = None

    def get_native(self) -> qk.circuit.Gate:
        if not self._gate:
            control_reg = qk.circuit.QuantumRegister(2, "control")
            working_reg = qk.circuit.QuantumRegister(self.width + 1, "working")
            ancilla_reg = qk.circuit.AncillaRegister(1, "ancilla")
            qc = qk.QuantumCircuit(control_reg, working_reg, ancilla_reg)

            qft = qk.circuit.library.QFTGate(working_reg.size)
            if self.apply_QFT:
                # Put `b` into Fourier basis
                qc.append(qft, working_reg)

            add_a_gate = AdditionGate(self.a, self.width, apply_QFT=False)
            add_N_gate = AdditionGate(self.N, self.width, apply_QFT=False)

            qc.append(
                add_a_gate.get_native().control(control_reg.size),
                control_reg[:] + working_reg[:],
            )
            qc.append(
                add_N_gate.get_native().inverse(),
                working_reg,
            )

            qc.append(qft.inverse(), working_reg)
            qc.cx(working_reg[-1], ancilla_reg)
            qc.append(qft, working_reg)

            qc.append(
                add_N_gate.get_native().control(ancilla_reg.size),
                ancilla_reg[:] + working_reg[:],
            )
            qc.append(
                add_a_gate.get_native().inverse().control(control_reg.size),
                control_reg[:] + working_reg[:],
            )

            qc.append(qft.inverse(), working_reg)
            qc.x(working_reg[-1])
            qc.cx(working_reg[-1], ancilla_reg)
            qc.x(working_reg[-1])
            qc.append(qft, working_reg)

            qc.append(
                add_a_gate.get_native().control(control_reg.size),
                control_reg[:] + working_reg[:],
            )

            if self.apply_QFT:
                # Put `b` into compute basis
                qc.append(qft.inverse(), working_reg)

            self._gate = qc.to_gate()
            self._gate.name = f"PhiADD({self.a})MOD({self.N})"
            self.num_qubits = self._gate.num_qubits

        return self._gate
