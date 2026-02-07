import qiskit as qk
import numpy as np

from quantum.gates import CCModularAdditionGate


class CModularMultiplicationGate:
    """Implements the CMULT(a)MOD(N) gate from 'Circuit for Shor's algorithm using 2n+3 qubits'
    by Stephane Beauregard. See https://arxiv.org/abs/quant-ph/0205095

    Given classical n-bit values `a`, `N` and  |x>_n, |b>_n ⊗ |0>_1,
    computes |x>_n ⊗ |b+(ax) mod N>_n ⊗ |0>_1. It is assumed that a,b < N.

    The circuit is controlled on the first qubit.
    """

    def __init__(self, a, N, width) -> None:
        self.a = a
        self.N = N
        self.width = width
        self.num_qubits = None
        self._gate = None

    def get_native(self) -> qk.circuit.Gate:
        if not self._gate:
            control_reg = qk.circuit.QuantumRegister(1, "control")
            x_reg = qk.circuit.QuantumRegister(self.width, "x")
            b_reg = qk.circuit.QuantumRegister(self.width + 1, "b")
            ancilla_reg = qk.circuit.AncillaRegister(1, "ancilla")
            qc = qk.QuantumCircuit(control_reg, x_reg, b_reg, ancilla_reg)

            # Put `b` into Fourier basis
            qft = qk.circuit.library.QFTGate(b_reg.size)
            qc.append(qft, b_reg)

            factor = self.a
            for bit in range(self.width):
                mod_add = CCModularAdditionGate(
                    factor % self.N,
                    self.N,
                    self.width,
                    apply_QFT=False,
                )
                qc.append(
                    mod_add.get_native(),
                    [control_reg[0], x_reg[bit]] + b_reg[:] + ancilla_reg[:],
                )

                factor *= 2

            # Put `b` into compute basis
            qc.append(qft.inverse(), b_reg)

            self._gate = qc.to_gate()
            self._gate.name = f"CMULT({self.a})MOD({self.N})"
            self.num_qubits = self._gate.num_qubits

        return self._gate
