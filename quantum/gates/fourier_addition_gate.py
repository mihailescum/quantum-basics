import qiskit as qk
import numpy as np

from quantum.utils import get_bitmask


class FourierAdditionGate:
    """Implements the PhiADD(a) gate from 'Circuit for Shor's algorithm using 2n+3 qubits'
    by Stephane Beauregard. See https://arxiv.org/abs/quant-ph/0205095

    Given a classical n-bit value a and a n-qubit state |phi(b)>, computes |phi(a+b)>, where
    phi(b) is the quantum Fourier transform of b and phi(a+b) is the quantum Fourier transform
    of a + b mod 2^n"""

    def __init__(self, a, width, apply_QFT) -> None:
        self.a = a
        self.width = width
        self.apply_QFT = apply_QFT
        self.num_qubits = None
        self._gate = None

    def get_native(self) -> qk.circuit.Gate:
        if not self._gate:
            working_reg = qk.circuit.QuantumRegister(self.width + 1, "working")
            qc = qk.QuantumCircuit(working_reg)

            # Get bits of `a` for classical control
            a_bit_mask = get_bitmask(self.a, self.width, big_endian=True)

            if self.apply_QFT:
                # Put `b` into Fourier basis
                qc.append(
                    qk.circuit.library.QFTGate(working_reg.size),
                    working_reg,
                )

            for target_bit in range(self.width + 1):
                phase_shift = 0
                for control_bit in range(min(self.width, target_bit + 1)):
                    if a_bit_mask[control_bit]:
                        phase_shift += np.pi / np.pow(2, target_bit - control_bit)

                if phase_shift > 0:
                    # QFT reverses the order of bits. We accout for that here
                    qc.p(phase_shift, working_reg[self.width - target_bit])

            if self.apply_QFT:
                # Put `b` into compute basis
                qc.append(
                    qk.circuit.library.QFTGate(working_reg.size).inverse(),
                    working_reg,
                )

            self._gate = qc.to_gate()
            self._gate.name = f"PhiADD({self.a})"
            self.num_qubits = self._gate.num_qubits

        return self._gate
