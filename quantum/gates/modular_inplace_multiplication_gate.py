import qiskit as qk

import math

from quantum.gates import ModularMultiplicationGate


class ModularInplaceMultiplicationGate:
    """Implements the U_a gate from 'Circuit for Shor's algorithm using 2n+3 qubits'
    by Stephane Beauregard. See https://arxiv.org/abs/quant-ph/0205095

    Given classical n-bit values `a`, `N` and  |x>_n,
    computes |(ax) mod N>_n ⊗ |b+(ax) mod N>_n ⊗ |0>_1. It is assumed that a,b < N
    and that `gcd(a,N)=1`.
    """

    def __init__(self, a, N, width) -> None:
        assert a < N, "a must be strictly smaller than N"
        assert math.gcd(a, N) == 1, "gcd(a,N) must be equal to 1."

        self.a = a
        self.N = N
        self.width = width
        self.num_qubits = None
        self._gate = None

    def get_native(self) -> qk.circuit.Gate:
        if not self._gate:
            working_reg = qk.circuit.QuantumRegister(self.width, "working")
            ancilla_reg = qk.circuit.AncillaRegister(self.width + 2, "ancilla")
            qc = qk.QuantumCircuit(working_reg, ancilla_reg)

            mult_a = ModularMultiplicationGate(self.a, self.N, self.width)
            qc.append(mult_a.get_native(), range(qc.num_qubits))

            qc.swap(working_reg, ancilla_reg[:-2])

            a_inv = pow(self.a, -1, self.N)
            mult_a_inv = ModularMultiplicationGate(a_inv, self.N, self.width)
            qc.append(mult_a_inv.get_native().inverse(), range(qc.num_qubits))

            self._gate = qc.to_gate()
            self._gate.name = f"U_a({self.a})MOD({self.N})"
            self.num_qubits = self._gate.num_qubits

        return self._gate
