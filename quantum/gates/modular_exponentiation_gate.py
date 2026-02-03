import qiskit as qk

from quantum.gates import BasisPermutationGate
from quantum.utils import reduce_basis_state, combine_basis_state


class ModularExponentiationGate:
    def __init__(self, x: int, M: int, dim_q0: int, dim_q1: int) -> None:
        def action(state: int) -> int:
            j, k = reduce_basis_state(state, dim_q0)
            result = k ^ pow(x, j, M)
            return combine_basis_state(j, result, dim_q0)

        self.x = x
        self.M = M
        self.dim_q0 = dim_q0
        self.dim_q1 = dim_q1
        self.num_qubits = self.dim_q0 + self.dim_q1
        self._gate = None

        self._permutation_gate = BasisPermutationGate(action, dim_q0 + dim_q1)

    def get_native(self) -> qk.circuit.Gate:
        if not self._gate:
            self._gate = self._permutation_gate.get_native()

        return self._gate

    def mod_pow(base, exponent, modulus):
        if modulus == 1:
            return 0

        result = 1
        base = base % modulus
        while exponent > 0:
            if exponent % 2 == 1:
                result = (result * base) % modulus

            exponent = exponent >> 1
            base = (base * base) % modulus
        return result
