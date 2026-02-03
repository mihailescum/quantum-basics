import qiskit as qk

from quantum.gates import BasisPermutationGate
from quantum.utils import reduce_basis_state, combine_basis_state

from collections.abc import Callable


class AutoOracleGate:
    def __init__(self, f: Callable[[int], int], n: int, m: int) -> None:
        """Implements an oracle for a function f:{0,1}^n \mapsto {0,1}^m"""

        def f_extended_to_permutation(x):
            q_0, q_1 = reduce_basis_state(x, n)
            return combine_basis_state(q_0, q_1 ^ f(q_0), n)

        self.n = n
        self.m = m
        self._gate = None

        self._permutation_gate = BasisPermutationGate(f_extended_to_permutation, n + m)

    def get_native(self) -> qk.circuit.Gate:
        if not self._gate:
            self._gate = self._permutation_gate.get_native()

        return self._gate
