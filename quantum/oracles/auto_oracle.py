import numpy as np
import itertools

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum.utils import combine_basis_state

from collections.abc import Callable


class AutoOracle:
    def __init__(self, f: Callable[[int], int], n: int, m: int) -> None:
        """Implements an oracle for a function f:{0,1}^n \mapsto {0,1}^m"""
        self.f = f
        self.n = n
        self.m = m
        self.gate = None

        self._build_gate()

    def _build_gate(self) -> None:
        if not self.gate:
            action = [self.f(x) for x in range(0, 2 << (self.n - 1))]

            qc = QuantumCircuit(self.n + self.m)
            dims = 2 << (self.n + self.m - 1)
            self._matrix = np.array(
                [
                    Statevector.from_int(
                        combine_basis_state(j, k ^ action[j], self.n), dims
                    )
                    for (k, j) in itertools.product(
                        range(2 << (self.m - 1)), range(2 << (self.n - 1))
                    )
                ]
            )
            qc.unitary(self._matrix, qubits=range(self.n + self.m), label="Oracle")
            self.gate = qc  # .to_instruction()
