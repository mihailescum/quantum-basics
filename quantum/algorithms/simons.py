import qiskit as qk

import numpy as np

from quantum.utils import lower_triangular_form
from quantum.algorithms import AlgorithmException

from collections.abc import Callable


class Simons:
    def __init__(self, f: qk.circuit.Gate, n: int, m: int) -> None:
        self.f = f
        self.n = n
        self.m = m

        self.qc = None

    def build_circuit(self) -> qk.QuantumCircuit:
        qc = qk.QuantumCircuit(self.n + self.m, self.n)
        qc.h(range(self.n))
        qc.append(self.f, range(self.n + self.m))
        qc.h(range(self.n))
        qc.measure(range(self.n), range(self.n))

        self.qc = qc
        return qc

    def run(self, run_circuit: Callable[[qk.QuantumCircuit], qk.result.Counts]) -> int:
        if not self.qc:
            self.qc = self.build_circuit()

        qc_result_counts = run_circuit(self.qc)
        result = self._analyze_counts(qc_result_counts)
        return result

    def _analyze_counts(self, counts: qk.result.Counts) -> int:
        bit_matrix = Simons._counts_to_bitmatrix(counts).astype(np.float64)
        bit_matrix = lower_triangular_form(bit_matrix)
        rank = np.where(bit_matrix.any(axis=1))[0].size

        if rank < self.n - 1:
            raise AlgorithmException(
                "Could not gather enough linearly independent samples."
            )
        elif rank == self.n:
            return 0
        else:
            bit_matrix = bit_matrix[: self.n - 1]

            solution = np.zeros(self.n)
            solution[-1] = 1

            for i in range(rank - 1, -1, -1):
                solution[i] = -np.dot(bit_matrix[i], solution) / bit_matrix[i, i]
            solution_base_2 = np.mod(solution.astype(np.int64), 2)
            s = int("".join([str(x) for x in solution_base_2]), 2)
            return s

    def _counts_to_bitmatrix(counts: qk.result.Counts) -> np.ndarray:
        x = np.array(list(counts.keys()), dtype=bytes)
        matrix = x.view("S1").reshape((x.size, -1)).astype(np.int8)
        return matrix
