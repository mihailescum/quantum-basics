from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.result import Counts

from enum import Enum

from quantum.algorithms import AlgorithmException

from collections.abc import Callable


class DeutschJozsa:
    class Result(Enum):
        Constant = 1
        Balanced = 2

    def __init__(self, f: QuantumCircuit | Instruction, n: int):
        self.f = f
        self.n = n

        self.qc = None

    def run(self, run_circuit: Callable[[QuantumCircuit], Counts]) -> Result:
        if not self.qc:
            self.qc = self.build_circuit()

        qc_result_counts = run_circuit(self.qc)
        result = self._analyze_counts(qc_result_counts)
        return result

    def build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n + 1, self.n)

        qc.x(self.n)

        qc.h(range(self.n + 1))
        qc.compose(self.f, inplace=True)
        qc.h(range(self.n))

        qc.measure(range(self.n), range(self.n))

        self.qc = qc
        return qc

    def _analyze_counts(self, counts: Counts) -> Result:
        zero_num_counts = counts.int_outcomes().get(0, 0)
        num_shots = counts.shots()

        if zero_num_counts == num_shots:
            return DeutschJozsa.Result.Constant
        elif zero_num_counts == 0:
            return DeutschJozsa.Result.Balanced

        raise AlgorithmException("The function is neither balanced nor constant")
