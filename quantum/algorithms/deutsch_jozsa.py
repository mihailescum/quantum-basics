from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.result import Counts

from enum import Enum

from collections.abc import Callable


class DeutschJozsa:
    class Result(Enum):
        Error = 1
        Constant = 2
        Balanced = 3

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
        label_zero = "0" * self.n
        if label_zero not in counts:
            counts[label_zero] = 0

        num_shots = sum(counts.values())

        result = DeutschJozsa.Result.Error
        if counts[label_zero] == num_shots:
            result = DeutschJozsa.Result.Constant
        elif counts[label_zero] == 0:
            result = DeutschJozsa.Result.Balanced

        return result
