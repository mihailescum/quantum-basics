from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.result import Counts
from qiskit_aer import AerSimulator

from enum import Enum
import typing


class DeutschJozsa:
    class Result(Enum):
        Error = 1
        Constant = 2
        Balanced = 3

    def __init__(self, f: QuantumCircuit | Instruction, n: int):
        self.f = f
        self.n = n

        self.qc = None

    def run(self, simulator: AerSimulator, **run_options: typing.Any) -> Result:
        if not self.qc:
            self.qc = self.build_circuit()

        simulation_result = simulator.run(
            self.qc, parameter_binds=None, run_options=run_options
        ).result()
        result = self._analyze_counts(simulation_result.get_counts())
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
