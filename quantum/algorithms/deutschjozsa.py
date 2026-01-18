from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum.utils import combine_basis_state


class DeutschJozsa:
    def __init__(self, f, n: int) -> None:
        self.f = f
        self.n = n

        self.qc = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n + 1, self.n)

        dims = 2 << self.n
        init = Statevector.from_int(combine_basis_state(0, 1, self.n), dims)
        qc.initialize(init)
        qc.h(range(self.n + 1))
        # qc.barrier()

        qc.compose(self.f, inplace=True)

        # qc.barrier()

        qc.h(range(self.n))

        qc.measure(range(self.n), range(self.n))
        return qc

    def analyze_counts(self, counts) -> str:
        label_zero = "0" * self.n
        if label_zero not in counts:
            counts[label_zero] = 0

        num_shots = sum(counts.values())

        result = "error"
        if counts[label_zero] == num_shots:
            result = "constant"
        elif counts[label_zero] == 0:
            result = "balanced"

        return result
