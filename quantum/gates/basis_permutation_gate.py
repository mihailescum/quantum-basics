import numpy as np

from qiskit import QuantumCircuit

from quantum.utils import get_bitmask

from collections.abc import Callable


class BasisPermutationGate:
    def __init__(self, f: Callable[[int], int], num_qubits: int) -> None:
        """Implements a permutation given by an injective function f:{0,1}^n mapsto {0,1}^n"""
        self.f = f
        self.num_qubits = num_qubits
        self.gate = None

        self._build_gate()

    def _build_gate(self) -> None:
        if not self.gate:
            qc = QuantumCircuit(self.num_qubits)

            for cycle in self._permutation_get_cycles():
                transpositions = (
                    BasisPermutationGate._permutation_cycle_get_transpositions(cycle)
                )
                for transposition in transpositions:
                    qc.append(
                        self._swap_basis_states_circuit(
                            transposition[0], transposition[1]
                        ),
                        range(self.num_qubits),
                    )
            self.gate = qc

    def _permutation_get_cycles(self):
        indices = np.arange(2**self.num_qubits).tolist()
        current_cycle = []
        v = indices[0]

        while len(indices) > 0:
            current_cycle.append(v)
            indices.remove(v)
            v = self.f(v)
            if v == current_cycle[0]:
                yield current_cycle
                current_cycle = []
                if len(indices) > 0:
                    v = indices[0]

    def _permutation_cycle_get_transpositions(cycle):
        for x in range(len(cycle) - 2, -1, -1):
            yield (cycle[x], cycle[x + 1])

    def _flip_single_bit_circuit(self, a, bit):
        qc = QuantumCircuit(self.num_qubits)
        a_mask = get_bitmask(a, self.num_qubits)
        a_mask[bit] = 1

        controls = np.concat([np.arange(bit), np.arange(bit + 1, self.num_qubits)])
        controls_to_flip = np.where(a_mask == 0)[0]
        if controls_to_flip.size > 0:
            qc.x(controls_to_flip)
        qc.mcx(controls.tolist(), bit)
        if controls_to_flip.size > 0:
            qc.x(controls_to_flip)
        return qc

    def _swap_basis_states_circuit(self, a, b):
        qc = QuantumCircuit(self.num_qubits)
        if a == b:
            return qc

        bits_to_flip = np.where(get_bitmask(a ^ b, self.num_qubits) == 1)[0]

        for bit in bits_to_flip[:-1]:
            qc.append(self._flip_single_bit_circuit(a, bit), range(self.num_qubits))
            # flip_single_bit_circuit(a, width, bit, qc)
            a ^= 2**bit

        reversed = qc.reverse_ops()
        qc.append(
            self._flip_single_bit_circuit(a, bits_to_flip[-1]), range(self.num_qubits)
        )
        qc.append(reversed, range(self.num_qubits))
        return qc
