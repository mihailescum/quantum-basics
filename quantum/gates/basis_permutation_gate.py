import numpy as np
import numba

import qiskit as qk

from quantum.utils import get_bitmask

from collections.abc import Callable
from typing import Tuple, List


class BasisPermutationGate:
    def __init__(self, f: Callable[[int], int], num_qubits: int) -> None:
        """Implements a permutation given by an injective function f:{0,1}^n mapsto {0,1}^n"""
        self.f = f
        self.num_qubits = num_qubits
        self._gate = None

    def get_native(self) -> qk.circuit.Instruction:
        """Builds a gate which is equivalent to the permutation matrix obtained by

        ´´´
        matrix = np.zeros((2**self.num_qubits, 2**self.num_qubits), dtype=int)
        for v in range(matrix.shape[0]):
            matrix[self.f(v), v] = 1
        ´´´
        """
        if not self._gate:
            qc = qk.QuantumCircuit(self.num_qubits)

            cycles = BasisPermutationGate._permutation_get_cycles(
                self.f, self.num_qubits
            )
            for cycle in cycles:
                transpositions = (
                    BasisPermutationGate._permutation_cycle_get_transpositions(
                        cycle.astype(np.int64)
                    )
                )
                for transposition in transpositions:
                    self._add_swap_basis_states_circuit(
                        qc, transposition[0], transposition[1]
                    )  # ,
                    # range(self.num_qubits),
                    # )
            self._gate = qc.to_instruction()
        return self._gate

    # @numba.jit(nopython=True)
    def _permutation_get_cycles(f: Callable[[int], int], num_qubits: int):
        indices = np.arange(2**num_qubits)
        current_cycle = np.full(2**num_qubits, fill_value=-1, dtype=int)
        current_index = 0
        v = indices[0]

        while True:
            current_cycle[current_index] = v
            current_index += 1
            indices[v] = 0
            v = f(v)
            if v == current_cycle[0]:
                invalid = np.where(current_cycle == -1)[0]
                if invalid.size > 0:
                    yield current_cycle[: invalid[0]]
                else:
                    yield current_cycle

                current_cycle[:] = -1

                remaining_indices = np.nonzero(indices)[0]
                if remaining_indices.size > 0:
                    v = remaining_indices[0]
                    current_index = 0
                else:
                    return

    def _permutation_cycle_get_transpositions(
        cycle: np.ndarray,
    ) -> List[Tuple[int, int]]:
        return [(cycle[x], cycle[x + 1]) for x in range(cycle.size - 2, -1, -1)]

    def _add_flip_single_bit_circuit(
        self, qc: qk.QuantumCircuit, a: int, bit: int
    ) -> None:
        # qc = qk.QuantumCircuit(self.num_qubits)
        a_mask = get_bitmask(a, self.num_qubits)
        a_mask[bit] = 1

        controls = np.concat([np.arange(bit), np.arange(bit + 1, self.num_qubits)])
        controls_to_flip = np.where(a_mask == 0)[0]
        if controls_to_flip.size > 0:
            qc.x(controls_to_flip)
        qc.mcx(controls.tolist(), bit)
        if controls_to_flip.size > 0:
            qc.x(controls_to_flip)
        # return qc

    def _add_swap_basis_states_circuit(
        self, qc: qk.QuantumCircuit, a: int, b: int
    ) -> None:
        if a == b:
            return

        bits_to_flip = np.where(get_bitmask(a ^ b, self.num_qubits) == 1)[0]

        for bit in bits_to_flip[:-1]:
            self._add_flip_single_bit_circuit(qc, a, bit)
            a ^= 2**bit

        self._add_flip_single_bit_circuit(qc, a, bits_to_flip[-1])

        for bit in bits_to_flip[-2::-1]:
            self._add_flip_single_bit_circuit(qc, a, bit)
            a ^= 2**bit
