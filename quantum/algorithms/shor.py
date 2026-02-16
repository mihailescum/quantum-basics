import numpy as np
from math import gcd

from random import randint
from fractions import Fraction

import qiskit as qk

from collections.abc import Callable

from sympy import evaluate

from quantum.algorithms import AlgorithmException
from quantum.gates import CModularInplaceMultiplicationGate


class Shor:
    def __init__(self, N: int):
        self.N = N
        self.L = None

    def run(
        self, run_circuit: Callable[[qk.QuantumCircuit], dict[int, int]]
    ) -> tuple[int, int]:
        if self.N % 2 == 0:
            return (2, int(self.N / 2))

        # TODO: Test that N is not a power of primes
        while True:
            a = randint(3, self.N - 1)
            try:
                result = self._run_single_pass(a, run_circuit)
                break
            except AlgorithmException:
                continue
        return result

    def _run_single_pass(
        self, a: int, run_circuit: Callable[[qk.QuantumCircuit], dict[int, int]]
    ) -> tuple[int, int]:
        trivial_guess = gcd(a, self.N)
        if trivial_guess != 1:
            print("Guessed a non trivial divisor!")
            # return (trivial_guess, int(self.N / trivial_guess))
            raise AlgorithmException

        qc = self._build_circuit(a)
        qc_result_counts = run_circuit(qc)
        guesses = self._extract_guesses(qc_result_counts)

        result = None

        for guess in guesses:
            if a**guess % self.N == 1:
                try:
                    result = self._evaluate_guess(a, guess)
                except AlgorithmException:
                    break

        if result is not None:
            return result
        else:
            raise AlgorithmException

    def _build_circuit(self, a: int) -> qk.QuantumCircuit:
        n = int(np.ceil(np.log2(self.N)))
        self.L = 2 * n
        x_reg = qk.circuit.QuantumRegister(1, "x")
        b_reg = qk.circuit.QuantumRegister(n, "b")
        ancilla_reg = qk.circuit.AncillaRegister(n + 2, "a")
        meas_reg = qk.circuit.ClassicalRegister(self.L)
        qc = qk.QuantumCircuit(x_reg, b_reg, ancilla_reg, meas_reg)
        qc.h(x_reg)
        qc.x(b_reg)

        factors = np.empty(self.L, dtype=int)
        factors[0] = a
        for bit in range(1, self.L):
            factors[bit] = (factors[bit - 1] * factors[bit - 1]) % self.N
        factors = factors[::-1]

        for bit in range(self.L):
            if factors[bit] == 1:
                continue

            mult = CModularInplaceMultiplicationGate(int(factors[bit]), self.N, n)
            qc.append(mult.get_native(), [x_reg[0]] + b_reg[:] + ancilla_reg[:])
            for j in range(bit):
                theta = -2 * np.pi / (2 ** (j + 2))  # (bit - j + 1))
                with qc.if_test((meas_reg[j], 1)):
                    qc.rz(theta, x_reg)
            qc.h(x_reg)
            qc.measure(x_reg, meas_reg[bit])

            qc.reset(x_reg)
            qc.h(x_reg)

        return qc

    def _extract_guesses(self, outcomes: dict[int, int]) -> list[int]:
        guesses = []
        for outcome in {k: v for k, v in outcomes.items() if k not in [0, 1]}:
            phase = outcome / (pow(2, self.L))
            frac = Fraction(phase).limit_denominator(self.N)
            guesses.append(frac.denominator)
        return guesses

    def _evaluate_guess(self, a: int, guess: int) -> bool:
        if guess % 2 != 0 or (a ** int(guess / 2) + 1) % self.N == 0:
            raise AlgorithmException

        d1 = gcd(a ** int(guess / 2) + 1, self.N)
        d2 = gcd(a ** int(guess / 2) - 1, self.N)
        if d1 * d2 == self.N:
            return (d1, d2)
        else:
            raise AlgorithmException
