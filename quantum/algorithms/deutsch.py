from qiskit import QuantumCircuit

def build_circuit(oracle: QuantumCircuit) -> QuantumCircuit:
    qc = QuantumCircuit(2, 1)

    qc.x(1)
    qc.h(range(2))
    qc.barrier()

    qc.compose(oracle, inplace=True)

    qc.barrier()

    qc.h(0)

    qc.measure(0, 0)
    return qc