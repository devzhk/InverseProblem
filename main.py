import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import time

from utils.qml_classical_shadow import calculate_classical_shadow, shadow_state_reconstruction

np.random.seed(2022)


num_qubits = 2
dev = qml.device('default.qubit', wires=num_qubits, shots=1)


@qml.qnode(device=dev)
def local_qubit_rotation_circuit(params, **kwargs):
    observables = kwargs.pop("observable")
    for w in dev.wires:
        qml.RY(params[w], wires=w)

    return [qml.expval(o) for o in observables]


num_snapshots = 10
params = np.array([1.0, 0.0])       # quantum state

shadow = calculate_classical_shadow(local_qubit_rotation_circuit, params,
                                    num_snapshots, num_qubits=num_qubits)

