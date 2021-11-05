import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import time

from utils.qml_classical_shadow import calculate_classical_shadow, shadow_state_reconstruction
from utils.helper import operator_2_norm

np.random.seed(2022)


num_qubits = 2
dev = qml.device('default.qubit', wires=num_qubits, shots=1)


@qml.qnode(device=dev)
def bell_state_circuit(params, **kwargs):
    observables = kwargs.pop("observable")

    qml.Hadamard(0)
    qml.CNOT(wires=[0, 1])

    return [qml.expval(o) for o in observables]


num_snapshots = 1000
params = []

shadow = calculate_classical_shadow(
    bell_state_circuit, params, num_snapshots, num_qubits
)

shadow_state = shadow_state_reconstruction(shadow)
print(np.round(shadow_state, decimals=6))

bell_state = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])

print(operator_2_norm(bell_state - shadow_state))