import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import time

##############################################################################
# Code from https://github.com/PennyLaneAI/qml/blob/master/demonstrations/tutorial_classical_shadows.py
# A classical shadow is a collection of :math:`N` individual snapshots :math:`\hat{\rho}_i`.
# Each snapshot is obtained with the following procedure:
#
# 1. The quantum state :math:`\rho` is prepared with a circuit.
# 2. A unitary :math:`U` is randomly selected from the ensemble and applied to :math:`\rho`.
# 3. A computational basis measurement is performed.
# 4. The snapshot is recorded as the observed eigenvalue :math:`1,-1` for :math:`|0\rangle,|1\rangle`, respectively, and the index of the randomly selected unitary :math:`U`.
#
# To obtain a classical shadow using PennyLane, we design the ``calculate_classical_shadow``
# function below.
# This function obtains a classical shadow for the state prepared by an input 
# ``circuit_template``.


def calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits):
    """
    Given a circuit, creates a collection of snapshots consisting of a bit string
    and the index of a unitary operation.
    Args:
        circuit_template (function): A Pennylane QNode.
        params (array): Circuit parameters.
        shadow_size (int): The number of snapshots in the shadow.
        num_qubits (int): The number of qubits in the circuit.
    Returns:
        Tuple of two numpy arrays. The first array contains measurement outcomes (-1, 1)
        while the second array contains the index for the sampled Pauli's (0,1,2=X,Y,Z).
        Each row of the arrays corresponds to a distinct snapshot or sample while each
        column corresponds to a different qubit.
    """
    # applying the single-qubit Clifford circuit is equivalent to measuring a Pauli
    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]

    # sample random Pauli measurements uniformly, where 0,1,2 = X,Y,Z
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))
    outcomes = np.zeros((shadow_size, num_qubits))

    for ns in range(shadow_size):
        # for each snapshot, add a random Pauli observable at each location
        obs = [unitary_ensemble[int(unitary_ids[ns, i])](i) for i in range(num_qubits)]
        outcomes[ns, :] = circuit_template(params, observable=obs)

    # combine the computational basis outcomes and the sampled unitaries
    return (outcomes, unitary_ids)
