from qiskit.quantum_info import QubitSparsePauli

# From Qiskit and adapted to only observables

def get_scale_factors(noise_model, observables):
    scale_factors = []
    for obs in observables:
        scale_factors.append(1/noise_model.pauli_fidelity(QubitSparsePauli(obs)))
    return scale_factors