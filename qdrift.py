from qiskit.circuit import QuantumCircuit, Parameter
import numpy as np

def qdrift_sample_matrix(chain_length, N, m, g, lamb):
    m_matrices = np.zeros((chain_length, 3))
    m_matrices[:, 0] = np.arange(1, chain_length+1)
    m_probabilities = m*np.ones(chain_length)
    g_matrices = np.zeros((chain_length-1, 3))
    g_matrices[:, 1] = np.arange(1, chain_length)
    g_probabilities = g*np.ones(chain_length-1)
    lamb_matrices = np.zeros((chain_length-1, 3))
    lamb_matrices[:, 2] = np.arange(1, chain_length)
    lamb_probabilities = lamb*np.ones(chain_length-1)
    possible_terms = np.vstack((m_matrices, g_matrices, lamb_matrices)).astype(int)
    probabilities = np.concatenate((m_probabilities, g_probabilities, lamb_probabilities))
    probabilities /= np.sum(probabilities)  # Normalize probabilities
    samples = np.random.choice(np.arange(len(possible_terms)), size=N, p=probabilities)
    return possible_terms[samples]