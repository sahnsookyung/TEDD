from .context import molecule_env
import gym
import numpy as np

from rdkit import Chem


# TESTS

# Class testing purpose methods
def matrices_to_mol(A, E, F, node_feature_list, edge_feature_list):
    """
    Converts matrices A, E, F to rdkit mol object
    :param A: adjacency matrix, numpy array, dim k x k. Entries are either 0 or 1
    :param E: edge attribute matrix, numpy array, dim k x k x de. Entries are
    edge wise probabilities
    :param F: node attribute matrix, numpy array, dim k x dn. Entries are node
    wise probabilities
    :param node_feature_list: list of d_n elements that specifies possible
    atomic symbols
    :param edge_feature_list: list of d_e elements that specifies possible rdkit
    bond types
    :return: rdkit mol object
    """

    k = A.shape[0]

    rw_mol = Chem.RWMol()

    matrix_atom_idx_to_mol_atom_idx = {}
    for l in range(k):
        if A[l, l] == 1.0:
            atom_feature = F[l, :]
            atom_symbol = node_feature_list[np.argmax(atom_feature)]
            atom = Chem.Atom(atom_symbol)
            mol_atom_idx = rw_mol.AddAtom(atom)
            matrix_atom_idx_to_mol_atom_idx[l] = mol_atom_idx

    matrix_atom_idxes = matrix_atom_idx_to_mol_atom_idx.keys()
    for i in range(len(matrix_atom_idxes) - 1):
        for j in range(i + 1, len(matrix_atom_idxes)):
            if A[i, j] == 1.0:
                bond_feature = E[i, j, :]
                bond_type = edge_feature_list[np.argmax(bond_feature)]
                begin_atom_idx = matrix_atom_idx_to_mol_atom_idx[i]
                end_atom_idx = matrix_atom_idx_to_mol_atom_idx[j]
                rw_mol.AddBond(begin_atom_idx, end_atom_idx, order=bond_type)

    return rw_mol.GetMol()


# Define external reward function for MoleculeEnvironment creation
def reward_func(action_space, observation_space):
    action_space = action_space
    observation_space = observation_space
    return 1


# End of test setup

# Beginning of tests
env = molecule_env.MoleculeEnvironment(reward_func(), n_iterations=2, max_iterations=10)

# TODO add tests for rewards, that it is an integer and it either increases or decreases

def test_example():
    assert issubclass(type(env), gym.Env)


def test_seed():
    seed = env.seed()
    assert isinstance(seed, list)
    # assert the list content is a discrete number


def test_reset():
    env.reset() # Mandatory resetting prior to running environment required

    env.step(np.array([0, 0, 1, 0]))
    env.reset()

    totalAtoms = env.total_atoms
    totalBonds = env.total_bonds
    InterimReward = env.interim_reward
    CumulativeReward = env.cumulative_reward
    counter = env.counter

    assert totalAtoms == 1
    assert totalBonds + InterimReward + CumulativeReward + counter == 0


def test_step():
    env.reset()
    assert Chem.MolToSmiles(env.mol) == "C"
    env.step([2, 0, 1, 0])
    assert Chem.MolToSmiles(env.mol) == "CO"
    env.step([1, 0, 2, 0])
    assert Chem.MolToSmiles(env.mol) == "NCO"
    ob, reward, done, info = env.step([2, 2, 3, 1])
    assert Chem.MolToSmiles(env.mol) == "O=NCO"

def test_render():
    # Test render method
    pass

