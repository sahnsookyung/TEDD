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
def reward_func():
    return 1


# End of test setup

# Beginning of tests
env = molecule_env.MoleculeEnvironment(reward_func(), n_iterations=2, max_iterations=10)


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

# # Test molecule generation and verify the final results are correct
#     ob, reward, done, info = env.step(np.array([2, 0, 1, 0]))
#     print(Chem.MolToSmiles(env.mol))
#     print("info* ", info)
#
#     ob, reward, done, info = env.step(np.array([2, 0, 2, 0]))
#     print(Chem.MolToSmiles(env.mol))
#     print("info* ", info)
#
#     ob, reward, done, info = env.step(np.array([2, 0, 3, 0]))
#     print(Chem.MolToSmiles(env.mol))
#
#     env.step(np.array([2, 0, 4, 0]))
#     print(Chem.MolToSmiles(env.mol))
#
#     env.mol
#
#     # # add carbon
#     # env.step(np.array([1, 0, 0]), 'add_atom')
#     # # add double bond between carbon 1 and carbon 2
#     # env.step(np.array([[0, 1, 0]]), 'modify_bond')
#     # # add carbon
#     # env.step(np.array([1, 0, 0]), 'add_atom')
#     # # add single bond between carbon 2 and carbon 3
#     # env.step(np.array([[0, 0, 0], [1, 0, 0]]), 'modify_bond')
#     # # add oxygen
#     # env.step(np.array([0, 0, 1]), 'add_atom')
#     # # add single bond between carbon 3 and oxygen
#     # env.step(np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]), 'modify_bond')
#     #
#     # env.render()
#     # assert Chem.MolToSmiles(env.mol, isomericSmiles=True) == 'C=CCO'
#     #
#     possible_atoms = ['C', 'N', 'O', 'S', 'Cl']
#     possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
#                       Chem.rdchem.BondType.TRIPLE]
#     # # test get_matrices 1
#     print("Testing get_matrices")
#     A, E, F = env.get_matrices()
#     print("A:")
#     print(A)
#     print("E:")
#     print(E)
#     print("F:")
#     print(F)
#     #
#     print(Chem.MolToSmiles(matrices_to_mol(A, E, F, possible_atoms,
#                                            possible_bonds), isomericSmiles=True))
#     # assert Chem.MolToSmiles(matrices_to_mol(A, E, F, possible_atoms,
#     #                                         possible_bonds), isomericSmiles=True) \
#     #        == 'C=CCO'
#
#     # print("Beginning valency test")
#     # # molecule check valency test
#     # env = MoleculeEnvironment(reward_func())
#     # # add carbon
#     # r = env.step(np.array([1, 0, 0]), 'add_atom')
#     # print(Chem.MolToSmiles(env.mol))
#     # # add oxygen
#     # r = env.step(np.array([0, 0, 1]), 'add_atom')
#     # print(Chem.MolToSmiles(env.mol))
#     # # add single bond between carbon and oxygen 1
#     # r = env.step(np.array([[1, 0, 0]]), 'modify_bond')
#     # print(Chem.MolToSmiles(env.mol))
#     #
#     # print(r)
#     #
#     # # add oxygen
#     # r = env.step(np.array([0, 0, 1]), 'add_atom')
#     # print(Chem.MolToSmiles(env.mol))
#     # # add single bond between carbon and oxygen 2
#     # r = env.step(np.array([[1, 0, 0], [0, 0, 0]]), 'modify_bond')
#     # print(Chem.MolToSmiles(env.mol))
#     #
#     # # add oxygen
#     # r = env.step(np.array([0, 0, 1]), 'add_atom')
#     # print(Chem.MolToSmiles(env.mol))
#     # # add single bond between carbon and oxygen 3
#     # r = env.step(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), 'modify_bond')
#     # print(Chem.MolToSmiles(env.mol))
#     #
#     # # add oxygen
#     # r = env.step(np.array([0, 0, 1]), 'add_atom')
#     # print(Chem.MolToSmiles(env.mol))
#     # # add single bond between carbon and oxygen 4
#     # r = env.step(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
#     #              'modify_bond')
#     # print(Chem.MolToSmiles(env.mol))
#     # # add oxygen
#     # r = env.step(np.array([0, 0, 1]), 'add_atom')
#     # print(Chem.MolToSmiles(env.mol))
#     # assert r == 1
#     # # add single bond between carbon and oxygen 4. This exceeds valency on C
#     # r = env.step(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
#     #              'modify_bond')
#     # print(Chem.MolToSmiles(env.mol))
#     # print(r)
#     # assert r == -1
#     #
#     # # test get_matrices 2
#     # A, E, F = env.get_matrices()
#     # assert Chem.MolToSmiles(matrices_to_mol(A, E, F, possible_atoms,
#     #                                         possible_bonds), isomericSmiles=True) \
#     #        == 'OC(O)(O)(O)O'
