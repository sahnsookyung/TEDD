from .context import molecule_env
import gym
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


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
def reward_func(moleculeEnvironment):
    action_space = moleculeEnvironment.action_space
    observation_space = moleculeEnvironment.observation_space
    return 1


def reward_func_alternate(moleculeEnvironment, divisor=2):
    # returns positive and negative reward of 1, alternating between the two. Modify divisor to change the frequency of
    # positive rewards per negative reward

    counter = moleculeEnvironment.step_counter
    if (counter % divisor) == 0:
        counter += 1
        return -1
    else:
        counter += 1
        return 1


def check_chemical_validity(self):
    """
    Checks the chemical validity of the mol object. Existing mol object is
    not modified
    :return: True if chemically valid, False otherwise
    """
    s = Chem.MolToSmiles(self.mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
    if m:
        return True
    else:
        return False


def check_valency(self):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(self.mol,
                         sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except ValueError:
        return False


# End of test setup

# Beginning of tests
env = gym.make('molecule-v0')
env.init(reward_func, n_iterations=2, max_iterations=10, reward_func_sees_all=True)


# env = molecule_env.MoleculeEnvironment(reward_func, n_iterations=2, max_iterations=10)

# TODO add tests for rewards, that it is an integer and it either increases or decreases

def test_example():
    assert issubclass(type(env), gym.Env)


def test_seed():
    seed = env.seed()
    assert isinstance(seed, list)
    # assert the list content is a discrete number


def test_reset():
    env.reset()

    totalAtoms = env.get_num_atoms()
    totalBonds = env.get_num_bonds()
    InterimReward = env.interim_reward
    CumulativeReward = env.cumulative_reward
    counter = env.step_counter

    assert totalAtoms == 1
    assert totalBonds + InterimReward + CumulativeReward + counter == 0


def test_reset_given_molecule():
    env.reset("NCO")

    totalAtoms = env.get_num_atoms()
    totalBonds = env.get_num_bonds()
    InterimReward = env.interim_reward
    CumulativeReward = env.cumulative_reward
    counter = env.step_counter

    assert totalAtoms == 3
    assert InterimReward + CumulativeReward + counter == 0
    assert totalBonds == 2


def test_step():
    env.reset()
    assert Chem.MolToSmiles(env.mol) == "C"
    env.step([2, 0, 1, 0])
    assert Chem.MolToSmiles(env.mol) == "CO"
    env.step([1, 0, 2, 0])
    assert Chem.MolToSmiles(env.mol) == "NCO"
    ob, reward, done, info = env.step([2, 2, 3, 1])
    assert Chem.MolToSmiles(env.mol) == "O=NCO"


def test_reward():
    env.reset()
    assert Chem.MolToSmiles(env.mol) == "C"

    ob, reward, done, info = env.step([2, 0, 1, 0])
    assert reward == 1
    assert 'interim_reward' not in info
    assert info['cumulative_reward'] == 1
    assert Chem.MolToSmiles(env.mol) == "CO"

    ob, reward, done, info = env.step([1, 0, 2, 0])
    assert reward == 1
    assert 'interim_reward' in info and info['interim_reward'] == 2
    assert info['cumulative_reward'] == 2
    assert Chem.MolToSmiles(env.mol) == "NCO"

    ob, reward, done, info = env.step([2, 2, 3, 1])
    assert reward == 1
    assert 'interim_reward' not in info
    assert info['cumulative_reward'] == 3
    assert Chem.MolToSmiles(env.mol) == "O=NCO"

    test_reset()


def test_molecule_rollback():
    env2 = gym.make('molecule-v0')
    env2.init(reward_func_alternate, n_iterations=2, max_iterations=10, molecule_rollback=True, reward_func_sees_all=True)

    env2.reset()
    assert Chem.MolToSmiles(env2.mol) == "C"

    ob, reward, done, info = env2.step([2, 0, 1, 0])
    assert reward == 1
    assert 'interim_reward' not in info
    assert info['cumulative_reward'] == 1
    assert Chem.MolToSmiles(env2.mol) == "CO"

    ob, reward, done, info = env2.step([1, 0, 2, 0])
    assert reward == -1
    assert 'interim_reward' in info and info['interim_reward'] == 0
    assert info['cumulative_reward'] == 0
    assert Chem.MolToSmiles(env2.mol) == "CO"

    ob, reward, done, info = env2.step([1, 0, 2, 0])
    assert reward == 1
    assert 'interim_reward' not in info
    assert info['cumulative_reward'] == 1
    assert Chem.MolToSmiles(env2.mol) == "NCO"

    ob, reward, done, info = env2.step([2, 2, 3, 1])
    assert reward == -1
    assert 'interim_reward' in info and info['interim_reward'] == 0
    assert info['cumulative_reward'] == 0
    assert Chem.MolToSmiles(env2.mol) == "NCO"

    test_reset()


def test_render():
    # Paracetamol
    env.reset("CC(=O)Nc1ccc(O)cc1")
    env.render()

    # Ibuprofen
    env.reset("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")
    env.render()

    # Phospholipase A2 A2 isozyme CM-I, snake venom known to have anti-tumor properties
    env.reset("CCCCCCCCNC(=O)Oc1cccc(OC(=O)C(c2ccccc2)(c2ccccc2)c2ccccc2)c1")
    env.render()

    # Bergenin (cuscutin), a drug that shows a potent immunomodulatory effect
    env.reset("OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2")
    env.render()

    # amphetamine, a powerful stimulator of the central nervous system
    env.reset("CC(N)Cc1ccccc1")
    env.render()

    # Squalene, an important candidate for COVID-19 vaccines, isomeric smiles
    env.reset("CC(=CCC/C(=C/CC/C(=C/CC/C=C(/CC/C=C(/CCC=C(C)C)\C)\C)/C)/C)C")
    env.render()

    # Squalene, canonical smiles
    env.reset("CC(=CCCC(=CCCC(=CCCC=C(C)CCC=C(C)CCC=C(C)C)C)C)C")
    env.render()

    # Halichondrin B, a molecule with exquisite anticancer properties isolated from the marine sponge Halichondria okadai
    env.reset(
        "OCC(O)CC(O)[C@@H]1C[C@@H]2O[C@@]3(C[C@H](C)[C@@H]2O1)C[C@H](C)[C@@H]4O[C@]%10(C[C@@H]4O3)C[C@H]%11O[C@H]%12[C@H](C)[C@H]%13OC(=O)C[C@H]8CC[C@@H]9O[C@H]7[C@H]6O[C@]5(O[C@H]([C@@H]7O[C@@H]6C5)[C@H]9O8)CC[C@H]%15C/C(=C)[C@H](CC[C@H]%14C[C@@H](C)\C(=C)[C@@H](C[C@@H]%13O[C@H]%12C[C@H]%11O%10)O%14)O%15")
    env.render()

