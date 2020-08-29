import gym
import numpy as np
from rdkit import Chem  # for debug


class MoleculeEnvironment():
    def __init__(self, reward_function):
        """
        :param reward_function: A function that returns a reward value
        :param possible_atom_types: The elements of the periodic table used in this class
        """

        # Default values
        self.possible_atoms = ['C', 'N', 'O']

        self.possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                               Chem.rdchem.BondType.TRIPLE]
        self.max_possible_atoms = 5


        self.mol = Chem.RWMol()
        self.possible_atom_types = np.array(self.possible_atoms)  # dim d_n. Array that
        # contains the possible atom symbols strs
        self.possible_bond_types = np.array(self.possible_bonds, dtype=object)  # dim
        # d_e. Array that contains the possible rdkit.Chem.rdchem.BondType objects
        self.current_atom_idx = None
        self.total_atoms = 0
        self.total_bonds = 0
        self.reward_function = reward_function

        self.action_space = gym.spaces.MultiDiscrete(
            [len(self.possible_atom_types), self.max_possible_atoms,
             self.max_possible_atoms, len(self.possible_bonds)]
        )


    def reset(self):
        self.mol = Chem.RWMol()
        self.current_atom_idx = None

        self._add_atom([0, 0, 0, 0])

        self.total_atoms = 0
        self.total_bonds = 0

    def step(self, action, action_type):
        """
    Perform a given action
    :param action:
    :param action_type:
    :return: reward of 1 if resulting molecule graph does not exceed valency,
    -1 if otherwise
    """
        if action_type == 'add_atom':
            self._add_atom(action)
        elif action_type == 'modify_bond':
            self._modify_bond(action)
        else:
            raise ValueError('Invalid action')

        # calculate rewards
        if self.check_valency():
            return self.reward_function  # arbitrary choice
        else:
            return -self.reward_function  # arbitrary choice

    def render(self):
        print(Chem.MolToSmiles(self.mol))

    def _add_atom(self, action):
        """
    Adds an atom
    :param action: one hot np array of dim d_n, where d_n is the number of
    atom types
    :return:
    """
        # assert action.shape == (len(self.possible_atom_types),)
        atom_type_idx = action[0]
        atom_symbol = self.possible_atom_types[atom_type_idx]
        self.current_atom_idx = self.mol.AddAtom(Chem.Atom(atom_symbol))
        self.total_atoms += 1

    def _modify_bond(self, action):
        """
    Adds or modifies a bond (currently no deletion is allowed)
    :param action: np array of dim N-1 x d_e, where N is the current total
    number of atoms, d_e is the number of bond types
    :return:
    """
        # print("Action space [modify bond]: ", action.shape)
        # assert action.shape == (self.current_atom_idx, len(self.possible_bond_types))
        # other_atom_idx = int(np.argmax(action.sum(axis=1)))  # b/c
        # print("Other atom index: ", other_atom_idx, "| action: ", np.argmax(action.sum(axis=1)))
        # GetBondBetweenAtoms fails for np.int64
        # bond_type_idx = np.argmax(action.sum(axis=0))
        bond_type = self.possible_bond_types[action[3]]

        # if bond exists between current atom and other atom, modify the bond
        # type to new bond type. Otherwise create bond between current atom and
        # other atom with the new bond type
        bond = self.mol.GetBondBetweenAtoms(int(action[1]), int(action[2]))
        if bond:
            # bond.SetBondType(bond_type)
            pass
        else:
            # self.mol.AddBond(self.current_atom_idx, other_atom_idx, order=bond_type)
            self.mol.AddBond(int(action[1]), int(action[2]), order=bond_type)
            self.total_bonds += 1

    def get_num_atoms(self):
        return self.total_atoms

    def get_num_bonds(self):
        return self.total_bonds

    # TODO(Bowen): check
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

    # TODO(Bowen): check
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

    def get_matrices(self):
        """
    Get the adjacency matrix, edge feature matrix and, node feature matrix
    of the current molecule graph
    :return: np arrays: adjacency matrix, dim n x n; edge feature matrix,
    dim n x n x d_e; node feature matrix, dim n x d_n
    """
        A_no_diag = Chem.GetAdjacencyMatrix(self.mol)
        A = A_no_diag + np.eye(*A_no_diag.shape)

        n = A.shape[0]

        d_n = len(self.possible_atom_types)
        F = np.zeros((n, d_n))
        for a in self.mol.GetAtoms():
            atom_idx = a.GetIdx()
            atom_symbol = a.GetSymbol()
            float_array = (atom_symbol == self.possible_atom_types).astype(float)
            assert float_array.sum() != 0
            F[atom_idx, :] = float_array

        d_e = len(self.possible_bond_types)
        E = np.zeros((n, n, d_e))
        for b in self.mol.GetBonds():
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)
            assert float_array.sum() != 0
            E[begin_idx, end_idx, :] = float_array
            E[end_idx, begin_idx, :] = float_array

        return A, E, F


# for testing get_matrices
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


# TESTS

# molecule construction test


def reward_func():
    return 1


env = MoleculeEnvironment(reward_func())
# add carbon
env.reset()
env.step(np.array([2, 0, 1, 0]), 'add_atom')
env.step(np.array([2, 0, 1, 0]), 'modify_bond')
print(Chem.MolToSmiles(env.mol))

env.step(np.array([2, 0, 2, 0]), 'add_atom')
env.step(np.array([2, 0, 2, 0]), 'modify_bond')
print(Chem.MolToSmiles(env.mol))

env.step(np.array([2, 0, 3, 0]), 'add_atom')
env.step(np.array([2, 0, 3, 0]), 'modify_bond')
print(Chem.MolToSmiles(env.mol))

env.step(np.array([2, 0, 4, 0]), 'add_atom')
env.step(np.array([2, 0, 4, 0]), 'modify_bond')
print(Chem.MolToSmiles(env.mol))


# # add carbon
# env.step(np.array([1, 0, 0]), 'add_atom')
# # add double bond between carbon 1 and carbon 2
# env.step(np.array([[0, 1, 0]]), 'modify_bond')
# # add carbon
# env.step(np.array([1, 0, 0]), 'add_atom')
# # add single bond between carbon 2 and carbon 3
# env.step(np.array([[0, 0, 0], [1, 0, 0]]), 'modify_bond')
# # add oxygen
# env.step(np.array([0, 0, 1]), 'add_atom')
# # add single bond between carbon 3 and oxygen
# env.step(np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]), 'modify_bond')
#
# env.render()
# assert Chem.MolToSmiles(env.mol, isomericSmiles=True) == 'C=CCO'
#
possible_atoms = ['C', 'N', 'O']
possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE]
# # test get_matrices 1
print("Testing get_matrices")
A, E, F = env.get_matrices()
# print("A:")
# print(A)
# print("E:")
# print(E)
# print("F:")
# print(F)
#
print(Chem.MolToSmiles(matrices_to_mol(A, E, F, possible_atoms,
                                       possible_bonds), isomericSmiles=True))
# assert Chem.MolToSmiles(matrices_to_mol(A, E, F, possible_atoms,
#                                         possible_bonds), isomericSmiles=True) \
#        == 'C=CCO'

# print("Beginning valency test")
# # molecule check valency test
# env = MoleculeEnvironment(reward_func())
# # add carbon
# r = env.step(np.array([1, 0, 0]), 'add_atom')
# print(Chem.MolToSmiles(env.mol))
# # add oxygen
# r = env.step(np.array([0, 0, 1]), 'add_atom')
# print(Chem.MolToSmiles(env.mol))
# # add single bond between carbon and oxygen 1
# r = env.step(np.array([[1, 0, 0]]), 'modify_bond')
# print(Chem.MolToSmiles(env.mol))
#
# print(r)
#
# # add oxygen
# r = env.step(np.array([0, 0, 1]), 'add_atom')
# print(Chem.MolToSmiles(env.mol))
# # add single bond between carbon and oxygen 2
# r = env.step(np.array([[1, 0, 0], [0, 0, 0]]), 'modify_bond')
# print(Chem.MolToSmiles(env.mol))
#
# # add oxygen
# r = env.step(np.array([0, 0, 1]), 'add_atom')
# print(Chem.MolToSmiles(env.mol))
# # add single bond between carbon and oxygen 3
# r = env.step(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), 'modify_bond')
# print(Chem.MolToSmiles(env.mol))
#
# # add oxygen
# r = env.step(np.array([0, 0, 1]), 'add_atom')
# print(Chem.MolToSmiles(env.mol))
# # add single bond between carbon and oxygen 4
# r = env.step(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
#              'modify_bond')
# print(Chem.MolToSmiles(env.mol))
# # add oxygen
# r = env.step(np.array([0, 0, 1]), 'add_atom')
# print(Chem.MolToSmiles(env.mol))
# assert r == 1
# # add single bond between carbon and oxygen 4. This exceeds valency on C
# r = env.step(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
#              'modify_bond')
# print(Chem.MolToSmiles(env.mol))
# print(r)
# assert r == -1
#
# # test get_matrices 2
# A, E, F = env.get_matrices()
# assert Chem.MolToSmiles(matrices_to_mol(A, E, F, possible_atoms,
#                                         possible_bonds), isomericSmiles=True) \
#        == 'OC(O)(O)(O)O'
