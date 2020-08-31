import gym
from gym import spaces, utils
from gym.utils import seeding

from rdkit import Chem

import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

import copy


# TODO
# 1. create action_space and observation_space
# 2. define all env methods
# 3. create the interface between env and agent
# 4. placeholder methods for choosing an action and for calculating a reward

class MoleculeEnvironment(gym.Env):
    def __init__(self, reward_function, n_iterations, max_iterations, max_molecule_size=7):
        """
        :param reward_function: A function that returns a reward value
        :param possible_atom_types: The elements of the periodic table used in this class
        :param n_iterations: The number of iterations before an interim reward is retured
        :param max_iterations: User specified, the environment stop after this many iterations
        """

        self.possible_atoms = ['C', 'N', 'O', 'S', 'Cl']
        self.possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                               Chem.rdchem.BondType.TRIPLE]
        self.max_molecule_size = max_molecule_size

        self.n_iterations = n_iterations
        self.max_iterations = max_iterations
        self.interim_reward = 0
        self.cumulative_reward = 0

        self.mol = Chem.RWMol()

        # dim d_n. Array that contains the possible atom symbols strs
        self.possible_atom_types = np.array(self.possible_atoms)
        # dim d_e. Array that contains the possible rdkit.Chem.rdchem.BondType objects
        self.possible_bond_types = np.array(self.possible_bonds, dtype=object)
        self.current_atom_idx = None
        self.total_atoms = 0
        self.total_bonds = 0
        self.reward_function = reward_function

        # counter for number of steps
        self.counter = 0

        self.action_space = gym.spaces.MultiDiscrete(
            [len(self.possible_atom_types), self.max_molecule_size,
             self.max_molecule_size, len(self.possible_bonds)]
        )

        # param adj: adjacency matrix, numpy array, dim k x k.
        # param edge: edge attribute matrix, numpy array, dim k x k x de.
        # param node: node attribute matrix, numpy array, dim k x dn.
        # k: maximum atoms in molecule
        # de: possible bond types
        # dn: possible atom types
        self.observation_space = {
            'adj': gym.Space(shape=[1, self.max_molecule_size, self.max_molecule_size]),
            'edge': gym.Space(shape=[len(self.possible_bonds), self.max_molecule_size, self.max_molecule_size]),
            'node': gym.Space(shape=[1, self.max_molecule_size, len(self.possible_atom_types)])
        }

    def reset(self):
        self.mol = Chem.RWMol()
        self.current_atom_idx = None

        self.total_atoms = 0
        self.total_bonds = 0

        self.interim_reward = 0
        self.cumulative_reward = 0
        self.counter = 0

        # Total atoms = 1 at this point
        self._add_atom([0, 0, 0, 0])

    def step(self, action):
        """
        Perform a given action
        :param action:
        :param action_type:
        :return: reward of 1 if resulting molecule graph does not exceed valency,
        -1 if otherwise
        """

        # Note: The user-specified action must be valid,
        # if you want to join atoms at location 2 and 3 with a bond, these atoms must exist through prior actions
        info = {}

        terminate_condition = (self.mol.GetNumAtoms() >= self.max_molecule_size or
                               self.counter >= self.max_iterations)
        if terminate_condition:
            done = True
        else:
            done = False

        self.counter += 1

        self._add_atom(action)
        self._modify_bond(action)

        # calculate rewards
        if self.check_valency():
            reward = self.reward_function  # arbitrary choice
        else:
            reward = -self.reward_function  # arbitrary choice

        # self.interim_reward += reward
        self.cumulative_reward += reward
        self.interim_reward += reward

        observation = self.get_matrices()

        # append to info
        info['smiles'] = Chem.MolToSmiles(self.mol)
        info['reward'] = reward

        if self.counter % self.n_iterations == 0:
            info['interim_reward'] = self._get_interim_reward()
        info['cumulative_reward'] = self.cumulative_reward
        info['num_steps'] = self.counter

        return observation, reward, done, info

    def render(self):
        print(Chem.MolToSmiles(self.mol))

    def seed(self, seed=None):
        np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

    def _get_interim_reward(self):
        """
        :return: returns the interim_reward and resets it to zero
        """
        reward = self.interim_reward
        self.interim_reward = 0
        return reward

    def _add_atom(self, action):
        """
        Adds an atom
        :param action: one hot np array of dim d_n, where d_n is the number of
        atom types
        :return:
        """
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
        bond_type = self.possible_bond_types[action[3]]
        bond = self.mol.GetBondBetweenAtoms(int(action[1]), int(action[2]))
        if bond:
            pass
        else:
            self.mol.AddBond(int(action[1]), int(action[2]), order=bond_type)
            self.total_bonds += 1

    def get_num_atoms(self):
        return self.total_atoms

    def get_num_bonds(self):
        return self.total_bonds

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
