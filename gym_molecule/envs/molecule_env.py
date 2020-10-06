import errno

import gym
from gym.utils import seeding
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.PyMol import MolViewer
from rdkit.Chem import rdmolops
from rdkit.Chem.rdmolops import FastFindRings
import os

import pymol
import copy


# TODO make sure the reward function only deals with the reward
#  revert to old molecule if necessary using copy function
#  compare the current product to the reference product

class MoleculeEnvironment(gym.Env):
    def __init__(self, reward_function, n_iterations, max_iterations, max_molecule_size=38, possible_atoms=None):
        """
        :param reward_function: A function that returns a reward value
        :param n_iterations: The number of iterations before an interim reward is retured
        :param max_iterations: User specified, the environment stop after this many iterations
        :param max_molecule_size: The maximum permitted number of atoms in this molecule
        :param possible_atoms: List of elements of the periodic table to be used in the environment
        """

        if possible_atoms is None:
            possible_atoms = ['C', 'N', 'O', 'S', 'Cl']
        self.possible_atoms = possible_atoms
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

        self.pymol_window_flag = False

        try:
            os.makedirs("./pymol_renderings")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def reset(self, molecule=None):
        """
        Resets the environment to its initial state, as defined by optional parameters
        :param molecule: SMILES string specifying the molecule to start with when the reset function is called
        :return:
        """
        self.mol = Chem.RWMol()
        if molecule is None:
            self._add_atom([0, 0, 0, 0])
        else:
            self.mol = Chem.RWMol(Chem.MolFromSmiles(molecule))

        self.current_atom_idx = None

        self.interim_reward = 0
        self.cumulative_reward = 0
        self.counter = 0

    # TODO: Think about how to constrain the action to the valid actions (e.g. no atoms trying to bond to non-existing atoms)
    def step(self, action):
        """
        Perform a given action, returns a four tuple
        :param action: a list of 4 elements [atom_type, atom_index_1, atom_index_2, bond_type]
        Returns
        -------
        observation : [dict]
            [Dictionary containing the np arrays of: adjacency matrix, dim n x n; edge feature matrix,
                dim n x n x d_e; node feature matrix, dim n x d_n]
        reward: [float]
            [The reward resulting from completing the action]
        done: [boolean]
            [flag indicating whether the training process has been completed as defined by the user]
        info: [dict]
            [Contains the interim and cumulative rewards, as well as a bunch of other information useful for debugging]
        """

        # TODO Need to implement termination on conditions being fulfilled

        # Note: The user-specified action must be valid,
        # if you want to join atoms at location 2 and 3 with a bond, these atoms must exist through prior actions
        info = {}
        # mol_old = copy.deepcopy(self.mol)  # keep old mol

        # Checking that the action contains valid values
        assert action[0] < len(self.possible_atom_types)  # Atom choice
        assert action[3] < len(self.possible_bond_types)  # Bond choice
        assert action[1] <= self.get_num_atoms() and action[2] <= self.get_num_atoms()  # Connecting existing atoms

        terminate_condition = (self.mol.GetNumAtoms() >= self.max_molecule_size or
                               self.counter >= self.max_iterations)
        if terminate_condition:
            done = True
        else:
            done = False

        self.counter += 1

        self._add_atom(action)
        self._modify_bond(action)

        # Todo: Check if the reward is negative, and revert back to old molecule if necessary?
        # The reward function is assumed to check the molecule's valency and validity, and return negative values where relevant
        reward = self.reward_function(self.action_space, self.observation_space)  # arbitrary choice

        self.cumulative_reward += reward
        # print(self.cumulative_reward)

        self.interim_reward += reward

        observation = self.get_matrices()

        # append to info
        info['smiles'] = Chem.MolToSmiles(self.mol)
        info['reward'] = reward

        # Reset the interim reward every n iterations
        if self.counter % self.n_iterations == 0:
            info['interim_reward'] = self._get_interim_reward()
        info['cumulative_reward'] = self.cumulative_reward
        info['num_steps'] = self.counter

        return observation, reward, done, info

    def render(self, mode="human"):
        """
        :param mode: Just a way of indicating whether the rendering is mainly for humans vs machines in OpenAI
        :return: void. This is just a visualization of the current molecule in our environment
        """
        # TODO add some exception handling, since the PYMOL server needs to be online for this to work
        if not self.pymol_window_flag:
            self.start_pymol()

        molecule = self.mol

        self.mol.UpdatePropertyCache(strict=False)  # Update valence information
        FastFindRings(
            self.mol)  # Quick for finding out if an atom is in a ring, use Chem.GetSymmSSR() if more reliability is desired
        Chem.AddHs(molecule)  # Add explicit hydrogen atoms for rendering purposes

        # remove stereochemistry information
        rdmolops.RemoveStereochemistry(molecule)

        # Generate 3D structure
        AllChem.EmbedMolecule(molecule)
        AllChem.MMFFOptimizeMolecule(molecule)

        v = MolViewer()
        v.ShowMol(molecule)
        v.GetPNG(h=400)

        # Save the rendering in pse format
        pymol.cmd.save("./pymol_renderings/" + Chem.MolToSmiles(molecule) + ".pse", format="pse")
        Chem.RemoveHs(molecule)  # Remove explicit hydrogen

    def start_pymol(self):
        """
        Starts the pymol process for rendering
        """
        pymol.pymol_argv = ['pymol', '-R']
        pymol.finish_launching()
        self.pymol_window_flag = True

    def end_pymol(self):
        """
        Ends the pymol process
        """
        pymol.cmd.quit()
        self.pymol_window_flag = False

    def seed(self, seed=None):
        """
        Sets the seed for the environment for replicable results where stochastic processes are involved.

        Parameters
        ----------
        seed : [long]
            [User specified long]
        Returns
        -------
        [seed] : [list]
            [List containing a single long value, following the convention of OpenAI Gym]
        """
        np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """
        Performs some cleanup of the environment
        """
        self.end_pymol()

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
        :param action: one hot np array of dim d_n, where d_n is the number of atom types
        :return:
        """
        atom_type_idx = action[0]
        atom_symbol = self.possible_atom_types[atom_type_idx]
        self.current_atom_idx = self.mol.AddAtom(Chem.Atom(atom_symbol))
        self.total_atoms = self.get_num_atoms()

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
            self.total_bonds = self.get_num_bonds()

    def get_num_atoms(self):
        return self.mol.GetNumAtoms()

    def get_num_bonds(self):
        return len(self.mol.GetBonds())

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
