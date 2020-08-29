import gym
from gym import spaces, utils
from gym.utils import seeding
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
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.mol = Chem.RWMol()

        possible_atoms = ['C', 'N', 'O', 'S', 'Cl']
        self.possible_atoms = possible_atoms
        possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE]

        self.atom_type_num = len(possible_atoms)
        self.possible_atom_types = np.array(possible_atoms)
        self.possible_bond_types = np.array(possible_bonds, dtype=object)
        self.d_n = len(self.possible_atom_types)

        # number of actions
        max_action = 2
        min_action = 1
        self.max_action = max_action
        self.min_action = min_action

        # maximum number of atoms this molecules will consist of
        self.max_atom = 13 + len(possible_atoms)

        # action and observation space definition
        self.action_space = gym.spaces.MultiDiscrete([self.max_atom, self.max_atom, 3, 2])
        self.observation_space = {
            'adj': gym.Space(shape=[len(possible_bonds), self.max_atom, self.max_atom]),
            'node': gym.Space(shape=[1, self.max_atom, self.d_n])
        }

        self.counter = 0

    def init(self, x=39, n=4, possible_atoms=None):
        pass
        # If we need to pass in any arguments then we need to use this function
        # This is because __init__(self) doesn't accept other parameters

    def step(self, action):

        return ob, reward, new, info

    def reset(self, smile=None):

        return ob

    def render(self, mode='human', close=False):
        return

    def close(self):
        print("Closing application")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
