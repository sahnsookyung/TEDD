import gym
from gym import spaces, utils
from gym.utils import seeding

from rdkit import Chem


# TODO
# 1. create action_space and observation_space
# 2. define all env methods
# 3. create the interface between env and agent
# 4. placeholder methods for choosing an action and for calculating a reward

class MoleculeEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.MultiDiscrete(0,0)
        self.observation_space = gym.spaces.MultiDiscrete(0,0)

        self.seed()
        self.reset()

    def step(self, action):
        print("step")

    def reset(self):
        print("reset")

    def render(self, mode='human', close=False):
        print("render")

    def close(self):
        print("Closing application")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
