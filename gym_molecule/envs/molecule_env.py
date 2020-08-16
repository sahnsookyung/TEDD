import gym
from gym import error, spaces, utils
from gym.utils import seeding


class MoleculeEnvironment(gym.Env):
    def __init__(self):
        super().__init__()

    def step(self, action):
        print("step")

    def reset(self):
        print("reset")

    def render(self, mode='human', close=False):
        print("render")

    def seed(self):
        print("seed not set")

    def close(self):
        print("Closing application")
