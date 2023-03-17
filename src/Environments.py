import gym
import copy
from abc import ABC, abstractmethod

class StatelessGym:
    @staticmethod
    def make(env_name, **kwargs):
        if env_name == 'CartPole-v1':
            return CustomCartPole(**kwargs)
        else:
            return CustomBaseEnv(env_name, **kwargs)

class CustomAbstractEnv(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def get_action_space(self):
        return self.env.action_space

class CustomBaseEnv(CustomAbstractEnv):
    def __init__(self, env_name, **kwargs):
        super().__init__(gym.make(env_name, **kwargs))

    def get_state(self):
        return copy.deepcopy(self.env)

    def set_state(self, state):
        self.env = copy.deepcopy(state)


class CustomCartPole(CustomAbstractEnv):
    def __init__(self, **kwargs):
        super().__init__(gym.make('CartPole-v1', **kwargs))

    def get_state(self):
        return copy.deepcopy(self.env.state)

    def set_state(self, state):
        self.env.reset()
        self.env.env.env.state = state