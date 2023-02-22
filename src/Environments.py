import gym
import copy
from abc import ABC, abstractmethod


class StatelessGym:
    @staticmethod
    def make(env_name, **kwargs):
        if env_name == 'CartPole-v1':
            return CustomCartPole(**kwargs)
        else:
            return CustomBaseEnv(gym.make(env_name, **kwargs))


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


class CustomBaseEnv(CustomAbstractEnv):

    def get_state(self):
        return copy.deepcopy(self.env)

    def set_state(self, state):
        self.env = state


class CustomCartPole(CustomAbstractEnv):
    def __init__(self, **kwargs):
        super().__init__(gym.make('CartPole-v1', **kwargs))

    def get_state(self):
        return copy.deepcopy(self.env.state)

    def set_state(self, state):
        self.env.env.env.state = state


if __name__ == '__main__':
    #env = StatelessGym.make('CartPole-v1')
    env = StatelessGym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    env.reset()
    next_state, _, _, _ = env.step(2)
    print(next_state)
    next_state, _, _, _ = env.step(2)
    print(next_state)
    first_state = env.get_state()
    print("first_state: ", first_state)

    for i in range(1):
        env.reset()
        print("env.state: ")
        env.render()
        env.set_state(first_state)
        print("env.state after set: ")
        env.render()
        done = False
        while not done:
            next_state, reward, done, _ = env.step(2)
            print(next_state)
        print("--------------------------------------------------------")
