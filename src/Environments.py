import gym
import copy
from abc import ABC, abstractmethod

class StatelessGym:
    @staticmethod
    def make(env_name, **kwargs):
        if env_name == 'CartPole-v1':
            return CustomCartPole(**kwargs)
        elif env_name == 'Acrobot-v1':
            return CustomAcrobot(**kwargs)
        elif env_name == 'MountainCar-v0':
            return CustomMountainCar(**kwargs)
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
        return (copy.deepcopy(self.env.env.env.state), self.env._elapsed_steps, self.env.env.env.steps_beyond_done, self.env.env._has_reset)

    def set_state(self, state):
        #self.env.reset()
        actual_state, elapsed_steps, steps_beyond_done, has_reset = state
        self.env.env.env.state = actual_state
        self.env._elapsed_steps = elapsed_steps
        self.env.env.env.steps_beyond_done = steps_beyond_done
        self.env.env._has_reset = has_reset
        
class CustomAcrobot(CustomAbstractEnv):
    def __init__(self, **kwargs):
        super().__init__(gym.make('Acrobot-v1', **kwargs))

    def get_state(self):
        return (copy.deepcopy(self.env.env.env.state), self.env._elapsed_steps, self.env.env._has_reset)

    def set_state(self, state):
        #self.env.reset()
        actual_state, elapsed_steps, has_reset = state
        self.env.env.env.state = actual_state
        self.env._elapsed_steps = elapsed_steps
        self.env.env._has_reset = has_reset

class CustomMountainCar(CustomAbstractEnv):
    def __init__(self, **kwargs):
        super().__init__(gym.make('MountainCar-v0', **kwargs))

    def get_state(self):
        return (copy.deepcopy(self.env.env.env.state), self.env._elapsed_steps, self.env.env._has_reset)

    def set_state(self, state):
        #self.env.reset()
        actual_state, elapsed_steps, has_reset = state
        self.env.env.env.state = actual_state
        self.env._elapsed_steps = elapsed_steps
        self.env.env._has_reset = has_reset