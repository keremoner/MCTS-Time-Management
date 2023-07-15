import gym
import copy
from abc import ABC, abstractmethod
from gym.envs.toy_text.frozen_lake import generate_random_map

class StatelessGym:
    @staticmethod
    def make(env_name, **kwargs):
        if env_name == 'CartPole-v1':
            return CustomCartPole(**kwargs)
        elif env_name == 'Acrobot-v1':
            return CustomAcrobot(**kwargs)
        elif env_name == 'MountainCar-v0':
            return CustomMountainCar(**kwargs)
        elif env_name == 'FrozenLake-v1':
            return CustomFrozenLake(**kwargs)
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
    
    @abstractmethod
    def randomize_parameters(self):
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
    
    def __str__(self):
        return self.env.spec.id
    
class CustomBaseEnv(CustomAbstractEnv):
    def __init__(self, env_name, **kwargs):
        super().__init__(gym.make(env_name, **kwargs))

    def get_state(self):
        return copy.deepcopy(self.env)

    def set_state(self, state):
        self.env = copy.deepcopy(state)
    
    def randomize_parameters(self):
        pass

class CustomFrozenLake(CustomBaseEnv):
    def __init__(self, **kwargs):
        super().__init__('FrozenLake-v1', **kwargs)
    
    def set_map(self, map):
        self.env = gym.make('FrozenLake-v1', desc=map, is_slippery=False)
    
    def randomize_parameters(self, map_size=4, freeze_prob=0.1, show_map=False):
        random_map = generate_random_map(size=map_size, p=freeze_prob)
        self.set_map(random_map)
        if show_map:
            self.reset()
            print(self.render(mode='ansi'))

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
    
    def randomize_parameters(self):
        pass
        
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
        
    def randomize_parameters(self):
        pass

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
        
    def randomize_parameters(self):
        pass