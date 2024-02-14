import gym
import copy
from abc import ABC, abstractmethod
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

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

class CustomFrozenLake(CustomAbstractEnv):
    def __init__(self, **kwargs):
        super().__init__(gym.make('FrozenLake-v1', **kwargs))
        self.map = kwargs.get('desc', None)
    
    def set_state(self, state):
        self.env.reset()
        s, lastaction = state
        self.env.env.env.s = s
        self.env.env.env.lastaction = lastaction
    
    def get_state(self):
        return (copy.deepcopy(self.env.env.env.s), copy.deepcopy(self.env.env.env.lastaction))
    
    def get_map(self):
        return self.map
    
    def set_map(self, map):
        self.env = gym.make('FrozenLake-v1', desc=map, is_slippery=False)
        self.map = map
    
    def randomize_parameters(self, map_size=4, freeze_prob=0.1, show_map=False):
        random_map = generate_random_map(size=map_size, p=freeze_prob)
        self.set_map(random_map)
        if show_map:
            self.reset()
            print(self.render(mode='ansi'))
        return random_map
    
    def reset(self, initial_state_random = False):
        s = self.env.reset()
        if initial_state_random:
            map_size = self.env.env.env.ncol * self.env.env.env.nrow
            map_byte = self.env.env.env.desc
            spawnable = []
            for i in range(len(map_byte)):
                for j in range(len(map_byte)):
                    if map_byte[i][j] != b'H' and map_byte[i][j] != b'G':
                        spawnable.append(i * len(map_byte) + j)
                    
            s = np.random.choice(spawnable)
            self.set_state((s, None))
        return s
             

class CustomCartPole(CustomAbstractEnv):
    def __init__(self, initial_state_default=True, low=-0.05, high=0.05, step_size=0.01, **kwargs):
        super().__init__(gym.make('CartPole-v1', **kwargs))
        self.initial_state_default = initial_state_default
        self.low = low
        self.high = high
        self.step_size = step_size
        self.arange = np.append(np.arange(self.low, 0, self.step_size), np.arange(0, self.high + self.step_size, self.step_size))

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
    
    def reset(self):
        s = self.env.reset()
        if not self.initial_state_default:
            state = self.get_state()
            actual_state, elapsed_steps, steps_beyond_done, has_reset = state
            if self.step_size > 0:
                s = np.random.choice(self.arange, size=4)
            else:
                s = np.random.uniform(low=self.low, high=self.high, size=4)
            self.set_state((s, elapsed_steps, steps_beyond_done, has_reset))
        return s
        
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