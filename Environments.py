import gym
import copy

class CustomCartPole():
    def __init__(self):
        self.env = gym.make('CartPole-v1')
    
    def set_state(self, state):
        self.env.env.env.state = state
    
    def get_state(self):
        return copy.deepcopy(self.env.state)
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def close(self):
        return self.env.close()
    
if __name__ == '__main__':
    env = CustomCartPole()
    env.reset()
    first_state = env.get_state()
    print("first_state: ", first_state)

    for i in range(10):
        env.reset()
        print("env.state: ", env.env.state)
        env.set_state(first_state)
        print("env.state after set: ", env.env.state)
        done = False
        while not done:   
            next_state, reward, done, _ = env.step(0)
            print(next_state)
        print("--------------------------------------------------------")