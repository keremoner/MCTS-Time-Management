from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class TrialResult:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        plt.legend()
        plt.xlabel("Number of Simulations")
        plt.ylabel("Mean Cumulative Reward")
    
    def get_temperature(self):
        return self.kwargs['temperature']
    
    def get_simulation(self):
        return self.kwargs['simulation']
    
    def get_rewards(self):
        return self.kwargs['rewards']
    
    def get_mean_reward(self):
        return np.mean(self.kwargs['rewards'])
    
    def get_total(self):
        return self.kwargs['total']
    
    def get_error(self):
        return np.std(self.kwargs['rewards']) / np.sqrt(self.kwargs['trial'])

class Experiment:
    def __init__(self, env, agent, **kwargs):
        self.env = env
        self.agent = agent
        self.kwargs = kwargs
        self.results = []

    def run(self):
        for temperature in self.kwargs['temperatures']:
            errors = []
            mean_rewards = []
            for simulation in self.kwargs['simulations']:
                result = self.run_trial(temperature, simulation)
                self.results.append(result)
                errors.append(result.get_error())
                mean_rewards.append(result.get_mean_reward())
                print("Simulation = ", simulation, "\tMean Cumulative Reward = ", mean_rewards[len(mean_rewards) - 1], "\tError = ", errors[len(errors) - 1])
            print("Temperature = ", temperature, " Results = ", mean_rewards)
            plt.errorbar(self.kwargs['simulations'], mean_rewards, yerr=errors, label=str(temperature))

    def run_trial(self, temperature, simulation):
        rewards = []
        self.agent.set_temp(temperature)
        self.agent.set_simulations(simulation)
        for i in range(self.kwargs['trial']):
            cumulative_reward = 0
            root = self.env.reset()
            action = self.agent.select_action(root)
            next_state, reward, done, _ = self.env.step(action)
            cumulative_reward += reward
            while not done:
                action = self.agent.select_action(next_state)
                next_state, reward, done, _ = self.env.step(action)
                cumulative_reward += new_reward
            rewards.append(cumulative_reward)
        return TrialResult(temperature=temp, simulation=simulation, rewards=rewards, total=self.kwargs['trial'])

    def show_results(self):
        plt.show()
    
    def save_results(self):
        plt.savefig(self.kwargs['name'])