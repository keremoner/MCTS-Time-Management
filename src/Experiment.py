from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

def file_dir(relative_path):
    absolute_path = os.path.dirname(__file__)
    return os.path.join(absolute_path, relative_path)

class TrialResult:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def get_temperature(self):
        return self.kwargs['temperature']
    
    def get_simulation(self):
        return self.kwargs['simulation']
    
    def get_rewards(self):
        return self.kwargs['rewards']
    
    def get_mean_reward(self):
        return np.mean(self.kwargs['rewards'])
    
    def get_trial(self):
        return self.kwargs['trial']
    
    def get_error(self):
        return np.std(self.get_rewards()) / np.sqrt(self.get_trial())
    
    def __str__(self):
        return  "Simulation = " + str(self.get_simulation()) + "\tMean Reward = " + str(self.get_mean_reward()) + "\tError = " + str(self.get_error())

class Experiment:
    def __init__(self, env, agent, **kwargs):
        self.env = env
        self.agent = agent
        self.kwargs = kwargs
        self.results = []

    #TODO: Change it to allow a non-printing option
    def run(self, save=False):
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
        plt.legend()
        plt.xlabel("Number of Simulations")
        plt.ylabel("Mean Cumulative Reward")
        if save:
            plt.savefig(file_dir("./../results/" + self.kwargs['experiment_name'] + ".png"))
            self.save_results()
            
    def save_results(self):
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')
        string = "Experiment Name: " + self.kwargs['experiment_name'] + " (" + git_hash + ")" + "\n"
        string += "Environment: " + str(self.env) + "\n" 
        string += "Agent: " + str(self.agent) + "\n"
        string += "Temperatures: " + str(self.kwargs['temperatures']) + "\n"
        string += "Simulations: " + str(self.kwargs['simulations']) + "\n"
        string += "Trials: " + str(self.kwargs['trial']) + "\n"
        string += "----------------------------------------\n"
        for temperature in self.kwargs['temperatures']:
            string += "Temperature: " + str(temperature) + "\n"
            temp_result = [result for result in self.results if result.get_temperature() == temperature]
            for result in temp_result:
                string += "\t" + str(result) + "\n"
        with open(file_dir("./../results/" + self.kwargs['experiment_name'] + ".txt"), "w") as f:
            f.write(string)

    def run_trial(self, temperature, simulation):
        rewards = []
        self.agent.set_temperature(temperature)
        self.agent.set_simulations(simulation)
        for i in range(self.kwargs['trial']):
            cumulative_reward = 0
            self.env.reset()
            action = self.agent.select_action(self.env)
            next_state, reward, done, _ = self.env.step(action)
            cumulative_reward += reward
            while not done:
                action = self.agent.select_action(self.env)
                next_state, reward, done, _ = self.env.step(action)
                cumulative_reward += reward
            rewards.append(cumulative_reward)
        return TrialResult(temperature=temperature, simulation=simulation, rewards=rewards, trial=self.kwargs['trial'])
    
    def show_results(self):
        plt.show()