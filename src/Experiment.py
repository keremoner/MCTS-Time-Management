from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import yaml
import random
import csv

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
        data = {}
        data["results"] = []
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
            data_temp_result = [result.get_rewards() for result in self.results if result.get_temperature() == temperature]
            data["results"].append(data_temp_result)
        
        data["experiment_name"] = self.kwargs['experiment_name']
        data["environment"] = str(self.env)
        data["agent"] = str(self.agent)
        data["temperatures"] = self.kwargs['temperatures']
        data["simulations"] = self.kwargs['simulations']
        data["trial"] = self.kwargs['trial']
        
        with open(file_dir("./../results/" + self.kwargs['experiment_name'] + ".txt"), "w") as f:
            f.write(string)
        
        with open(file_dir("./../results/" + self.kwargs['experiment_name'] + '.yaml'), 'w') as file:
            yaml.dump(data, file)

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

class RandomExperiment():
    def __init__(self, env, agent, **kwargs):
        self.env = env
        self.agent = agent
        self.kwargs = kwargs
        self.results = []
        
    def run(self, initial_state_random = False):
        simulation = random.randint(self.kwargs['simulations'][0], self.kwargs['simulations'][1])
        temperature = self.kwargs['temperature']
        self.agent.set_temperature(temperature)
        self.agent.set_simulations(simulation)
        e_return = 0
        dis_e_return = 0
        discount_factor = self.agent.discount_factor
        initial_state = None
        if initial_state_random == True:
            initial_state = self.env.reset(True)
        else: 
            initial_state = self.env.reset()
                
        action = self.agent.select_action(self.env)
        next_state, reward, done, _ = self.env.step(action)
        e_return += reward
        dis_e_return += reward
        while not done:
            action = self.agent.select_action(self.env)
            next_state, reward, done, _ = self.env.step(action)
            e_return += reward         
            dis_e_return += reward * discount_factor
            discount_factor *= self.agent.discount_factor
            
        if self.env.env.spec.id == "CartPole-v1":
            return [temperature] + initial_state.tolist() + [simulation, e_return, dis_e_return]
        else:
            return [temperature, simulation, initial_state, e_return, dis_e_return]

        
    def create_dataset(self, n, file_name, initial_state_random=False):
        dataset = []
        if self.env.env.spec.id == "CartPole-v1":
            dataset.append(["Temperature", "Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity", "Simulations", "Return", "Discounted Return"])
        else:
            dataset.append(["Temperature", "Simulations", "Initial State", "Return", "Discounted Return"])
        
        with open(file_dir("./../datasets/" + file_name + ".csv"), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(1, n + 1):
                dataset.append(self.run(initial_state_random))
                if n < 10 or i % (n // 10) == 0:
                    print("%s: %d/%d" % (file_name, i, n))
                    writer.writerows(dataset)
                    dataset = []
        return dataset
    
class ParametrizedRandomExperiment():
    def __init__(self, env, agent, **kwargs):
        self.env = env
        self.agent = agent
        self.kwargs = kwargs
        self.results = []
        
    def run(self, initial_state_random=False):
        #Randomize the number of simulations
        simulation = random.randint(self.kwargs['simulations'][0], self.kwargs['simulations'][1])
        
        #Randomize environment parameters
        if self.env.env.spec.id == "FrozenLake-v1":
            map_size = random.randint(self.kwargs['map_size'][0], self.kwargs['map_size'][1])
            freeze_prob = random.uniform(self.kwargs['freeze_prob'][0], self.kwargs['freeze_prob'][1])
            parameters = self.env.randomize_parameters(map_size=map_size, freeze_prob=freeze_prob)
        else:
            parameters = self.env.randomize_parameters()
        
        #Set experiment parameters
        temperature = self.kwargs['temperature']
        self.agent.set_temperature(temperature)
        self.agent.set_simulations(simulation)
        e_return = 0
        dis_e_return = 0
        discount_factor = self.agent.discount_factor
        initial_state = None
        if initial_state_random == True:
            initial_state = self.env.reset(True)
        else: 
            initial_state = self.env.reset()
        action = self.agent.select_action(self.env)
        next_state, reward, done, _ = self.env.step(action)
        e_return += reward
        dis_e_return += reward
        while not done:
            action = self.agent.select_action(self.env)
            next_state, reward, done, _ = self.env.step(action)
            e_return += reward         
            dis_e_return += reward * discount_factor
            discount_factor *= self.agent.discount_factor
            
        if self.env.env.spec.id == "CartPole-v1":
            return [temperature] + initial_state.tolist() + [simulation, e_return, dis_e_return]
        elif self.env.env.spec.id == "FrozenLake-v1":
            return [temperature, parameters, simulation, initial_state, e_return, dis_e_return]
        else:
            return [temperature, simulation, e_return, dis_e_return]

        
    def create_dataset(self, n, file_name, initial_state_random=False):
        dataset = []
        if self.env.env.spec.id == "CartPole-v1":
            dataset.append(["Temperature", "Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity", "Simulations", "Return", "Discounted Return"])
        elif self.env.env.spec.id == "FrozenLake-v1":
            dataset.append(["Temperature", "Map", "Simulations", "Initial State", "Return", "Discounted Return"])
        else:
            dataset.append(["Temperature", "Simulations", "Return", "Discounted Return"])            
        
        with open(file_dir("./../datasets/" + file_name + ".csv"), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(1, n + 1):
                dataset.append(self.run(initial_state_random=initial_state_random))
                if i % (n // 10) == 0:
                    print("%s: %d/%d" % (file_name, i, n))
                    writer.writerows(dataset)
                    dataset = []
        return dataset