import src.MCTS as MCTS
from src.Environments import StatelessGym
from src.Experiment import Experiment
import time
import numpy as np
import matplotlib.pyplot as plt

TEMPERATURES = [500]
SIMULATIONS = [1, 2, 4, 8]
TRIAL = 100

env1 = StatelessGym.make('CartPole-v1')
agent = MCTS.mcts_agent()

experiment1 = Experiment(env1, agent, temperatures=TEMPERATURES, simulations=SIMULATIONS, trial=TRIAL, experiment_name="Cartpole_Base_MCTS")

times = []
intervals = []
mean_rewards = []
for temperature in TEMPERATURES:
    errors = []

    for simulation in SIMULATIONS:
        total_time = 0
        rewards = []
        agent.set_temperature(temperature)
        agent.set_simulations(simulation)
        for i in range(TRIAL):
            start_time = time.time()
            cumulative_reward = 0
            env1.reset()
            action = agent.select_action(env1)
            next_state, reward, done, _ = env1.step(action)
            cumulative_reward += reward
            while not done:
                action = agent.select_action(env1)
                next_state, reward, done, _ = env1.step(action)
                cumulative_reward += reward
            rewards.append(cumulative_reward)
            end_time = time.time()
            total_time += end_time - start_time
        times.append(total_time / TRIAL)
        intervals.append(np.std(rewards) / np.sqrt(TRIAL))
        mean_rewards.append(np.mean(rewards))

print("Time taken: ", times)
print("Standard Error: ", intervals)
print("Total time: ", sum(times))
print("Mean rewards: ", mean_rewards)

plt.title("Cartpole-v1 Stateless Gym Fast")
plt.errorbar(SIMULATIONS, times, yerr=intervals)
plt.legend()
plt.xlabel("Number of Simulations")
plt.ylabel("Average time it takes to run a trial (s)")
plt.show()