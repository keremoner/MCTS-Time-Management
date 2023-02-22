import gym
import src.MCTS as MCTS
import matplotlib.pyplot as plt
import numpy as np

total = 10

for temp in [1]:
    results = []
    iterations = [1, 2, 4, 8, 16, 32, 64]
    intervals = []
    for iter in iterations:
        rewards = []
        for i in range(0, total):
            agg_reward = 0
            env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
            #env = gym.make("CartPole-v1")
            root_state = env.reset()
            #env.render()
            #agent = MCTSRecursive.mcts_agent(env, cp=temp, simulations=iter, discount_factor=0.997)
            agent = MCTS.mcts_agent(root_state, env, cp=temp, simulations=iter, discount_factor=0.997)
            terminal, new_reward = agent.step()
            agg_reward += new_reward
            while not terminal:
                terminal, new_reward = agent.step()
                agg_reward += new_reward
            rewards.append(agg_reward)
            #print(agg_reward)
        results.append(np.mean(rewards))
        intervals.append(np.std(rewards) / np.sqrt(total))
        print("Sim = ", iter, "\tAverage Aggregate Reward = ", results[len(results) - 1], "\tSd = ", intervals[len(intervals) - 1])
    print("Temp = ", temp, " Results = ", results)
    plt.errorbar(iterations, results, yerr=intervals, label=str(temp))
plt.legend()
plt.xlabel("# of Iterations")
plt.ylabel("Average Aggregate Reward")
plt.show()