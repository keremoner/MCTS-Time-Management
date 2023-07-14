import src.MCTS as MCTS
from src.Environments import StatelessGym
from src.Experiment import Experiment
import gym
import copy

TEMPERATURES_1 = [500]
TEMPERATURES_2 = [1]
SIMULATIONS = [100]
TRIAL = 1

env1 = StatelessGym.make("CartPole-v1")
env2 = StatelessGym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)

agent = MCTS.mcts_agent()

experiment1 = Experiment(env1, agent, temperatures=TEMPERATURES_1, simulations=SIMULATIONS, trial=TRIAL, experiment_name="Cartpole_Base_MCTS_test")
#experiment2 = Experiment(env2, agent, temperatures=TEMPERATURES_2, simulations=SIMULATIONS, trial=TRIAL, experiment_name="FrozenLake_Base_MCTS")

experiment1.run(save=True)
experiment1.show_results()

# experiment2.run()
# experiment2.show_results()



# env1.reset()
# done = False
# copy_env = env1.get_state()
# print(copy_env)
# while not done:
#     action = int(input())
#     if action == -1:
#         env1.set_state(copy_env)
#     else:
#         next_state, reward, done, _ = env1.step(action)
#         print(next_state)
# #     env1.render()


# total_reward = 0
# for i in range(1000):
#     if done:
#         print("Done in ", i, " steps")
#         break
#     next_state, reward, done, info = env1.step(0)
#     total_reward += reward
#     #print(next_state)
#     if i % 5 == 0:
#         env1.set_state(copy_env)
#     #print(i)
#     #env1.render()
# print(total_reward)
