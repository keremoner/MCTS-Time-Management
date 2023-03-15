import src.MCTS as MCTS
from src.Environments import StatelessGym
from src.Experiment import Experiment

TEMPERATURES_1 = [400]
TEMPERATURES_2 = [1]
SIMULATIONS = [2, 4, 8 , 16]
TRIAL = 10

env1 = StatelessGym.make("CartPole-v1")
env1.reset()
done = False
env1.render()
copy_env = env1.get_state()
while not done:
    action = int(input())
    if action == -1:
        env1.set_state(copy_env)
    else:
        next_state, reward, done, _ = env1.step(action)
    env1.render()
    
# for i in range(600):
#     if done:
#         print("Done in ", i, " steps")
#         break
#     next_state, reward, done, info = env1.step(0)
#     print(info)
#     if i % 5 == 0:
#         env1.set_state(copy_env)
#     #env1.render()

# env2 = StatelessGym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
# agent = MCTS.mcts_agent()

# experiment1 = Experiment(env1, agent, temperatures=TEMPERATURES_1, simulations=SIMULATIONS, trial=TRIAL, experiment_name="Cartpole_Base_MCTS")
# experiment2 = Experiment(env2, agent, temperatures=TEMPERATURES_2, simulations=SIMULATIONS, trial=TRIAL, experiment_name="FrozenLake_Base_MCTS")

# experiment1.run()
# experiment1.show_results()

# experiment2.run()
# experiment2.show_results()