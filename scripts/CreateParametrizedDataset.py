import src.MCTS as MCTS
from src.Environments import StatelessGym
from src.Experiment import ParametrizedRandomExperiment
import argparse
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--size')
    parser.add_argument('--temperature', help='The temperatures argument, enter the temperatures as an integer, eg. 100')
    parser.add_argument('--simulations', help='The simulations argument, enter the range of the simulations comma seperated, eg. 1,16')
    parser.add_argument('--environment', help='The environment argument, enter the name of the environment, eg. CartPole-v1')
    parser.add_argument('--dataset-name', help='The dataset_name argument, enter the name of the dataset, eg. 1-16_1000-Cartpole')
    parser.add_argument('--horizon', help='The horizon argument, enter the horizon, eg. 100')
    parser.add_argument('--map_size', help='The map_size argument, enter the min and max map size, eg. 4,8')
    parser.add_argument('--freeze_prob', help='The freeze_prob argument, enter the min and max freeze probability, eg. 0.1,0.9')
    parser.add_argument('--random-state')

    # Parse the arguments
    args = parser.parse_args()
    TEMPERATURE = int(args.temperature)
    SIMULATIONS = [int(simulation) for simulation in args.simulations.split(',')]
    SIZE = int(args.size)
    initial_state_random = bool(args.random_state)
    
    agent = MCTS.mcts_agent(horizon=int(args.horizon))
    
    if args.environment == 'FrozenLake-v1':
        env = StatelessGym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
        map_size = [int(size) for size in args.map_size.split(',')]
        freeze_prob = [float(prob) for prob in args.freeze_prob.split(',')]
        rand_experiment = ParametrizedRandomExperiment(env, agent, temperature=TEMPERATURE, simulations=SIMULATIONS, map_size=map_size, freeze_prob=freeze_prob)
    else:
        env = StatelessGym.make(args.environment)
        rand_experiment = ParametrizedRandomExperiment(env, agent, temperature=TEMPERATURE, simulations=SIMULATIONS)
    
    rand_experiment.create_dataset(SIZE, args.dataset_name, initial_state_random=initial_state_random)