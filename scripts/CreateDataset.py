import src.MCTS as MCTS
from src.Environments import StatelessGym
from src.Experiment import RandomExperiment
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
    parser.add_argument('--random-state')

    # Parse the arguments
    args = parser.parse_args()
    TEMPERATURE = int(args.temperature)
    SIMULATIONS = [int(simulation) for simulation in args.simulations.split(',')]
    SIZE = int(args.size)
    initial_state_random = bool(args.random_state)
    if args.environment == 'FrozenLake-v1':
        env = StatelessGym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
    else:
        env = StatelessGym.make(args.environment)
    
    agent = MCTS.mcts_agent(horizon=int(args.horizon))
    rand_experiment = RandomExperiment(env, agent, temperature=TEMPERATURE, simulations=SIMULATIONS)
    rand_experiment.create_dataset(SIZE, args.dataset_name, initial_state_random=initial_state_random)
    
    