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
    parser.add_argument('--random-state', default=str(False))
    parser.add_argument('--initial-state-default', default=str(True))
    parser.add_argument('--low', default=str(-0.05))
    parser.add_argument('--high', default=str(0.05))
    parser.add_argument('--step-size', default=str(0.01))

    # Parse the arguments
    args = parser.parse_args()
    TEMPERATURE = int(args.temperature)
    SIMULATIONS = [int(simulation) for simulation in args.simulations.split(',')]
    SIZE = int(args.size)
    initial_state_random = eval(args.random_state)
    if args.environment == 'FrozenLake-v1':
        env = StatelessGym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
    elif args.environment == 'CartPole-v1':
        initial_state_default = eval(args.initial_state_default)
        low = float(args.low)
        high = float(args.high)
        step_size = float(args.step_size)
        print(initial_state_default, low, high, step_size)
        env = StatelessGym.make("CartPole-v1", initial_state_default=initial_state_default, low=low, high=high, step_size=step_size)
    else:
        env = StatelessGym.make(args.environment)
    agent = MCTS.mcts_agent(horizon=int(args.horizon))
    rand_experiment = RandomExperiment(env, agent, temperature=TEMPERATURE, simulations=SIMULATIONS)
    rand_experiment.create_dataset(SIZE, args.dataset_name, initial_state_random=initial_state_random)
    
    