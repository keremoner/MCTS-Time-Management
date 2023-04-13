import src.MCTS as MCTS
from src.Environments import StatelessGym
from src.Experiment import Experiment
import argparse

TEMPERATURES = [250, 500]
SIMULATIONS = [2, 4]
TRIAL = 10


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--temperatures', help='The temperatures argument, enter the temperatures comma seperated, eg. 250,500')
    parser.add_argument('--simulations', help='The simulations argument, enter the simulations comma seperated, eg. 2,4,8,16')
    parser.add_argument('--trial', help='The trial argument, enter the number of trials, eg. 10')
    parser.add_argument('--environment', help='The environment argument, enter the name of the environment, eg. CartPole-v1')
    parser.add_argument('--experiment_name', help='The experiment_name argument, enter the name of the experiment, eg. Cartpole_Base_MCTS')
    parser.add_argument('--horizon', help='The horizon argument, enter the horizon, eg. 100')
    parser.add_argument('--save', action='store_true', help='To save the results')

    # Parse the arguments
    args = parser.parse_args()
    TEMPERATURES = [int(temperature) for temperature in args.temperatures.split(',')]
    SIMULATIONS = [int(simulation) for simulation in args.simulations.split(',')]
    TRIAL = int(args.trial)
    if args.environment == 'FrozenLake-v0':
        env = StatelessGym.make("FrozenLake-v0", desc=None, map_name="4x4", is_slippery=False)
    else:
        env = StatelessGym.make(args.environment)
    
    agent = MCTS.mcts_agent(horizon=int(args.horizon))
    experiment = Experiment(env, agent, temperatures=TEMPERATURES, simulations=SIMULATIONS, trial=TRIAL, experiment_name=args.experiment_name)
    experiment.run(save=True) 
    
    