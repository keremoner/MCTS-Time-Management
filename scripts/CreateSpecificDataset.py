import src.MCTS as MCTS
from src.Environments import StatelessGym
from src.Experiment import RandomExperiment
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import csv

def file_dir(relative_path):
    absolute_path = os.path.dirname(__file__)
    return os.path.join(absolute_path, relative_path)

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--size')
    parser.add_argument('--interval')
    parser.add_argument('--temperature', help='The temperatures argument, enter the temperatures as an integer, eg. 100')
    parser.add_argument('--dataset-name', help='The dataset_name argument, enter the name of the dataset, eg. 1-16_1000-Cartpole')
    parser.add_argument('--horizon', help='The horizon argument, enter the horizon, eg. 100')

    args = parser.parse_args()
    TEMPERATURE = int(args.temperature)
    SIZE = int(args.size)
    INTERVAL = int(args.interval)
    
    directory = file_dir("../datasets/FrozenLake-v1_m4-4_s1-100_t1/")
    #directory = "../datasets/10k/"
    dataset_names = os.listdir(directory)
    dataset = pd.DataFrame()

    for dataset_name in dataset_names:
        dataset = dataset.append(pd.read_csv(directory + dataset_name), ignore_index=True)
    
    grouped = dataset.groupby(["Map", "Simulations"])["Discounted Return"]
    errors = grouped.std() / (grouped.count() ** 0.5)
    errors = errors[errors > 0.1][INTERVAL * SIZE : (INTERVAL + 1) * SIZE]

    env = StatelessGym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
    agent = MCTS.mcts_agent(horizon=int(args.horizon))
    
    new_dataset = []
    new_dataset.append(["Temperature", "Simulations", "Return", "Discounted Return", "Map"])
    for index, vals in errors.items():
        map = ast.literal_eval(index[0])
        sim = index[1]
        rand_experiment = RandomExperiment(env, agent, temperature=TEMPERATURE, simulations=[sim, sim])
        result = rand_experiment.run() + [map]
        new_dataset.append(result)
        
    with open(file_dir("./../datasets/" + args.dataset_name + ".csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(new_dataset)
