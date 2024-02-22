import src.MCTS as MCTS
from src.Environments import StatelessGym
from src.Experiment import Experiment, RandomExperiment, ParametrizedRandomExperiment
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.metrics import median_absolute_error, mean_squared_log_error, max_error
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance
from sklearn.preprocessing import OneHotEncoder
import ast
import math
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self, input_size=2):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            return self(torch.tensor(x, dtype=torch.float32))


def encode_maze(maze):
    num_rows = len(maze)
    num_cols = len(maze[0])

    encoded_maze = []

    for i in range(num_rows):
        for j in range(num_cols):
            if maze[i][j] == 'S':
                encoded_maze.append(0)
            elif maze[i][j] == 'F':
                encoded_maze.append(1)
            elif maze[i][j] == 'H':
                encoded_maze.append(2)
            elif maze[i][j] == 'G':
                encoded_maze.append(3)
            elif maze[i][j] == 'E':
                encoded_maze.append(4)
    return encoded_maze

def add_padding(map, target_size):
    current_size = len(map)
    diff = target_size - current_size
    if diff < 0:
        raise Exception("Current map size is greater than target size")
    elif diff == 0:
        return map
    else:
        result = []
        padding = diff // 2
        left_out = diff % 2
        for i in range(padding):
            result.append('E' * target_size)
        for row in map:
            new_row = 'E' * padding + row + 'E' * padding + 'E' * left_out
            result.append(new_row)
        for i in range(padding + left_out):
            result.append('E' * target_size)
        return result
    
def encode_map(map, categories='auto'):
    # Convert the map to a 2D array
    map_array =  []
    for row in map:
        for letter in row:
            map_array.append([letter])
    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder(sparse=False, categories=categories)

    # Fit and transform the map array
    encoded_map = encoder.fit_transform(map_array).astype('int64')

    # Get the categories (unique values) from the encoder
    categories = encoder.categories_[0]

    # Create a dictionary to map the encoded values to the original categories
    category_mapping = {i: category for i, category in enumerate(categories)}

    # Return the encoded map and the category mapping
    return encoded_map, encoder.categories_, category_mapping

            
def file_dir(relative_path):
    absolute_path = os.path.dirname(__file__)
    return os.path.join(absolute_path, relative_path)

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()
    result_string = ""
    # Add arguments to the parser
    parser.add_argument('--experiment-code')
    args = parser.parse_args()
    os.mkdir('../results/' + args.experiment_code + '/')
    print("Experiment code: ", args.experiment_code)
    # DATASET LOAD
    directory = "../datasets/CartPole-v1_disc-0.05_s1-100_t500/"
    dataset_names = os.listdir(directory)
    dataset = pd.DataFrame()
    
    for dataset_name in dataset_names:
        dataset = dataset.append(pd.read_csv(directory + dataset_name), ignore_index=True)
    print ("Dataset Loaded Successfully: ", len(dataset))
    
    # PARAMETERS
    padding = 4
    NN = True
    n_epochs = 100000
    n_train = 100000
    fold = 4
    test_sims = [ 3, 5, 9, 13, 14, 18, 21, 25, 26, 27, 29, 38, 41, 43, 47, 48, 53, 55, 59, 60, 65, 67, 69, 70, 75, 78, 84, 85, 86, 89, 90, 96, 98]
    #test_sims = np.sort(np.random.choice(np.arange(sim_min, sim_max + 1), size=math.ceil((sim_max - sim_min + 1) * 0.33), replace=False))
    #train_sizes = [100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 750000]
    #train_sizes = [600]
    train_sizes = [25600, 51200, 102400]    

    if 'Map' in dataset.columns:
        if padding > 0: 
            dataset['List_Map'] = dataset['Map'].apply(ast.literal_eval).apply(lambda x: add_padding(x, padding))
        else: 
            dataset['List_Map'] = dataset['Map'].apply(ast.literal_eval)
        #dataset['F_count'] = dataset['Map'].apply(lambda x: sum(row.count('F') for row in x))
        dataset['Encoded_Map'] = dataset['List_Map'].apply(lambda x: encode_maze(x))
        categories = encode_map(dataset['List_Map'].iloc[0])[1]
        dataset['OneHotEncoded_Map'] = dataset['List_Map'].apply(lambda x: np.reshape(encode_map(x, categories)[0], (-1)))
    
    models = {
    #'LinearRegression': LinearRegression(),
    # #'Ridge': Ridge(alpha=1.0),
    # #'Lasso': Lasso(alpha=1.0),
    # #'ElasticNet':  ElasticNet(alpha=1.0, l1_ratio=0.5),
    #'SVR': SVR(),
    #'DecisionTreeRegressor': DecisionTreeRegressor(),
    #'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, max_depth=10),
    #'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=5),
    #'MLPRegressor': MLPRegressor(hidden_layer_sizes=(200, 200, 200, 200, 200), activation='tanh', max_iter=1000000, n_iter_no_change=10, tol=1e-4)
    }
    
    #Getting min and max number of simulations
    sim_min = dataset['Simulations'].min()
    sim_max = dataset['Simulations'].max()

    #Features to be used in the model
    features = ['Simulations', 'Cart Velocity', 'Cart Position', 'Pole Angle', 'Pole Angular Velocity']
    #features = ['Simulations']

    #Creating test set by taking average for test set simulations
    test_set = dataset[dataset['Simulations'].isin(test_sims)].groupby(features)['Discounted Return'].mean()
    data = test_set.index.values
    test_set.to_csv('../results/' + args.experiment_code + '/test_set.csv')
    if len(features) > 1:
        test_set_x = [[i for i in x] for x in data]
    else: 
        test_set_x = data.reshape(-1, 1)

    # test_set_x = np.hstack((categorical_values, numerical_values))
    # test_set_x = test_set.index.values.reshape(-1, len(features))
    test_set_y = test_set.values


    #train_sizes = list(range(100, 1001, 125)) + list(range(1000, 5001, 250)) + [10000, 20000, 40000, 80000]
    train_scores = []
    train_scores2 = []
    test_scores = []
    train_scores_abs = []
    train_scores2_abs = []
    test_scores_abs = []
    
    for training_set_size in  train_sizes:
        train_scores.append([])
        train_scores2.append([])
        test_scores.append([])
        train_scores_abs.append([])
        train_scores2_abs.append([])
        test_scores_abs.append([])
        
        for i in range(fold):
            #Randomly sampling number of simulations to be included in the test and training set
            training_sims = np.setdiff1d(np.arange(sim_min, sim_max + 1), test_sims)
            
            #Creating training set by sampling sim numbers for training set from the remainnig datapoints
            training_set = dataset[dataset['Simulations'].isin(training_sims)].sample(n=training_set_size, replace=True)
            training_set.to_csv('../results/' + args.experiment_code + '/training_set_' + str(training_set_size) + '_' + str(i) + '.csv')
            training_set_x = training_set[features].values.reshape(-1, len(features))
            training_set_y = training_set['Discounted Return'].values
            
            #Creating training score sets
            training_score_set = training_set.groupby(features)['Discounted Return'].mean()
            data = training_score_set.index.values
            if len(features) > 1:
                training_score_set_x = [[i for i in x] for x in data]
            else: 
                training_score_set_x = data.reshape(-1, 1)
            # training_score_set_x = training_score_set.index.values.reshape(-1, len(features))
            training_score_set_y = training_score_set.values
            
            training_score_set2 = dataset[dataset['Simulations'].isin(training_sims)].groupby(features)['Discounted Return'].mean()
            data = training_score_set2.index.values
            if len(features) > 1:
                training_score_set2_x = [[i for i in x] for x in data]
            else: 
                training_score_set2_x = data.reshape(-1, 1)
            training_score_set2_y = training_score_set2.values

            #Training
            if NN:
                model = MyModel(input_size=len(features))
                loss_fn = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
                batch_size = len(training_set_x)
                
                for iter in range(n_train):
                    # indices = list(range(len(training_set_x)))
                    # sample_indices = np.random.choice(indices, batch_size, replace=False)
                    Xbatch = torch.tensor(training_set_x, dtype=torch.float32)
                    y_pred = model(Xbatch)
                    ybatch = torch.tensor(training_set_y, dtype=torch.float32).reshape(-1, 1)
                    loss = loss_fn(y_pred, ybatch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if iter == 0 or (iter + 1) % 1000 == 0:
                        print(f'Finished epoch {iter + 1}, latest loss {loss}')
            else: 
                model = GradientBoostingRegressor(n_estimators=200, max_depth=20)
                model.fit(np.asarray(training_set_x).astype('float32'), training_set_y)
            
            #Predicting on test set
            y_pred = model.predict(test_set_x)
            #Calculating MSE
            test_score = mean_squared_error(test_set_y, y_pred)
            test_scores[-1].append(test_score)
            
            test_score_abs = mean_absolute_error(test_set_y, y_pred)
            test_scores_abs[-1].append(test_score_abs)
            
            
            #Predicting on training set
            y_pred = model.predict(training_score_set_x)
            train_score = mean_squared_error(training_score_set_y, y_pred)
            train_scores[-1].append(train_score)
            train_score_abs = mean_absolute_error(training_score_set_y, y_pred)
            train_scores_abs[-1].append(train_score_abs)
            
            y_pred = model.predict(training_score_set2_x)
            train_score2 = mean_squared_error(training_score_set2_y, y_pred)
            train_scores2[-1].append(train_score2)
            train_score2_abs = mean_absolute_error(training_score_set2_y, y_pred)
            train_scores2_abs[-1].append(train_score2_abs)
            print("\n\nSaving model\n\n")
            if NN:
                torch.save(model.state_dict(), '../results/' + args.experiment_code + '/model.pt')
            else:
                pickle.dump(model, '../results/' + args.experiment_code + '/model.sav')  
                    
        result_string += "Training set size: %d\nTraining error 1: %f ± %f\nTraining error 2: %f ± %f\nTest error: %f ± %f\n" % (training_set_size, np.mean(train_scores[-1]), np.std(train_scores[-1]) / (fold ** 0.5), np.mean(train_scores2[-1]), np.std(train_scores2[-1]) / (fold ** 0.5), np.mean(test_scores[-1]), np.std(test_scores[-1]) / (fold ** 0.5))
        print(result_string)
    
    with open('./../results/' + args.experiment_code + '/result-string.txt', 'w') as f:
        print(result_string, file=f)
    
    with open('./../results/' + args.experiment_code + '/arrays.txt', 'w') as f:
        print(test_scores, file=f)
        print(train_scores, file=f)
        print(train_scores2, file=f)
        print(test_scores_abs, file=f)
        print(train_scores_abs, file=f)
        print(train_scores2_abs, file=f)
        
    # Calculate the mean and standard deviation of the training and test scores
    train1_mean = np.mean(train_scores, axis=1)
    train1_std = np.std(train_scores, axis=1) / (fold ** 0.5)
    train2_mean = np.mean(train_scores2, axis=1)
    train2_std = np.std(train_scores2, axis=1) / (fold ** 0.5)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1) / (fold ** 0.5)

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train1_mean, marker='o', markersize=4, label='Training error 1')
    plt.plot(train_sizes, train2_mean, marker='o', markersize=4, label='Training error 2')
    plt.plot(train_sizes, test_mean, marker='o', markersize=4, label='Test error')

    # Add error bands showing the standard deviation
    plt.fill_between(train_sizes, train1_mean - train1_std, train1_mean + train1_std, alpha=0.1)
    plt.fill_between(train_sizes, train2_mean - train2_std, train2_mean + train2_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

    # Add labels and title
    plt.xlabel('Training Set Size')
    plt.ylabel('MSE')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.ylim([-5, 2000])
    plt.savefig("../results/" + args.experiment_code + "/curve.png")
