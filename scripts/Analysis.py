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
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import os
import pandas as pd
import ast
import numpy as np
import argparse
from sklearn.preprocessing import OneHotEncoder
import pickle

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

def encode_map(map):
    # Convert the map to a 2D array
    map_array =  []
    for row in map:
        for letter in row:
            map_array.append([letter])
    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder(sparse=False)

    # Fit and transform the map array
    encoded_map = encoder.fit_transform(map_array).astype('int64')

    # Get the categories (unique values) from the encoder
    categories = encoder.categories_[0]

    # Create a dictionary to map the encoded values to the original categories
    category_mapping = {i: category for i, category in enumerate(categories)}

    # Return the encoded map and the category mapping
    return encoded_map, category_mapping

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

def file_dir(relative_path):
    absolute_path = os.path.dirname(__file__)
    return os.path.join(absolute_path, relative_path)
    
if __name__ == "__main__":
    directory = file_dir("../datasets/after_bug/FrozenLake-v1_m4-6_s1-100_t1/")
    #directory = "../datasets/10k/"
    dataset_names = os.listdir(directory)
    dataset = pd.DataFrame()
    parser = argparse.ArgumentParser()

    for dataset_name in dataset_names:
        dataset = dataset.append(pd.read_csv(directory + dataset_name), ignore_index=True)

    # Add arguments to the parser
    parser.add_argument('--size', '-s', default=str(dataset.shape[0]))
    parser.add_argument('--cores', '-c', default=str(1))
    args = parser.parse_args()
    size = int(args.size)
    cores = int(args.cores)
    
    padding = 7

    if 'Map' in dataset.columns:
        if padding > 0: 
            dataset['Map'] = dataset['Map'].apply(ast.literal_eval).apply(lambda x: add_padding(x, padding))
        else: 
            dataset['Map'] = dataset['Map'].apply(ast.literal_eval)
        #dataset['F_count'] = dataset['Map'].apply(lambda x: sum(row.count('F') for row in x))
        dataset['Encoded_Map'] = dataset['Map'].apply(lambda x: encode_maze(x))
        #dataset['OneHotEncoded_Map'] = dataset['Map'].apply(lambda x: np.reshape(encode_map(x)[0], (-1)))
    
    #features = ['Simulations', 'Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']
    #features = ['Simulations']
    features = ['Simulations', 'Encoded_Map']

    if 'Encoded_Map' in features:
        features.remove('Encoded_Map')
        X = np.append(dataset[features].values.reshape(-1, len(features)), dataset['Encoded_Map'].apply(pd.Series).values, axis=1)
        print(type(X))
    elif 'OneHotEncoded_Map' in features:
        features.remove('OneHotEncoded_Map')
        X = np.append(dataset[features].values.reshape(-1, len(features)), dataset['OneHotEncoded_Map'].apply(pd.Series).values, axis=1).astype('int64')
    else:
        X = dataset[features].values.reshape(-1, len(features))
    y = dataset['Discounted Return'].values
    
    models = {
    #'LinearRegression': LinearRegression(),
    # #'Ridge': Ridge(alpha=1.0),
    # #'Lasso': Lasso(alpha=1.0),
    # #'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
    #'SVR': SVR(),
    #'DecisionTreeRegressor': DecisionTreeRegressor(),
    #'RandomForestRegressor': RandomForestRegressor(),
    #'GradientBoostingRegressor': GradientBoostingRegressor(),
    #'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=5),
    'MLPRegressor': MLPRegressor(hidden_layer_sizes=(150, 200, 200, 150), activation='tanh', learning_rate='adaptive', max_iter=1000000)
    }

    train_sizes, train_scores, test_scores = learning_curve(models['MLPRegressor'], X[:size], y[:size], cv=10, train_sizes=np.append(np.linspace(0.001, 0.1, 10, endpoint=False), np.linspace(0.1, 1.0, 10)), scoring='r2', n_jobs=cores, verbose=2)

    # Calculate the mean and standard deviation of the training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    models['MLPRegressor'].fit(X[:size], y[:size])
    filename = file_dir('../results/after_bug/FrozenLake/FL_m4-6_encodedmap_150-150-150-150.sav')
    pickle.dump(models['MLPRegressor'], open(filename, 'wb'))
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')

    # Add error bands showing the standard deviation
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)


    # Add labels and title
    plt.xlabel('Training Set Size')
    plt.ylabel('R2 Score')
    plt.title('Learning Curve | FrozenLake-v1 (m4-6) | Temp=1 | MLPRegressor (150,200,200,150)')
    plt.legend(loc='best')
    plt.savefig(file_dir('../results/after_bug/FrozenLake/learning_curve_m4-6_150-200-200-150.png'))
    df = pd.DataFrame({'train_sizes': train_sizes, 'train_mean': train_mean, 'train_std': train_std, 'test_mean': test_mean, 'test_std': test_std})
    df.to_csv(file_dir('../results/after_bug/FrozenLake/learning_curve_m4-6_150-200-200-150.csv'), index=False)
