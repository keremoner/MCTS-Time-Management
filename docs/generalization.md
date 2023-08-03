# How to measure generalization and a proper way to evaluate models

## First Setting - Learning to predict expected discounted return given number of simulations per step

### Experiment
 - Use the same map for FrozenLake - 4x4 given map
 - Create a dataset for different number of simulations - between certain range, e.g. 1 - 150
   - where standard error of mean for each sim is under some thershold, e.g. 0.01
 - For differing amount of training set sizes
   - There will be certainly coinciding datapoints with same sim numbers as we created the dataset by checking standard error
   - Randomly sample sim numbers between the min and max number number of sims 
     - Number of samples = 20% of possible sim numbers
   - Took out the datapoints for those simulation numbers 
     - Calculate the mean for each number of simulations, along with their std error
     - This is the test set
   - From the remaining points in the original dataset
     - Sample number of points equaling to the current training set size
     - This is the current training set
   - 
   - Run a supervised learning algo on the training set
   - Find means for each sim number in the training set - training_means
   - Predict values for training_means and calculate MSE
   - Predict the values for test set and calculate MSE
   - Difference between those MSE shows the overfitting and generalization
   - Run this for 5-10 times and take the average for MSE values
   - If the std deviation for MSE values is high, it means which sim numbers are chosen for the training and test are significant in generalization
     - This effect is expected to be higher with lower training set sizes
### Results
 - Training errors stay close to zero for all number of training sizes
 - Test errors are higher for low training set sizes, but converge quickly to training errors
 - After 2-5 training sizes, the model starts to perform good on the test
 - This result shows that the model is able to generalize for unseen number of simulations per step pretty well (sim number 1 - 100)
 - approx. 2-3 datapoints for each number of simulation seems to be enough for good generalizations

## Second Setting - Learning to predict expected discounted return given the number of sims per step and the parameters
 - Now, our goal is to generalize both for unseen number of simulations and unseen environments (unseen maps for frozenlake)
 - For simplicity, I will first try with 4x4 maps
 - I checked my dataset of 4x4 FrozenLake with sims 1 - 100
   - Size: 151k
   - Unique maps: 3823
   - Mean std error for a map: 0.064
   - Max std error for a map: 0.14

### Experimentation
 - I picked out unique maps for test set and used remaining for training
 - Prediction is done for mean - grouped by map and simulation

### Results
 - It shows that the model is able to generalize
 - Training error increases as the training set size increases
   - Reason for that is my original dataset size is not large enough (guess)