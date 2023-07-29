## Todo
- [ ] Check datasets for FrozenLake different sized maps
  - [ ] Calculate how much time is required for each map size
  - [ ] Run scripts on cluster to get the maps
- [ ] Modify the dataset generation code so that it can generate samples for a specific map
  - [ ] Generate samples for a specific map (7x7 - 8x8 - 9x9 - 10x10) using cluster
- [ ] Implement mean prediction version of regression

## Add to LaTeX
- Cartpole-v1 Only Simulation results
  - [ ] Simulation vs Discounted Return graph (one with std error one with std deviation)
  - [ ] Regression results for different methods
  - [ ] Plot the learning curve for best method
  
- Cartpole-v1 Simulation + Inital state results
  - [ ] Simulation vs Discounted Return graph
  - [ ] Regression results for different methods
  - [ ] Do a grid search for NN
  - [ ] Plot the learning curve for NN
  - [ ] Add variation to the initial state - create new dataset
  - [ ] Do the same things for cartpole with variation

- FrozenLake-v1 Only Simulation results
  - [ ] Simulation vs Discounted Return Graph
  - [ ] Regression results for different methods
  - [ ] Plot the learning curve for best method

- FrozenLake-v1 Same Map Size Different Maps
  - [ ] Simulation vs Discounted Return Graph
  - Categorical Encoding
    - [ ] Do grid search for NN
    - [ ] Plot the learning curve for NN
  - One hot encoding
    - [ ] Do grid search for NN
    - [ ] Plot the learning curve for NN
  - [ ] Put categorical encoding and one-hot encoding side-to-side
  - [ ] Show different maps from the dataset
    - [ ] Find an easy and a hard map
    - [ ] Show that for the same number of simulations predictions differ for each map

- FrozenLake-v1 Different Maps and Sizes
  - [ ] Get maps for 7x7
  - [ ] Simulation vs Discounted Return Graph
  - [ ] Add padding to make them 10x10
  - Categorical Encoding
    - [ ] Do grid search for NN
    - [ ] Plot the learning curve for NN
  - [ ] Show results for a specific map of size that has not been trained
  - One hot encoding
    - [ ] Do grid search for NN
    - [ ] Plot the learning curve for NN
  - Multi-Channel encoding
    - [ ] Implement multi-channel encoding
    - [ ] Do grid search for NN
    - [ ] Plot the learning curve for NN