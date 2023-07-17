# TODO

## 15.07.2023
- [x] Look at results for Frozen Lake, decide on dataset size
    - 8k-10k seems to be enough for convergence
    - R2 is 6% better with discounted return than return (for gradient boosting regressor)
- [x] Implement randomizer for environments, first Frozen Lake
- [x] Implement dataset generation for randomly parametrized Frozen Lake
- [x] Run dataset generation on cluster
- [x] Run regression for randomly generated FL
    - How to encode the map?
        - Categorical Encoding: Convert letters to numbers, than flatten it
        - One hot encoding
        - 4 channel encoding: For each tile type, create a grid consisting of 0s and 1s indicating whether that cell is the given channel's type
    - Different sized maps? Probably wouldn't work with MLP.
        - Padding can be used. Make the map size constant such as 15x15 and add empty cells for maps smaller than 15x15. 

## 16.07.2023
- [x] Use a little dataset to achieve high training scores for parametrized frozen lake. Decide on NN size. (categorical encoding with constant map size)
    - Best I found was (150, 150, 150, 150) and activation='relu'
- [x] Run a learning curve seperately for the found NN size
    - It looks like it can benefit from more samples
  
## 17.07.2023
- [ ] Get learning curve for FL using 151k samples - (150, 150, 150, 150)
- [ ] Implement one-hot encoding - FL
- [ ] Get learning curve using one-hot encoding - FL constant mapsize
- [ ] Get proper visuals for both CartPole-v1 and FrozenLake-v1, first setting
- [ ] Get results for mean prediction
- [ ] Try to divide the simulation data for frozen lake such that the model is tested on unseen number of simulations
- [ ] Check regression models, see if we can use the combination of features
- [ ] Check supervised learning to see if it is appropriate to use data points with the same X
- [ ] Implement multi-channel map encoding - FL
- [ ] Train using multi-channel map encoding - FL constant mapsize
- [ ] Implement padding for different sized maps
- [ ] Train using different sized maps

## Future
- Feature normalization?
- Look into MLP and how to choose the hyper-parameters
- How are learning curves created? What affects the uncertainty in CV scores?
- Writing the report
    - Check the document format for HPC
    - Write explanation for the current results
    - Write a outline for the whole report
    - Get a feedback for outline and explanations
    - Start writing the introduction
- Better pipeline
    - Dataset generation with folder
    - Experiment information on .yaml
    - Script for generating all of the data analysis with images