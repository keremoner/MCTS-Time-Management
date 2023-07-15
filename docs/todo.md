# TODO

## 15.07.2023
- [x] Look at results for Frozen Lake, decide on dataset size
    - 8k-10k seems to be enough for convergence
    - R2 is 6% better with discounted return than return (for gradient boosting regressor)
- [x] Implement randomizer for environments, first Frozen Lake
- [x] Implement dataset generation for randomly parametrized Frozen Lake
- [x] Run dataset generation on cluster
- [ ] Get results for mean prediction
- [ ] Try to divide the simulation data for frozen lake such that the model is tested on unseen number of simulations
- [ ] Check regression models, see if we can use the combination of features
- [ ] Check supervised learning to see if it is appropriate to use data points with the same X
- [ ] Get proper visuals for both CartPole-v1 and FrozenLake-v1
- [ ] Run regression for randomly generated FL
    - How to encode the map?
    - Look into MLP and how to choose the hyper-parameters
    - Different sized maps? Probably wouldn't work with MLP.
    - Feature normalization?

## Future
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