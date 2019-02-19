# Deep Sets
Tensorflow implementation of a [Deep Sets](https://arxiv.org/pdf/1703.06114.pdf) architecture for inference over sets of objects. 

deepSets.py-- Model and auxilar functions

dataEntropy.py-- Example based on 4.1.1 of [Deep Sets](https://arxiv.org/pdf/1703.06114.pdf). Generates the data using the following procedure:
- We choose 2×2 covariance matrix Σ, and then generated N sample sets from N(0,R(α)ΣR(α)T) of sizeM=300 for N random values of α∈[0,π]. 
- R(α) is the rotation matrix.
- The response variable for each set is the Entropy of the marginal distribution of the first dimension

main.py -- Runs the model using the example data
