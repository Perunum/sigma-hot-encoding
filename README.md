# Credibility in network regression with σ-hot encoding and weight balancing

Actuarial credibility can be modeled by adding random effects to risk groups. In recent literature on neural network regression several extensions are proposed, following the example of generalized linear mixed models (GLMMs).

We take a different approach by equalizing categorical and continuous input variables while keeping the original network. Standardization of continuous variables is generally considered crucial for the learning capacity of a network. Building on this, we introduce σ-hot encoding as the alternative of one-hot encoding, where relative frequencies of observed categories are used to set input values. At the same time, we propose weight balancing for each categorical variable during training. This enables an equal level playing field for all input variables and ensures only small deviating responses of relatively sparse categories.

We illustrate σ-hot encoding and weight balancing in two examples, where we use density estimation under maximum likelihood and constrain the max norm of all weights. In the first example we repeat a simulation study applied to GLMMNet and show that our results are comparably well. In the second example we simulate a more complex disability insurance that does not meet the requirements of recent network models, also showing good results.

σ-hot encoding and weight balancing of categorical variables solve credibility in an original Machine Learning way. They can broaden the application of credibility in an assumption free setting, without even having to distinguish between fixed and randomized categorical variables. This general and easy credibility could stimulate the use of neural networks in the insurance practice.
