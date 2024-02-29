# Credibility in network regression with σ-hot encoding and weight balancing

Actuarial credibility can be modeled by adding random effects to risk groups. In recent literature on neural network regression several extensions are proposed following the example of generalized linear mixed models (GLMMs).

We take a different approach by equalizing categorical and continuous input variables while keeping the original network. Standardization of continuous variables is generally considered crucial for the learning capacity of a network. Building on this, we introduce σ-hot encoding as the alternative to one-hot encoding, using relative frequencies of observed categories to set input values. At the same time, we propose weight balancing for each categorical variable during training of the network. This enables an equal playing field for all input variables and ensures only minor deviating responses of relatively sparse categories.

With density estimation under maximum likelihood and constrain of the max norm of all weights, we illustrate σ-hot encoding and weight balancing in two examples. In the first example we repeat a simulation study applied to GLMMNet and show that our method produces similar results. In the second example our simulation is showing promising results for a more complex disability insurance which does not adhere to the conditions of recent network models.

In conclusion, we believe σ-hot encoding and weight balancing of categorical variables to solve actuarial credibility in an original Machine Learning way. This can broaden the application of credibility in an assumption free setting, and therefore increase the use of neural networks in the insurance industry.

