import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional

torch.manual_seed(0)  # for reproducibility
hasGPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if hasGPU else "cpu")

#  Sets up the data loader arguments to use parallel processing on the GPU if available.
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if hasGPU else {}

GAUSSIAN_SCALER = 1. / np.sqrt(2.0 * np.pi)

# Defines a Gaussian function with mean mu and standard deviation sigma and evaluates it at x.
def gaussian(x, mu, sigma):
    # Computes the exponential function element-wise.
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    # Clamps the input tensor to the range [min, max].
    return torch.clamp(GAUSSIAN_SCALER / sigma * bell, 1e-10, 1.)  # clip to avoid numerical issues


# Defines a mixture of two Gaussian priors and evaluates it at input. 
# The function returns the log probability of the input under the prior.
def scale_mixture_prior(input, PI, SIGMA_1, SIGMA_2):
    prob1 = PI * gaussian(input, 0., SIGMA_1)
    prob2 = (1. - PI) * gaussian(input, 0., SIGMA_2)
    return torch.log(prob1 + prob2)


# Single Bayesian fully connected Layer with linear activation function
# Gaussian weights and biases
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, parent):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialise weights and bias
        if parent.GOOGLE_INIT: # These are used in the Tensorflow implementation.
            self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0., .05))  # or .01
            self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-5., .05))  # or -4
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0., .05))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-5., .05))
        else: # These are the ones we've been using so far.
            self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0., .1))
            self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-3., -3.))
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0., .1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-3., -3.))

        # Initialise prior and posterior
        self.lpw = 0.
        self.lqw = 0.

        self.PI = parent.PI
        self.SIGMA_1 = parent.SIGMA_1
        self.SIGMA_2 = parent.SIGMA_2
        self.hasScalarMixturePrior = parent.hasScalarMixturePrior

    # Forward propagation
    # The forward function includes the weight sampling step and computes the approximate posterior and prior log probabilities.
    def forward(self, input, infer=False):
        if infer:
            return F.linear(input, self.weight_mu, self.bias_mu)

        # Obtain positive sigma from logsigma, as in paper
        # torch.log Computes the natural logarithm of a tensor element-wise.
        weight_sigma = torch.log(1. + torch.exp(self.weight_rho))
        bias_sigma = torch.log(1. + torch.exp(self.bias_rho))

        # Sample weights and bias
        # Variable: Creates a wrapper for a tensor that allows it to be differentiated during backpropagation.
        epsilon_weight = Variable(torch.Tensor(self.out_features, self.in_features).normal_(0., 1.)).to(DEVICE)
        epsilon_bias = Variable(torch.Tensor(self.out_features).normal_(0., 1.)).to(DEVICE)
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias

        # Compute posterior and prior probabilities
        if self.hasScalarMixturePrior:  # for Scalar mixture vs Gaussian analysis
            self.lpw = scale_mixture_prior(weight, self.PI, self.SIGMA_1, self.SIGMA_2).sum() + scale_mixture_prior(
                bias, self.PI, self.SIGMA_1, self.SIGMA_2).sum()
        else:
            self.lpw = torch.log(gaussian(weight, 0, self.SIGMA_1).sum() + gaussian(bias, 0, self.SIGMA_1).sum())

        self.lqw = torch.log(gaussian(weight, self.weight_mu, weight_sigma)).sum() + torch.log(
            gaussian(bias, self.bias_mu, bias_sigma)).sum()

        # Pass sampled weights and bias on to linear layer
        return F.linear(input, weight, bias)

class BayesianModule(nn.Module):

    """Base class for BNN to enable certain behaviour."""

    def __init__(self):
        super().__init__()

    def kld(self, *args):
        raise NotImplementedError('BayesianModule::kld()')

#  Defines a Bayesian neural network with multiple fully connected layers.
class BayesianNetwork(nn.Module):
    def __init__(self, inputSize, CLASSES, layers, activations, SAMPLES, BATCH_SIZE, NUM_BATCHES, hasScalarMixturePrior,
                 PI, SIGMA_1, SIGMA_2, GOOGLE_INIT=False):
        super().__init__()
        self.inputSize = inputSize
        self.activations = activations
        self.CLASSES = CLASSES
        self.SAMPLES = SAMPLES
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_BATCHES = NUM_BATCHES
        self.DEPTH = 0  # captures depth of network
        self.GOOGLE_INIT = GOOGLE_INIT
        # to make sure that number of hidden layers is one less than number of activation function
        assert (activations.size - layers.size) == 1

        self.SIGMA_1 = SIGMA_1
        self.hasScalarMixturePrior = hasScalarMixturePrior
        if hasScalarMixturePrior == True:
            self.SIGMA_2 = SIGMA_2
            self.PI = PI

        self.layers = nn.ModuleList([])  # To combine consecutive layers
        if layers.size == 0:
            self.layers.append(BayesianLinear(inputSize, CLASSES, self))
            self.DEPTH += 1
        else:
            self.layers.append(BayesianLinear(inputSize, layers[0], self))
            self.DEPTH += 1
            for i in range(layers.size - 1):
                self.layers.append(BayesianLinear(layers[i], layers[i + 1], self))
                self.DEPTH += 1
            self.layers.append(BayesianLinear(layers[layers.size - 1], CLASSES, self))  # output layer
            self.DEPTH += 1

    # Forward propagation and assigning activation functions to linear layers
    # The forward function applies the specified activation functions to each layer and returns the output of the final layer.
    def forward(self, x, infer=False):
        x = x.view(-1, self.inputSize)
        layerNumber = 0
        for i in range(self.activations.size):
            if self.activations[i] == 'relu':
                x = F.relu(self.layers[layerNumber](x, infer))
            elif self.activations[i] == 'softmax':
                x = F.log_softmax(self.layers[layerNumber](x, infer), dim=1)
            else:
                x = self.layers[layerNumber](x, infer)
            layerNumber += 1
        return x

    # returns the summed log probabilities of the posterior and prior over all layers.
    def get_lpw_lqw(self):
        lpw = 0.
        lpq = 0.

        for i in range(self.DEPTH):
            lpw += self.layers[i].lpw
            lpq += self.layers[i].lqw
        return lpw, lpq

    #  The BBB_loss function computes the Evidence Lower Bound (ELBO) loss for training the network, 
    #  using the weight sampling procedure to approximate the posterior distribution over the weights.
    def BBB_loss(self, input, target, batch_idx = None):

        s_log_pw, s_log_qw, s_log_likelihood, sample_log_likelihood = 0., 0., 0., 0.
        for _ in range(self.SAMPLES):
            output = self.forward(input)
            sample_log_pw, sample_log_qw = self.get_lpw_lqw()
            if self.CLASSES > 1:
                # Computes the negative log likelihood loss for a classification task with the specified target.
                sample_log_likelihood = -F.nll_loss(output, target, reduction='sum')
            else:
                sample_log_likelihood = -(.5 * (target - output) ** 2).sum()
            s_log_pw += sample_log_pw
            s_log_qw += sample_log_qw
            s_log_likelihood += sample_log_likelihood

        l_pw, l_qw, l_likelihood = s_log_pw / self.SAMPLES, s_log_qw / self.SAMPLES, s_log_likelihood / self.SAMPLES

        # KL weighting
        if batch_idx is None: # standard literature approach - Graves (2011)
            return (1. / (self.NUM_BATCHES)) * (l_qw - l_pw) - l_likelihood
        else: # alternative - Blundell (2015)
            return 2. ** ( self.NUM_BATCHES - batch_idx - 1. ) / ( 2. ** self.NUM_BATCHES - 1 ) * (l_qw - l_pw) - l_likelihood





