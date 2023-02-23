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
            self.epsilon_normal = torch.distributions.Normal(0, 1)
        else: # These are the ones we've been using so far.
            self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0., .1))
            self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-3., -3.))
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0., .1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-3., -3.))
            self.epsilon_normal = torch.distributions.Normal(0, 1)

        # Initialise prior and posterior
        self.lpw = 0.
        self.lqw = 0.

        # uncomment for mixture
        #self.PI = parent.PI
        self.SIGMA_1 = parent.SIGMA_1
        #self.SIGMA_2 = parent.SIGMA_2
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

        act_mu = F.linear(input, self.weight_mu)
        act_std = torch.sqrt(F.linear(input.pow(2), weight_sigma.pow(2)))

        # Sample weights and bias
        # Variable: Creates a wrapper for a tensor that allows it to be differentiated during backpropagation.
        epsilon_weight = Variable(torch.Tensor(self.out_features, self.in_features).normal_(0., 1.)).to(DEVICE)
        epsilon_bias = Variable(torch.Tensor(self.out_features).normal_(0., 1.)).to(DEVICE)

        # LRT option
        w_eps = self.epsilon_normal.sample(act_mu.size())
        bias_eps = self.epsilon_normal.sample(bias_sigma.size())
        

        #weight = self.weight_mu + weight_sigma * epsilon_weight
        #bias = self.bias_mu + bias_sigma * epsilon_bias

        # LRT option
        weight = act_mu + act_std * w_eps
        bias = self.bias_mu + bias_sigma * bias_eps


        w_kl = self.kld(
            mu_prior=0.0,
            std_prior=self.SIGMA_1,
            mu_posterior=self.weight_mu,
            std_posterior=weight_sigma
        )

        bias_kl = self.kld(
            mu_prior=0.0,
            std_prior=0.1,
            mu_posterior=self.bias_mu,
            std_posterior=bias_sigma
        )

        self.kl_divergence = w_kl + bias_kl
        

        # Compute posterior and prior probabilities
        if self.hasScalarMixturePrior:  # for Scalar mixture vs Gaussian analysis
            self.lpw = scale_mixture_prior(weight, self.PI, self.SIGMA_1, self.SIGMA_2).sum() + scale_mixture_prior(
                bias, self.PI, self.SIGMA_1, self.SIGMA_2).sum()
        else:
            self.lpw = torch.log(gaussian(weight, 0, self.SIGMA_1).sum() + gaussian(bias, 0, self.SIGMA_1).sum())

        #print(weight.shape)
        #print(torch.squeeze(self.weight_mu).shape)
        #print(weight_sigma.shape)

        self.lqw = torch.log(gaussian(weight, torch.squeeze(self.weight_mu), torch.squeeze(weight_sigma))).sum() + torch.log(
            gaussian(bias, self.bias_mu, bias_sigma)).sum()

        # Pass sampled weights and bias on to linear layer
        #print("W", weight.shape)
        #print("BIAS", bias.shape)
        #print(torch.transpose(input, 0, 1).shape)
        #
        #return F.linear(torch.transpose(input, 0, 1), torch.transpose(weight, 0, 1), torch.squeeze(bias))
        return F.linear(input, torch.transpose(weight, 0, 1), bias)

        #return weight + bias

    def kld(self,
            mu_prior: float,
            std_prior: float,
            mu_posterior: torch.Tensor,
            std_posterior: torch.Tensor) -> torch.Tensor:

        """Calculates the KL Divergence.
        The only 'downside' to the local reparameterisation trick is
        that, as the weights are not being sampled directly, the KL
        Divergence can not be calculated through the use of MC sampling.
        Instead, the closed form of the KL Divergence must be used;
        this restricts the prior and posterior to be Gaussian.
        However, the use of a Gaussian prior / posterior results in a
        lower variance and hence faster convergence.
        Parameters
        ----------
        mu_prior : float
            Mu of the prior normal distribution.
        std_prior : float
            Sigma of the prior normal distribution.
        mu_posterior : Tensor
            Mu to approximate the posterior normal distribution.
        std_posterior : Tensor
            Sigma to approximate the posterior normal distribution.
        Returns
        -------
        Tensor
            Calculated KL Divergence.
        """

        kl_divergence = 0.5 * (
                2 * torch.log(std_prior / std_posterior) -
                1 +
                (std_posterior / std_prior).pow(2) +
                ((mu_prior - mu_posterior) / std_prior).pow(2)
        ).sum()

        return kl_divergence


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
        #print("INPUT SIZE", self.inputSize)
        #print("INPUT SIZE", x.shape)
        #print("ACT", self.activations.size)
        layerNumber = 0
        for i in range(self.activations.size):
            
            if self.activations[i] == 'relu':
                x = F.relu(self.layers[layerNumber](x, infer))
                #print("relu")
            elif self.activations[i] == 'softmax':
                x = F.log_softmax(self.layers[layerNumber](x, infer), dim=1)
            else:
                #print(self.layers[layerNumber])
                x = self.layers[layerNumber](x, infer)
                #print(x.shape)
                #print("last")
                

            layerNumber += 1
        
        return x

    # returns the summed log probabilities of the posterior and prior over all layers.
    def get_lpw_lqw(self):
        lpw = 0.
        lpq = 0.

        for i in range(self.DEPTH):
            #lpw += self.layers[i].lpw
            #lpq += self.layers[i].lqw
            lpq += self.layers[i].kl_divergence
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
                print("TARGET", target.shape)
                print("OUTPUT", output.shape)
                sample_log_likelihood = -(.5 * (target - torch.squeeze(output)) ** 2).sum()
                print("LOG!", sample_log_likelihood)
            s_log_pw += sample_log_pw
            s_log_qw += sample_log_qw
            s_log_likelihood += sample_log_likelihood
            print("LOG!", s_log_likelihood)


        l_pw, l_qw, l_likelihood = s_log_pw / self.SAMPLES, s_log_qw / self.SAMPLES, s_log_likelihood / self.SAMPLES

        # KL weighting
        if batch_idx is None: # standard literature approach - Graves (2011)
            #return (1. / (self.NUM_BATCHES)) * (l_qw - l_pw) 
            print("LOG",l_likelihood)
            return (1. / (self.NUM_BATCHES)) * (l_qw - l_pw) - l_likelihood
        else: # alternative - Blundell (2015)
            # 
            return 2. ** ( self.NUM_BATCHES - batch_idx - 1. ) / ( 2. ** self.NUM_BATCHES - 1 ) * (l_qw - l_pw) - l_likelihood
            #return 2. ** ( self.NUM_BATCHES - batch_idx - 1. ) / ( 2. ** self.NUM_BATCHES - 1 ) * (l_qw - l_pw) 

    



# code taken from danielkelshaw/WeightUncertainty to implement Local Reparametrization Trick
class BayesLinearLRT(BayesianModule):

    """Bayesian Linear Layer with Local Reparameterisation Trick.
    Implementation of a Bayesian Linear Layer utilising the 'local
    reparameterisation trick' in order to sample directly from the
    activations.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 std_prior: Optional[float] = 1.0) -> None:

        """Bayesian Linear Layer with Local Reparameterisation Trick.
        Parameters
        ----------
        in_features : int
            Number of features to feed into the layer.
        out_features : int
            Number of features produced by the layer.
        std_prior : float
            Sigma to be used for the normal distribution in the prior.
        """

        super().__init__()

        self.in_feature = in_features
        self.out_feature = out_features
        self.std_prior = std_prior

        w_mu = torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        self.w_mu = nn.Parameter(w_mu)

        w_rho = torch.empty(out_features, in_features).uniform_(-5.0, -4.0)
        self.w_rho = nn.Parameter(w_rho)

        bias_mu = torch.empty(out_features).uniform_(-0.2, 0.2)
        self.bias_mu = nn.Parameter(bias_mu)

        bias_rho = torch.empty(out_features).uniform_(-5.0, -4.0)
        self.bias_rho = nn.Parameter(bias_rho)

        self.epsilon_normal = torch.distributions.Normal(0, 1)

        self.kl_divergence = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Calculates the forward pass through the linear layer.
        The local reparameterisation trick is used to estimate the
        gradients with respect to the parameters of a distribution - it
        takes advantage of the fact that, for a fixed input and Gaussian
        distributions over the weights, the resulting distribution over
        the activations is also Gaussian.
        Instead of sampling the weights individually and using them to
        compute a sample from the activation - we can sample from the
        distribution over activations. This yields a lower variance
        gradient estimator which makes training faster and more stable.
        Parameters
        ----------
        x : Tensor
            Inputs to the Bayesian Linear Layer.
        Returns
        -------
        Tensor
            Output from the Bayesian Linear Layer.
        """

        w_std = torch.log(1 + torch.exp(self.w_rho))
        b_std = torch.log(1 + torch.exp(self.bias_rho))

        act_mu = F.linear(x, self.w_mu)
        act_std = torch.sqrt(F.linear(x.pow(2), w_std.pow(2)))

        w_eps = self.epsilon_normal.sample(act_mu.size())
        bias_eps = self.epsilon_normal.sample(b_std.size())

        w_out = act_mu + act_std * w_eps
        b_out = self.bias_mu + b_std * bias_eps

        w_kl = self.kld(
            mu_prior=0.0,
            std_prior=self.std_prior,
            mu_posterior=self.w_mu,
            std_posterior=w_std
        )

        bias_kl = self.kld(
            mu_prior=0.0,
            std_prior=0.1,
            mu_posterior=self.bias_mu,
            std_posterior=b_std
        )

        self.kl_divergence = w_kl + bias_kl

        return w_out + b_out

    def kld(self,
            mu_prior: float,
            std_prior: float,
            mu_posterior: torch.Tensor,
            std_posterior: torch.Tensor) -> torch.Tensor:

        """Calculates the KL Divergence.
        The only 'downside' to the local reparameterisation trick is
        that, as the weights are not being sampled directly, the KL
        Divergence can not be calculated through the use of MC sampling.
        Instead, the closed form of the KL Divergence must be used;
        this restricts the prior and posterior to be Gaussian.
        However, the use of a Gaussian prior / posterior results in a
        lower variance and hence faster convergence.
        Parameters
        ----------
        mu_prior : float
            Mu of the prior normal distribution.
        std_prior : float
            Sigma of the prior normal distribution.
        mu_posterior : Tensor
            Mu to approximate the posterior normal distribution.
        std_posterior : Tensor
            Sigma to approximate the posterior normal distribution.
        Returns
        -------
        Tensor
            Calculated KL Divergence.
        """

        kl_divergence = 0.5 * (
                2 * torch.log(std_prior / std_posterior) -
                1 +
                (std_posterior / std_prior).pow(2) +
                ((mu_prior - mu_posterior) / std_prior).pow(2)
        ).sum()

        return kl_divergence

