import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional
import math

torch.manual_seed(0)  # for reproducibility
hasGPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if hasGPU else "cpu")

#  Sets up the data loader arguments to use parallel processing on the GPU if available.
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if hasGPU else {}

GAUSSIAN_SCALER = 1. / np.sqrt(2.0 * np.pi)
def scale_mixture_prior(input, PI, SIGMA_1, SIGMA_2):
    prob1 = PI * gaussian(input, 0., SIGMA_1)
    prob2 = (1. - PI) * gaussian(input, 0., SIGMA_2)
    return torch.log(prob1 + prob2)

# Defines a Gaussian function with mean mu and standard deviation sigma and evaluates it at x.
def gaussian(x, mu, sigma):
    # Computes the exponential function element-wise.
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    # Clamps the input tensor to the range [min, max].
    return torch.clamp(GAUSSIAN_SCALER / sigma * bell, 1e-10, 1.)  # clip to avoid numerical issues

class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()



class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, parent):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.SIGMA_1 = parent.SIGMA_1
        self.SIGMA_2 = parent.SIGMA_2
        self.PI = parent.PI


        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0., .05))  #(0., .1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5., .05)) #(-5., .05))  (-3., -3.))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0., .05)) #(0., .05)) (0., .1))  
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5., .05)) #(-5., .05))(-3., -3.))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(self.PI, self.SIGMA_1, self.SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(self.PI, self.SIGMA_1, self.SIGMA_2)
        
        self.log_prior = 0 #lpw
        self.log_variational_posterior = 0 #lqw

    def forward(self, input, sample=False, calculate_log_probs=False):
        
        weight_sigma = torch.log(1. + torch.exp(self.weight_rho))
        bias_sigma = torch.log(1. + torch.exp(self.bias_rho))
        epsilon_weight = Variable(torch.Tensor(self.out_features, self.in_features).normal_(0., 1.)).to(DEVICE)
        epsilon_bias = Variable(torch.Tensor(self.out_features).normal_(0., 1.)).to(DEVICE)
        
        if self.training or sample:
            #print("right")
            weight = self.weight_mu + weight_sigma * epsilon_weight
            bias = self.bias_mu + bias_sigma * epsilon_bias
            #weight = self.weight.sample()
            #bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        

        if self.training or calculate_log_probs:
            #self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_prior = scale_mixture_prior(weight, self.PI, self.SIGMA_1, self.SIGMA_2).sum() + scale_mixture_prior(
                bias, self.PI, self.SIGMA_1, self.SIGMA_2).sum()
            
            #self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
            self.log_variational_posterior = torch.log(gaussian(weight, torch.squeeze(self.weight_mu), torch.squeeze(weight_sigma))).sum() + torch.log(
                    gaussian(bias, self.bias_mu, bias_sigma)).sum()
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)



class BayesianNetwork(nn.Module):
    def __init__(self, inputSize, CLASSES, layers, activations, SAMPLES, BATCH_SIZE, NUM_BATCHES, hasScalarMixturePrior,
                 PI, SIGMA_1, SIGMA_2):
        super().__init__()

        self.inputSize = inputSize
        self.activations = activations
        self.CLASSES = CLASSES
        self.SAMPLES = SAMPLES
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_BATCHES = NUM_BATCHES
        self.DEPTH = 0  # captures depth of network

        assert (activations.size - layers.size) == 1

        self.SIGMA_1 = SIGMA_1
        self.hasScalarMixturePrior = hasScalarMixturePrior
        if hasScalarMixturePrior == True:
            self.SIGMA_2 = SIGMA_2
            self.PI = PI

        if layers.size == 0:
            self.l1 = BayesianLinear(inputSize, CLASSES, self)
            self.DEPTH += 1
        else:
            self.l1= BayesianLinear(inputSize, layers[0], self)
            self.DEPTH += 1
            self.l2= BayesianLinear(layers[0], layers[1], self)
            self.DEPTH += 1
            self.l3= BayesianLinear(layers[1], layers[2], self)
            self.DEPTH += 1
            self.l4= BayesianLinear(layers[2], CLASSES, self)
            self.DEPTH += 1
    

    def forward(self, x, infer=False):
        x = x.view(-1, self.inputSize)
        x = F.relu(self.l1(x, infer))
        x = F.relu(self.l2(x, infer))
        x = F.relu(self.l3(x, infer))

        # when performing regression
        x = self.l4(x, infer)

        return x

    
    def log_prior(self):
        lpw = self.l1.log_prior + self.l2.log_prior + self.l3.log_prior + self.l4.log_prior
        return lpw
    
    def log_variational_posterior(self):
        lqw = self.l1.log_variational_posterior + self.l2.log_variational_posterior + self.l3.log_variational_posterior + self.l4.log_variational_posterior
        return lqw
    
    def BBB_loss(self, input, target, batch_idx = None):
        outputs = torch.zeros(self.SAMPLES, self.BATCH_SIZE, self.CLASSES)
        log_priors = torch.zeros(self.SAMPLES)
        log_variational_posteriors = torch.zeros(self.SAMPLES)
        log_likelihoods = torch.zeros(self.SAMPLES)
        log_prior, log_variational_posterior, log_likelihood =  0., 0., 0.

        for i in range(self.SAMPLES):
            outputs = self.forward(input)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
            log_likelihoods[i] = -(.5 * (target - torch.squeeze(outputs)) ** 2).sum()
            

        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        log_likelihood = log_likelihoods.mean()
        
        #print(log_likelihood)
        #print("============================================")
        loss = (1. / (self.NUM_BATCHES)) * (log_variational_posterior - log_prior) - log_likelihood
        return loss

