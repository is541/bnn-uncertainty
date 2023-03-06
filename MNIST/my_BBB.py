
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

torch.manual_seed(0)


writer = SummaryWriter()
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device", DEVICE)
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.backends.mps.is_available() else {}

PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(DEVICE)
SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(DEVICE)
BATCH_SIZE = 100
TEST_BATCH_SIZE = 5

def _my_normalization(x):
    return x * 255. / 126.



CLASSES = 10
TRAIN_EPOCHS = 20
SAMPLES = 2
TEST_SAMPLES = 10



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
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


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

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample().to(DEVICE)
            bias = self.bias.sample().to(DEVICE)
        else:
            weight = self.weight.mu.to(DEVICE)
            bias = self.bias.mu.to(DEVICE)
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)

class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BayesianLinear(28*28, 400).to(DEVICE)
        self.l2 = BayesianLinear(400, 400).to(DEVICE)
        self.l3 = BayesianLinear(400, 10).to(DEVICE)
    
    def forward(self, x, sample=False):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.log_softmax(self.l3(x, sample), dim=1)
        return x
    
    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior
    
    def sample_elbo(self, input, target, samples=SAMPLES):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior)/NUM_BATCHES + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood



def write_weight_histograms(epoch):
    writer.add_histogram('histogram/w1_mu', net.l1.weight_mu,epoch)
    writer.add_histogram('histogram/w1_rho', net.l1.weight_rho,epoch)
    writer.add_histogram('histogram/w2_mu', net.l2.weight_mu,epoch)
    writer.add_histogram('histogram/w2_rho', net.l2.weight_rho,epoch)
    writer.add_histogram('histogram/w3_mu', net.l3.weight_mu,epoch)
    writer.add_histogram('histogram/w3_rho', net.l3.weight_rho,epoch)
    writer.add_histogram('histogram/b1_mu', net.l1.bias_mu,epoch)
    writer.add_histogram('histogram/b1_rho', net.l1.bias_rho,epoch)
    writer.add_histogram('histogram/b2_mu', net.l2.bias_mu,epoch)
    writer.add_histogram('histogram/b2_rho', net.l2.bias_rho,epoch)
    writer.add_histogram('histogram/b3_mu', net.l3.bias_mu,epoch)
    writer.add_histogram('histogram/b3_rho', net.l3.bias_rho,epoch)

def write_loss_scalars(epoch, batch_idx, loss, log_prior, log_variational_posterior, negative_log_likelihood):
    writer.add_scalar('logs/loss', loss, epoch*NUM_BATCHES+batch_idx)
    writer.add_scalar('logs/complexity_cost', log_variational_posterior-log_prior, epoch*NUM_BATCHES+batch_idx)
    writer.add_scalar('logs/log_prior', log_prior, epoch*NUM_BATCHES+batch_idx)
    writer.add_scalar('logs/log_variational_posterior', log_variational_posterior, epoch*NUM_BATCHES+batch_idx)
    writer.add_scalar('logs/negative_log_likelihood', negative_log_likelihood, epoch*NUM_BATCHES+batch_idx)

def train(net, optimizer, epoch):
    net.train()
    #if epoch == 0: # write initial distributions
        #write_weight_histograms(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
        loss.backward()
        optimizer.step()
        #write_loss_scalars(epoch, batch_idx, loss, log_prior, log_variational_posterior, negative_log_likelihood)
    return loss
    
def evaluate(model, loader, infer=True, samples=1):
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        if samples == 1:
            output = model(data, infer=infer).to(DEVICE)
        else:
            output = model(data)
            for i in range(samples - 1):
                output += model(data)

        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).cpu().sum().item()
        acc_sum += acc
    return acc_sum / len(loader)



if __name__ == '__main__':
    net = BayesianNetwork().to(DEVICE)

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(_my_normalization),  # divide as in paper
        ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
        'data', train=True, download=True,
        transform=transform), 
        batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS)

    print("Train data loader")
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
        'data', train=False, download=True,
        transform=transform),
        batch_size=TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)

    print("Test data loader")
    
    TRAIN_SIZE = len(train_loader.dataset)
    TEST_SIZE = len(test_loader.dataset)
    NUM_BATCHES = len(train_loader)
    NUM_TEST_BATCHES = len(test_loader)
    assert (TRAIN_SIZE % BATCH_SIZE) == 0
    assert (TEST_SIZE % TEST_BATCH_SIZE) == 0

    print("train size", TRAIN_SIZE)
    print("test size", TEST_SIZE)

    test_accs = np.zeros(TRAIN_EPOCHS)
    train_losses = np.zeros(TRAIN_EPOCHS)

    optimizer = optim.Adam(net.parameters())
    for epoch in range(TRAIN_EPOCHS):

        print("Begin trainin epoch ", epoch)
        train_loss = train(net, optimizer, epoch)
        test_acc = evaluate(net, test_loader, samples=TEST_SAMPLES)

        print('Epoch', epoch + 1, 'Loss', float(train_loss),
            'Test Error',  round(100 * (1 - test_acc / NUM_TEST_BATCHES), 3), '%')

        test_accs[epoch] = test_acc
        train_losses[epoch] = train_loss


    path = 'Results/BBB_MNIST' + '_' + str(400) + '_Adam_samples_ID' + str(2)
    wr = csv.writer(open(path + '.csv', 'w'), delimiter=',', lineterminator='\n')
    wr.writerow(['epoch','test_acc', 'train_losses'])

    for i in range(TRAIN_EPOCHS):
        wr.writerow((i + 1, 1 - test_accs[i] / TEST_BATCH_SIZE, train_losses[i]))

    torch.save(net.state_dict(), path + '.pth')