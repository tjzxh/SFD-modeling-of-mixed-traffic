import torch
import torch.nn as nn
import numpy as np


class MDN(nn.Module):
    def __init__(self, input_dim, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Sequential(
            nn.Linear(n_hidden, n_gaussians),
            nn.ReLU()
        )

    def forward(self, x):
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu


oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0 * np.pi)  # normalization factor for Gaussians
min_speed, max_speed = 0.0, 35.0  # m/s
min_gap, max_gap = 0.0, 100.0  # m


def gaussian_distribution(y, mu, sigma):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI


def truncated_gaussian_distribution(y, mu, sigma):
    min_speed_tensor, max_speed_tensor = torch.tensor(min_gap), torch.tensor(max_gap)
    min_speed_norm = (min_speed_tensor.expand_as(mu) - mu) * torch.reciprocal(sigma)
    max_speed_norm = (max_speed_tensor.expand_as(mu) - mu) * torch.reciprocal(sigma)
    phi_min = 0.5 * (1 + torch.erf(min_speed_norm / np.sqrt(2)))
    phi_max = 0.5 * (1 + torch.erf(max_speed_norm / np.sqrt(2)))
    prob_density = gaussian_distribution(y, mu, sigma) / (phi_max - phi_min)
    return prob_density


# compute the cdf of the Truncated Gaussian distribution between y-0.5 and y+0.5 (m/s)
def truncated_gaussian_cdf(y, mu, sigma, pi):
    interval = 2.5  # m
    min_speed_tensor, max_speed_tensor = torch.tensor(min_gap), torch.tensor(max_gap)
    yr_norm = ((y + interval).expand_as(mu) - mu) * torch.reciprocal(sigma)
    yl_norm = ((y - interval).expand_as(mu) - mu) * torch.reciprocal(sigma)
    min_speed_norm = (min_speed_tensor.expand_as(mu) - mu) * torch.reciprocal(sigma)
    max_speed_norm = (max_speed_tensor.expand_as(mu) - mu) * torch.reciprocal(sigma)
    phi_yr = 0.5 * (1 + torch.erf(yr_norm / np.sqrt(2)))
    phi_yl = 0.5 * (1 + torch.erf(yl_norm / np.sqrt(2)))
    phi_min = 0.5 * (1 + torch.erf(min_speed_norm / np.sqrt(2)))
    phi_max = 0.5 * (1 + torch.erf(max_speed_norm / np.sqrt(2)))
    yr_prob = (phi_yr - phi_min) / (phi_max - phi_min)
    yl_prob = (phi_yl - phi_min) / (phi_max - phi_min)
    prob = yr_prob - yl_prob
    weighted_prob = torch.sum(prob * pi, dim=1)
    return weighted_prob


def mdn_loss_fn(pi, sigma, mu, y):
    result = truncated_gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=1)
    eps = 1e-10
    result = -torch.log(result + eps)
    # debug NaN loss
    if torch.isnan(result).any():
        print('NaN loss')
    return torch.mean(result)
