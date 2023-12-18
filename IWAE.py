import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import torch.distributions as dists

from VAE import VAEModel


class IWAEModel(VAEModel):
    def __init__(self, x_dim, hidden_dim, latent_dim, k=10):
        super(IWAEModel, self).__init__(x_dim, hidden_dim, latent_dim, k=k)

    def loss_function(self, x, theta, mean, log_var, z):
        eps = 1e-10

        x_ki = x.unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        mu_z_ki = mean.unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        sigma_z_ki = torch.exp(0.5*log_var).unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        
        log_p = 1/self.k * torch.sum(x_ki * torch.log(theta + eps) + (1 - x_ki) * torch.log(1 - theta + eps))
        
        log_prior_z = dists.Normal(0, 1).log_prob(z).sum(2) # should this not be sum(1) ? 
        log_q_z_g_x = dists.Normal(mu_z_ki, sigma_z_ki).log_prob(z).sum(2)
        log_w = log_p + log_prior_z - log_q_z_g_x

        # copmute normalized importance weights (no gradient)
        log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
        w_tilde = log_w_tilde.exp().detach()
        # compute loss (negative IWAE objective)
        loss = -(w_tilde * log_w).sum(1).mean()

        NLL = -log_p
        
        active_units = torch.sum(torch.cov(z.sum(1)).sum(0)>10**-2) # sum over k

        return loss, NLL, active_units

