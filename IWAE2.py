import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import torch.distributions as dists

from VAE2 import VAEModel2


class IWAEModel2(VAEModel2):
    def __init__(self, x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2, k=10):
        super(IWAEModel2, self).__init__(x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2, k=k)

    def loss_function(self, x, theta, mean, log_var, z):
        x_ki = x.unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        mu_z_ki = mean.unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        sigma_z_ki = torch.exp(0.5*log_var).unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        
        log_p = dists.Bernoulli(theta).log_prob(x_ki).sum(axis=2).mean() # 1/self.k * 
        
        log_prior_z = dists.Normal(0, 1).log_prob(z).sum(2) # should this not be sum(1) ? 
        log_q_z_g_x = dists.Normal(mu_z_ki, sigma_z_ki).log_prob(z).sum(2)
        log_w = log_p + log_prior_z - log_q_z_g_x

        # copmute normalized importance weights (no gradient)
        log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
        w_tilde = log_w_tilde.exp().detach()
        # compute loss (negative IWAE objective)
        loss = -(w_tilde * log_w).sum(1).mean()
        
        active_units = torch.sum(torch.cov(z.sum(1)).sum(0)>10**-2) # sum over k

        return loss, active_units

