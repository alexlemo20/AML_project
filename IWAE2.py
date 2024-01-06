import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import torch.distributions as dists

from VAE2 import VAEModel2


class IWAEModel2(VAEModel2):
    def __init__(self, x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2, k=10, compile_model=True):
        super(IWAEModel2, self).__init__(x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2, k=k, compile_model=compile_model)
        self.log_k = np.log(self.k)

    def loss_function(self, x, theta, mean1, log_var1, z1, mean2, log_var2, z2, z_d, mean_d, log_var_d):
        x_ki = x.unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        pxz1 = dists.Bernoulli(theta)

        lpxz1 = torch.sum(pxz1.log_prob(x_ki), dim=-1)

        pz1z2 = dists.Normal(mean_d, (0.5*log_var_d).exp()) 
        lpz1z2 = torch.sum(pz1z2.log_prob(z1), dim=-1) 
        
        pz2 = dists.Normal(0, 1) # Between encode / decoder
        lpz2 = torch.sum(pz2.log_prob(z2), dim=-1)
        
        mean1 = mean1.unsqueeze(1).repeat(1, self.k, 1).to(self.device) # CHECK
        log_var1 = log_var1.unsqueeze(1).repeat(1, self.k, 1).to(self.device) # CHECK

        qz1x = dists.Normal(mean1, (0.5*log_var1).exp())
        # z1 = qz1x.sample(k) # we may need to sample and then update mean2
        lqz1x = torch.sum(qz1x.log_prob(z1), dim=-1)


        qz2z1 = dists.Normal(mean2, (0.5*log_var2).exp())
        lqz2z1 = torch.sum(qz2z1.log_prob(z2), dim=-1)
        
        log_w = lpxz1 + lpz1z2 + lpz2 - lqz1x - lqz2z1
        
        # loss = - torch.mean(log_w) # CHECK: tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)

        # copmute normalized importance weights (no gradient)
        #log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True) # check axis
        #w_tilde = log_w_tilde.exp().detach()
        # compute loss (negative IWAE objective)
        #loss = -(w_tilde * log_w).sum(1).mean(0) # check axis

        loss = (-torch.logsumexp(log_w, dim=1) + self.log_k).mean(0) # - self.log_k too?

        #with torch.no_grad():
        #    active_units = np.array([torch.sum(torch.cov(z1.sum(1)).sum(0)>10**-2).item(), # sum over k
        #                    torch.sum(torch.cov(z2.sum(1).sum(1)).sum(0)>10**-2).item()]) # sum over k
        active_units = 0
        return loss, active_units

