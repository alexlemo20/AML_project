import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import torch.distributions as dists

"""
TODO: Check the k implementation
      Check how the weights are calculated
      Maybe optimize our code to get our results quicker (e.g. replace .item)
      
      Do torch.no_grad() as much as possible
      Every function with a device kwarg, pass self.device in order to init the var on the current device used

"""


class VAEEncoder(nn.Module):
    # encoder outputs the parameters of variational distribution "q"
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAEEncoder, self).__init__()

        self.FC_enc1 = nn.Linear(input_dim, hidden_dim) # FC stands for a fully connected layer
        self.FC_enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var  = nn.Linear (hidden_dim, latent_dim)

        self.activation = nn.Tanh() # will use this to add non-linearity to our model

        self.training = True

    def forward(self, x):
        h_1     = self.activation(self.FC_enc1(x))
        h_2     = self.activation(self.FC_enc2(h_1))
        mean    = self.FC_mean(h_2)  # mean
        log_var = self.FC_var(h_2)   # log of variance

        return mean, log_var
    

class VAEDecoder(nn.Module):
    # decoder generates the success parameter of each pixel
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.FC_dec1   = nn.Linear(latent_dim, hidden_dim)
        self.FC_dec2   = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.Tanh() # again for non-linearity

    def forward(self, z):
        h_out_1  = self.activation(self.FC_dec1(z))
        h_out_2  = self.activation(self.FC_dec2(h_out_1))

        theta = torch.sigmoid(self.FC_output(h_out_2))
        return theta

class VAECoreModel(nn.Module):
    def __init__(self, Encoder, Decoder, x_dim, device, k=10):
        super(VAECoreModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.device = device

        self.k = k
        self.x_dim = x_dim

    def reparameterization(self, mean, var):
        with torch.no_grad():
            eps = torch.randn_like(var, device=self.device)
        z = mean + var * eps
        return z


    def forward(self, x):
        mean, log_var = self.Encoder(x)

        #if k > 1:
        log_var_ki = log_var.unsqueeze(1).repeat(1, self.k, 1) #torch.tile(log_var, (k,1))
        mean_ki = mean.unsqueeze(1).repeat(1, self.k, 1) # torch.tile(mean, (k,1))

        z = self.reparameterization(mean_ki, torch.exp(0.5*log_var_ki))# takes exponential function (log var -> var)
        # z = [batch_size, k, x_dim]
        
        theta = self.Decoder(z)

        return theta, mean, log_var, z
    

class VAEModel():
    def __init__(self, x_dim, hidden_dim, latent_dim, k=10, device=None) -> None:
        self.k = k
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = 0
        
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.encoder = VAEEncoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

        self.model = VAECoreModel(Encoder=self.encoder, Decoder=self.decoder, x_dim=x_dim, k=k, device=self.device)

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.model.to(self.device)

    def calculate_lr(self, i, i_max):
        self.lr = 0.001 * 10 ** (-i/i_max)
        
    def train(self, train_loader, i_max, batch_size = 20):
        self.model.train()

        total_epochs = np.sum([3**i for i in range(i_max+1)])
        loss_epochs = torch.zeros(total_epochs)
        active_units = torch.zeros(total_epochs)        
        
        epoch_counter = 0 

        with tqdm(total=total_epochs) as pbar:
            for i in range(i_max + 1):

                self.calculate_lr(i, i_max) # update the learning rate 
                optimizer = Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999),eps=0.0001)

                for j in range(3**i):                    
                    overall_loss = 0
                    overall_active_units = 0
                    for batch_idx, (x, _) in enumerate(train_loader):
                        x = x.view(batch_size, self.x_dim)

                        x = x.to(self.device)
                        optimizer.zero_grad(set_to_none=True)

                        theta, mean, log_var, z = self.model(x)
                        loss, au = self.loss_function(x, theta, mean, log_var, z )
                        

                        overall_loss += loss.item()
                        overall_active_units += au.item()
                        
                        loss.backward()
                        optimizer.step()

                    loss_epochs[epoch_counter] = overall_loss/(len(train_loader)) # (len(data) - 1 * batch_Size)
                    active_units[epoch_counter] = overall_active_units/(len(train_loader))
                    print("\tEpoch", epoch_counter + 1, "\tAverage Loss: ",  overall_loss / (len(train_loader)), "\tAverage AU: ", overall_active_units/(len(train_loader)))
                    epoch_counter += 1
                    pbar.update(1)

        return loss_epochs

    def loss_function(self, x, theta, mean, log_var, z):
        mu_square = mean.mul(mean).to(self.device)
        D_KL = 0.5 * torch.sum(mu_square+torch.exp(log_var)-1-log_var, axis=1).mean().to(self.device)


        x_ki = x.unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        log_p =  dists.Bernoulli(theta).log_prob(x_ki).sum(axis=2).mean()
        
        elbo = log_p - D_KL
        loss = -elbo

        # z.shape = [batch_size, k, latent_dim]
        active_units = torch.sum(torch.cov(z.sum(1)).sum(0)>10**-2) # sum over k


        # Calculate activation probabilities
        #activation_probs = torch.sigmoid(mean)
        # Calculate covariance term
        #cov_x_exp_u = torch.matmul(activation_probs.t(), activation_probs)
        # Compare the covariance term to a threshold (e.g., 1e-2)
        #threshold = 1e-2
        #num_active_units = (cov_x_exp_u > threshold).sum().item()
        #print("NUM OF ACTIVE UNITS: ", num_active_units)

        
        return loss, active_units


    def compute_evaluation_loss(self, x, k):
        # do stuff
        NLL = 0
        old_k = self.k
        # Set k to the new k for evaluation
        self.k = k
        self.model.k = k

        theta, mean, log_var, z = self.model(x)


        x_ki = x.unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        mu_z_ki = mean.unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        sigma_z_ki = torch.exp(0.5*log_var).unsqueeze(1).repeat(1, self.k, 1).to(self.device)        

        log_p =  dists.Bernoulli(theta).log_prob(x_ki).sum(axis=2)

        log_prior_z = dists.Normal(0, 1).log_prob(z).sum(axis=2) # should this not be sum(1) ? 
        log_q_z_g_x = dists.Normal(mu_z_ki, sigma_z_ki).log_prob(z).sum(axis=2)
        log_w = log_p + log_prior_z - log_q_z_g_x # shape: [batch_size, k]
        
        NLL = -(torch.logsumexp(log_w, 1) -  np.log(k)).mean()

        # Reset k
        self.k = old_k
        self.model.k = old_k

        return NLL


    def evaluate(self, test_loader, batch_size, k=5000):
        self.model.eval()
    
        total_samples = len(test_loader)
        total_NLL = 0

        # below we get decoder outputs for test data
        with tqdm(total=total_samples) as pbar:
            with torch.no_grad():
                for batch_idx, (x, _) in enumerate(test_loader):
                    x = x.view(batch_size, self.x_dim).to(self.device)

                    NLL = self.compute_evaluation_loss(x, k)

                    total_NLL += NLL.item()
                    pbar.update(1)

        avg_NLL = total_NLL / total_samples

        return avg_NLL


