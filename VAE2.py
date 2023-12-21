import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import torch.distributions as dists
from VAE import VAEModel


class VAEEncoder2(nn.Module):
    # encoder outputs the parameters of variational distribution "q"
    def __init__(self, input_dim, hidden_dim_1=200, latent_dim_1=100, hidden_dim_2=100, latent_dim_2=50):
        super(VAEEncoder2, self).__init__()

        self.FC_enc1  = nn.Linear(input_dim, hidden_dim_1) # FC stands for a fully connected layer
        self.FC_enc2  = nn.Linear(hidden_dim_1, hidden_dim_1)
        self.FC_mean1 = nn.Linear(hidden_dim_1, latent_dim_1)
        self.FC_var1  = nn.Linear (hidden_dim_1, latent_dim_1)

        self.FC_enc3  = nn.Linear(latent_dim_1, hidden_dim_2) # should we sample Z from var1 and mean1 as input? Yes!
        self.FC_enc4  = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.FC_mean2 = nn.Liner(hidden_dim_2, latent_dim_2)
        self.FC_var2  = nn.Linear(hidden_dim_2, latent_dim_2)

        self.activation = nn.Tanh() # will use this to add non-linearity to our model

        self.training = True

    def forward(self, x):
        h_1      = self.activation(self.FC_enc1(x))
        h_2      = self.activation(self.FC_enc2(h_1))
        mean1    = self.FC_mean(h_2)  # mean
        log_var1 = self.FC_var(h_2)   # log of variance

        z = self.reparameterization(mean1, log_var1.exp())

        h_3      = self.activation(self.FC_enc3(z))
        h_4      = self.activation(self.FC_enc3(h_3))
        mean2    = self.FC_mean(h_4)  # mean
        log_var2 = self.FC_var(h_4)   # log of variance

        return mean1, log_var1, mean2, log_var2
    
    def reparameterization(self, mean, var):
        with torch.no_grad():
            eps = torch.randn_like(mean)
        z = mean + var * eps
        return z
    

class VAEDecoder2(nn.Module):
    # decoder generates the success parameter of each pixel
    def __init__(self, input_dim, hidden_dim_1=200, latent_dim_1=100, hidden_dim_2=100, latent_dim_2=50):
        super(VAEDecoder2, self).__init__()
        self.FC_dec1   = nn.Linear(latent_dim_2, hidden_dim_2)
        self.FC_dec2   = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.FC_mean1  = nn.Linear(hidden_dim_2, latent_dim_1)
        self.FC_var1   = nn.Linear (hidden_dim_2, latent_dim_1)

        

        self.FC_dec3   = nn.Linear(latent_dim_1, hidden_dim_1)
        self.FC_dec4   = nn.Linear(hidden_dim_1, hidden_dim_1)
        self.FC_output = nn.Linear(hidden_dim_1, input_dim)

        self.activation = nn.Tanh() # again for non-linearity

    def forward(self, z):
        h_out_1  = self.activation(self.FC_dec1(z))
        h_out_2  = self.activation(self.FC_dec2(h_out_1))
        
        mean1    = self.FC_mean(h_out_2)  # mean
        log_var1 = self.FC_var(h_out_2)   # log of variance
        
        z = self.reparameterization(mean1, log_var1.exp())


        h_out_3      = self.activation(self.FC_dec3(z))
        h_out_4      = self.activation(self.FC_dec4(h_out_3))
        theta = torch.sigmoid(self.FC_output(h_out_4))
        return theta

class VAECoreModel2(nn.Module):
    def __init__(self, Encoder, Decoder, x_dim, k=10):
        super(VAECoreModel2, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

        self.k = k
        self.x_dim = x_dim

    def reparameterization(self, mean, var):
        with torch.no_grad():
            eps = torch.randn_like(mean)
        
        z = mean + var * eps
        return z


    def forward(self, x):
        mean1, log_var1, mean2, log_var2 = self.Encoder(x)

        #if k > 1:
        log_var_ki = log_var2.unsqueeze(1).repeat(1, self.k, 1) #torch.tile(log_var, (k,1))
        mean_ki = mean2.unsqueeze(1).repeat(1, self.k, 1) # torch.tile(mean, (k,1))

        z = self.reparameterization(mean_ki, torch.exp(log_var_ki))# takes exponential function (log var -> var)
        # z = [batch_size, k, x_dim]
        
        theta = self.Decoder(z)

        return theta, mean2, log_var2, z
    

class VAEModel2(VAEModel):
    def __init__(self, x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2, k=10) -> None:
        self.k = k
        self.x_dim = x_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2
        self.lr = 0
        
        self.encoder = VAEEncoder2(input_dim=x_dim, hidden_dim_1=hidden_dim_1, latent_dim_1=latent_dim_1, hidden_dim_2=hidden_dim_2, latent_dim_2=latent_dim_2)
        self.decoder = VAEDecoder2(latent_dim_1=latent_dim_1, latent_dim_2=latent_dim_2, hidden_dim_1 = hidden_dim_1, latent_dim_2=latent_dim_2, output_dim = x_dim)

        self.model = VAECoreModel2(Encoder=self.encoder, Decoder=self.decoder, x_dim=x_dim, k=k)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.model.to(self.device)
