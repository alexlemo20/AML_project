import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm


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
    def __init__(self, Encoder, Decoder, x_dim, k=10):
        super(VAECoreModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

        self.k = k
        self.x_dim = x_dim

    def reparameterization(self, mean, var):
       
        z = mean + var * torch.randn_like(mean)
        return z


    def forward(self, x):
        mean, log_var = self.Encoder(x)

        #if k > 1:
        log_var_ki = log_var.unsqueeze(1).repeat(1, self.k, 1) #torch.tile(log_var, (k,1))
        mean_ki = mean.unsqueeze(1).repeat(1, self.k, 1) # torch.tile(mean, (k,1))

        z = self.reparameterization(mean_ki, torch.exp(log_var_ki)) # takes exponential function (log var -> var)

        theta = self.Decoder(z)

        return theta, mean, log_var
    

class VAEModel():
    def __init__(self, x_dim, hidden_dim, latent_dim, k=10) -> None:
        """
        learning_rate_funcion := calculate learning rate each epoch
        """
        self.k = k
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = 0
        
        self.encoder = VAEEncoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

        self.model = VAECoreModel(Encoder=self.encoder, Decoder=self.decoder, x_dim=x_dim, k=k)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.model.to(self.device)

    def calculate_lr(self, i):
        self.lr = 0.001 * 10 ** (-i/7)
        
    def train(self, train_loader, i_max, batch_size = 20):
        self.model.train()

        loss_epochs = torch.zeros(np.sum([3**i for i in range(i_max+1)]))        
        
        epoch_counter = 0 

        for i in range(i_max+1):

            self.calculate_lr(i) # update the learning rate 

            for j in tqdm(range(3**i)):
                #for epoch in range(epochs):
                optimizer = Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999),eps=0.0001)
                
                overall_loss = 0
                for batch_idx, (x, _) in enumerate(train_loader):
                    
                    x = x.view(batch_size, self.x_dim)

                    optimizer.zero_grad()

                    theta, mean, log_var = self.model(x)
                    loss = self.loss_function(x, theta, mean, log_var, batch_size)

                    overall_loss += loss.item()
                    
                    loss.backward()
                    optimizer.step()
                    #exit(1)
                
                loss_epochs[epoch_counter] = overall_loss
                print("\tEpoch", epoch_counter + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
                epoch_counter += 1

        return overall_loss, loss_epochs

    def loss_function(self, x, theta, mean, log_var, batch_size):
        eps = 1e-8
        mu_square = mean.mul(mean)
        sigma_square = torch.exp(log_var)
        log_sigma_square = log_var
        ones = torch.ones([batch_size, self.latent_dim])
        D_KL = torch.sum((1/2)*(mu_square+sigma_square-ones-log_sigma_square))

        #print("X: ", x.shape) # 20 x 784
        #print("theta : ",theta.shape)
        #print("2nd term: ", torch.log(theta + eps).shape) # 20 x 10 x 784
        x_ki = x.unsqueeze(1).repeat(1, self.k, 1)
        #print("x_ki.shape : ",x_ki.shape)
        log_p = 1/self.k * torch.sum(x_ki * torch.log(theta + eps) + (1 - x_ki) * torch.log(1 - theta + eps))

        elbo = log_p - D_KL

        return -elbo
    
    def loss_function2(x, theta, mean, log_var, batch_size):
        e_log_p = torch.sum(torch.log(x*theta + (1-x)*(1-theta))) 
        kl = 0.5 * torch.sum(-log_var + log_var.exp() + mean.pow(2) - 1)

        elbo = e_log_p - kl

        return -elbo

    def evaluate(self, test_loader, batch_size):
        self.model.eval()
    
        total_samples = len(test_loader.dataset)
        total_loss = 0

        # below we get decoder outputs for test data
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(test_loader):
                x = x.view(batch_size, self.x_dim)
                #x = torch.round(x)

                # insert your code below to generate theta from x
                theta, mean, log_var = self.model(x)    
                loss = self.loss_function(x, theta, mean, log_var)

                total_loss += loss.item()
                

        avg_loss = total_loss / total_samples
        
        return avg_loss


