import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import torch.distributions as dists


class VAEEncoder2(nn.Module):
    # encoder outputs the parameters of variational distribution "q"
    def __init__(self, input_dim, hidden_dim_1=200, latent_dim_1=100, hidden_dim_2=100, latent_dim_2=50, k=1):
        super(VAEEncoder2, self).__init__()

        self.FC_enc1  = nn.Linear(input_dim, hidden_dim_1) # FC stands for a fully connected layer
        self.FC_enc2  = nn.Linear(hidden_dim_1, hidden_dim_1)
        self.FC_mean1 = nn.Linear(hidden_dim_1, latent_dim_1)
        self.FC_var1  = nn.Linear (hidden_dim_1, latent_dim_1)

        self.FC_enc3  = nn.Linear(latent_dim_1, hidden_dim_2) # should we sample Z from var1 and mean1 as input? Yes!
        self.FC_enc4  = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.FC_mean2 = nn.Linear(hidden_dim_2, latent_dim_2)
        self.FC_var2  = nn.Linear(hidden_dim_2, latent_dim_2)

        self.k = k

        self.activation = nn.Tanh() # will use this to add non-linearity to our model

        self.training = True

    def forward(self, x):
        h_1      = self.activation(self.FC_enc1(x))
        h_2      = self.activation(self.FC_enc2(h_1))
        mean1    = self.FC_mean1(h_2)  # mean
        log_var1 = self.FC_var1(h_2)   # log of variance


        #if k > 1:
        log_var_ki = log_var1.unsqueeze(1).repeat(1, self.k, 1) #torch.tile(log_var, (k,1))
        mean_ki = mean1.unsqueeze(1).repeat(1, self.k, 1) # torch.tile(mean, (k,1))

        z1 = self.reparameterization(mean_ki, torch.exp(0.5*log_var_ki)) # 0.5*

        h_3      = self.activation(self.FC_enc3(z1))
        h_4      = self.activation(self.FC_enc3(h_3))
        mean2    = self.FC_mean2(h_4)  # mean
        log_var2 = self.FC_var2(h_4)   # log of variance
        # mean1, log_var1, z1, mean2, log_var2 
        
        return mean1, log_var1, z1, mean2, log_var2
    
    def reparameterization(self, mean, var):
        with torch.no_grad():
            eps = torch.randn_like(mean)
        z = mean + var * eps
        return z
    

class VAEDecoder2(nn.Module):
    # decoder generates the success parameter of each pixel
    def __init__(self, output_dim, hidden_dim_1=200, latent_dim_1=100, hidden_dim_2=100, latent_dim_2=50):
        super(VAEDecoder2, self).__init__()
        self.FC_dec1   = nn.Linear(latent_dim_2, hidden_dim_2)
        self.FC_dec2   = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.FC_mean1  = nn.Linear(hidden_dim_2, latent_dim_1)
        self.FC_var1   = nn.Linear (hidden_dim_2, latent_dim_1)

        self.FC_dec3   = nn.Linear(latent_dim_1, hidden_dim_1)
        self.FC_dec4   = nn.Linear(hidden_dim_1, hidden_dim_1)
        self.FC_output = nn.Linear(hidden_dim_1, output_dim)

        self.activation = nn.Tanh() # again for non-linearity

    def forward(self, z1, z2):
        h_out_1  = self.activation(self.FC_dec1(z2))
        h_out_2  = self.activation(self.FC_dec2(h_out_1))
        
        mean1    = self.FC_mean1(h_out_2)  # mean
        log_var1 = self.FC_var1(h_out_2)   # log of variance
        
        zd = self.reparameterization(mean1, torch.exp(0.5*log_var1))

        h_out_3      = self.activation(self.FC_dec3(z1))
        h_out_4      = self.activation(self.FC_dec4(h_out_3))
        theta = torch.sigmoid(self.FC_output(h_out_4))
        return theta, zd, mean1, log_var1
    
    def reparameterization(self, mean, var):
        with torch.no_grad():
            eps = torch.randn_like(mean)
        z = mean + var * eps
        return z

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
        mean1, log_var1, z1, mean2, log_var2 = self.Encoder(x)

        log_var2_kki = log_var2.unsqueeze(1).repeat(1, self.k, 1, 1)
        mean2_kki = mean2.unsqueeze(1).repeat(1, self.k, 1, 1)

        z2 = self.reparameterization(mean2_kki, torch.exp(0.5*log_var2_kki))# takes exponential function (log var -> var)

        theta, z_d, mean_d, log_var_d = self.Decoder(z1,z2)

        return theta, mean1, log_var1, z1, mean2, log_var2, z2, z_d, mean_d, log_var_d


class VAEModel2():
    def __init__(self, x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2, k=10) -> None:
        self.k = k
        self.x_dim = x_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2
        self.lr = 0
        
        self.encoder = VAEEncoder2(x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2, k)
        self.decoder = VAEDecoder2(x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2)

        self.model = VAECoreModel2(Encoder=self.encoder, Decoder=self.decoder, x_dim=x_dim, k=k)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.model.to(self.device)
        #Self.model = torch.compile(self.model, mode="reduce-overhead")

    def calculate_lr(self, i, i_max):
        self.lr = 0.001 * 10 ** (-i/i_max)
    def train(self, train_loader, i_max, batch_size = 20):
        self.model.train()

        total_epochs = np.sum([3**i for i in range(i_max+1)])
        loss_epochs = torch.zeros(total_epochs)
        active_units = [0,] * total_epochs
        
        epoch_counter = 0 

        with tqdm(total=total_epochs) as pbar:
            for i in range(i_max + 1):

                self.calculate_lr(i, i_max) # update the learning rate 
                optimizer = Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999),eps=0.0001)

                for j in range(3**i):
                    overall_loss = 0
                    overall_active_units = np.array([0,0])
                    for batch_idx, (x, y) in enumerate(train_loader):

                        x = x.view(batch_size, self.x_dim)

                        x = x.to(self.device)
                        optimizer.zero_grad(set_to_none=True)

                        theta, mean1, log_var1, z1, mean2, log_var2, z2, z_d, mean_d, log_var_d = self.model(x)
                        loss,  au = self.loss_function1(x, theta, mean1, log_var1, z1, mean2, log_var2, z2, z_d, mean_d, log_var_d)

                        overall_loss += loss.item()
                        
                        overall_active_units += au
                        
                        loss.backward()
                        optimizer.step()

                    loss_epochs[epoch_counter] = overall_loss/(len(train_loader)) # (len(data) - 1 * batch_Size)
                    active_units[epoch_counter] = overall_active_units/(len(train_loader))
                    print("\tEpoch", epoch_counter + 1, "\tAverage Loss: ",  overall_loss / (len(train_loader)), "\tAverage AU: ", overall_active_units/(len(train_loader)))
                    epoch_counter += 1
                    pbar.update(1)

        return loss_epochs

    def loss_function1(self, x, theta, mean1, log_var1, z1, mean2, log_var2, z2, z_d, mean_d, log_var_d):

        #z1, qz1x, z2, qz2z1 = self.encoder(x, n_samples)
        #pz1z2 = self.decode_z2_to_z1(z2) # tfd.Normal(self.lmu(h2), self.lstd(h2) + 1e-6)
        #logits = self.decode_z1_to_x(z1)
        #pxz1 = tfd.Bernoulli(logits=logits)
        ###
        #logits, pxz1, pz1z2 = self.decoder(z1, z2)

        # ---- loss
        x_ki = x.unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        pxz1 = dists.Bernoulli(theta)

        lpxz1 = torch.sum(pxz1.log_prob(x_ki), dim=-1)


        z1k = z1.unsqueeze(1).repeat(1, self.k,1, 1)
        #print("z1k : ",z1k.shape)
        pz1z2 = dists.Normal(mean_d, (0.5*log_var_d).exp())
        lpz1z2 = torch.sum(pz1z2.log_prob(z1k), dim=-1)
        
        pz2 = dists.Normal(0, 1) # Between encode / decoder
        lpz2 = torch.sum(pz2.log_prob(z2), dim=-1)
        
        mean1 = mean1.unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        log_var1 = log_var1.unsqueeze(1).repeat(1, self.k, 1).to(self.device)

        qz1x = dists.Normal(mean1, (0.5*log_var1).exp())
        # z1 = qz1x.sample(k) # we may need to sample and then update mean2
        lqz1x = torch.sum(qz1x.log_prob(z1), dim=-1)

        mean2 = mean2.unsqueeze(1).repeat(1,self.k,  1,  1).to(self.device)
        log_var2 = log_var2.unsqueeze(1).repeat(1,self.k, 1, 1).to(self.device)
        #print("\nmean2 : ",mean2.shape)
        qz2z1 = dists.Normal(mean2, (0.5*log_var2).exp())
        lqz2z1 = torch.sum(qz2z1.log_prob(z2), dim=-1)
        
        #[batch_size, k, k]
        lpz1z2 = lpz1z2.mean(2)
        lpz2 = lpz2.mean(2)
        lqz2z1 = lqz2z1.mean(2)

        log_w = lpxz1 + lpz1z2 + lpz2 - lqz1x - lqz2z1


        #print(log_w.shape)

        # ---- regular VAE elbo
        # mean over samples and batch

        vae_elbo =  (torch.logsumexp(log_w, 1) -  np.log(self.k)).mean() # CHECK: tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)

        #NLL = -(torch.logsumexp(log_w, 1) -  np.log(self.k)).mean()

        #print("z1: ", z1.shape, "\t z2: ", z2.shape)
        #print("\n \n Z1: ", z1)
        #print("\n\nvae_elbo :",vae_elbo,"\n1st: ",torch.mean(lpxz1) , "\n 2nd: ", torch.mean(lpz1z2),"\n 3rd: ", torch.mean(lpz2), "\n 4th: ", torch.mean(lqz1x),"\n 5th: " ,torch.mean(lqz2z1))
        if vae_elbo > 0 :
            print("here")
            exit(1)
        # z.shape = [batch_size, k, latent_dim]


        active_units = np.array([torch.sum(torch.cov(z1.sum(1)).sum(0)>10**-2).item(), # sum over k
                        torch.sum(torch.cov(z2.sum(1).sum(1)).sum(0)>10**-2).item()]) # sum over k

        #print("lpxz1 : ",lpxz1, "lpz1z2 - lqz1x : ", lpz1z2 - lqz1x, "lpz2 - lqz2z1 : ",lpz2 - lqz2z1)

        return -vae_elbo, active_units


    def compute_evaluation_loss(self, x, k):
        # do stuff
        NLL = 0
        old_k = self.k
        # Set k to the new k for evaluation
        self.k = k
        self.model.k = k
        self.model.Encoder.k = k

        theta, mean, log_var, z = self.model(x)


        x_ki = x.unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        mu_z_ki = mean.unsqueeze(1).repeat(1, self.k, 1).to(self.device)
        sigma_z_ki = torch.exp(0.5*log_var).unsqueeze(1).repeat(1, self.k, 1).to(self.device)   # 0.5*      

        log_p =  dists.Bernoulli(theta).log_prob(x_ki).sum(axis=2)

        log_prior_z = dists.Normal(0, 1).log_prob(z).sum(axis=2) # should this not be sum(1) ? 
        log_q_z_g_x = dists.Normal(mu_z_ki, sigma_z_ki).log_prob(z).sum(axis=2)
        log_w = log_p + log_prior_z - log_q_z_g_x # shape: [batch_size, k]
        
        NLL = -(torch.logsumexp(log_w, 1) -  np.log(k)).mean()

        # Reset k
        self.k = old_k
        self.model.k = old_k
        self.model.Encoder.k = old_k

        return NLL, torch.mean(log_w) 


    def evaluate(self, test_loader, batch_size, k=5000):
        self.model.eval()
    
        total_samples = len(test_loader)
        total_NLL = 0

        # below we get decoder outputs for test data
        overall_active_units = 0
        with tqdm(total=total_samples) as pbar:
            with torch.no_grad():
                for batch_idx, (x, _) in enumerate(test_loader):

                    x = x.view(batch_size, self.x_dim)

                    x = x.to(self.device)
                    #optimizer.zero_grad(set_to_none=True)

                    theta, mean1, log_var1, z1, mean2, log_var2, z2, z_d, mean_d, log_var_d = self.model(x)
                    loss,  au  = self.loss_function1(x, theta, mean1, log_var1, z1, mean2, log_var2, z2, z_d, mean_d, log_var_d)
                    # ---------------------------------------
                    # may need to use compute evaluation loss
                    # ---------------------------------------
                    au = 0

                    total_NLL += loss.item()
                    pbar.update(1)

                    overall_active_units += au


        avg_NLL = total_NLL / total_samples
        active_units = overall_active_units / len(test_loader)

        return avg_NLL, active_units
