import torch
import torch.nn as nn
import time
import numpy as np
import os
from tqdm import tqdm

from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.distributions as dists


from VAE2 import VAEModel2
from IWAE2 import IWAEModel2


class BernoulliTransform(object):
  def __call__(self, x):
    #print("SIZE OF X: ", x.size())
    return dists.Bernoulli(x).sample().type(torch.float32) #torch.bernoulli(x).to(x.dtype)


if __name__ == '__main__':
  dataset_path = '~/datasets'
  outputs_dir = "outputs/L2"
  save_outputs = False # If the program should save the losses to file
  run_iwae = False
  run_vae = True
  batch_size = 20

  # Max i value
  max_i = 1 #7
  ks = [1,10,50]
  eval_k = 5000

  # Dimensions of the input, the hidden layer, and the latent space.
  x_dim  = 784
  hidden_dim_1 = 200
  latent_dim_1 = 100
  hidden_dim_2 = 100
  latent_dim_2 = 50

  mnist_transform = transforms.Compose([
          transforms.ToTensor(),
          BernoulliTransform(),
  ])


  if not os.path.exists(outputs_dir):
    # Create a new directory because it does not exist
    os.makedirs(outputs_dir)


  train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
  test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True) # num_workers=12
  test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True) # num_workers=12



  for k in ks:
    print("Running k: ", k)
    ### VAE
    if run_vae:
      vaeModel = VAEModel2(x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2, k=k)
      
      vae_train_loss, vae_train_nll = vaeModel.train(train_loader, max_i, batch_size)
      print("Training", vae_train_loss, "\nNLL train", vae_train_nll)

      vae_eval_nll = vaeModel.evaluate(test_loader, batch_size, k=eval_k)
      print("Evaluation complete!","\t NLL :",vae_eval_nll)

    if save_outputs:       

      # VAE
      if run_vae:
        torch.save(vae_eval_nll, f"{outputs_dir}/k{k}_vae_eval_nll.pt")
        torch.save(vae_train_loss, f"{outputs_dir}/k{k}_vae_train_loss.pt")
        torch.save(vae_train_nll, f"{outputs_dir}/k{k}_vae_train_nll.pt")
        torch.save(vaeModel.model.state_dict(), f"{outputs_dir}/k{k}_vae_trained_model.pt")        
