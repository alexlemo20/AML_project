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
from binarizations import BernoulliTransform


if __name__ == '__main__':
  dataset_path = '~/datasets'
  outputs_dir = "outputs/L2"
  save_outputs = False # If the program should save the losses to file
  run_iwae = True
  run_vae = False
  batch_size = 20

  # Max i value
  max_i = 5 #7
  ks = [1,5,50]
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

  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=12) # num_workers=12
  test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=12) # num_workers=12 <-- set workers to 0 if it crashes, idk why it doesnt work sometimes



  for k in ks:
    print("Running k: ", k)
    ### IWAE
    if run_iwae:
      iwaeModel = IWAEModel2(x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2, k=k)

      iwae_train_loss = iwaeModel.train(train_loader, max_i, batch_size)
      print("IWAE Training", iwae_train_loss)

      iwae_eval_nll, active_units = iwaeModel.evaluate(test_loader, batch_size, k=eval_k)
      print("IWAE Evaluation complete!", "\t NLL: ", iwae_eval_nll, "\t Active Units: ", active_units)

      if save_outputs:
        torch.save(iwae_eval_nll, f"{outputs_dir}/k{k}_iwae_eval_nll.pt")
        torch.save(iwae_train_loss, f"{outputs_dir}/k{k}_iwae_train_loss.pt")
        torch.save(iwaeModel.model.state_dict(), f"{outputs_dir}/k{k}_iwae_trained_model.pt")  

    ### VAE
    if run_vae:
      vaeModel = VAEModel2(x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2, k=k)

      vae_train_loss = vaeModel.train(train_loader, max_i, batch_size)
      print("Training", vae_train_loss)

      vae_eval_nll, active_units = vaeModel.evaluate(test_loader, batch_size, k=eval_k)
      print("Evaluation complete!","\t NLL :",vae_eval_nll, "\t Active Units: ", active_units)

      if save_outputs:       
        torch.save(vae_eval_nll, f"{outputs_dir}/k{k}_vae_eval_nll.pt")
        torch.save(vae_train_loss, f"{outputs_dir}/k{k}_vae_train_loss.pt")
        torch.save(vaeModel.model.state_dict(), f"{outputs_dir}/k{k}_vae_trained_model.pt")        
