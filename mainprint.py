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

from VAEprint import VAEModel
from IWAEprint import IWAEModel

from binarizations import BernoulliTransform, AlternativeTransform, ThresholdTransform



if __name__ == '__main__':
  dataset_path = '~/datasets'
  outputs_dir = "outputs/L1"
  save_outputs = True # If the program should save the losses to file
  run_iwae = False
  run_vae = True
  batch_size = 20

  # Max i value
  max_i = 5 #7
  ks =[50] # [1,5,50]
  eval_k = 5000

  # Dimensions of the input, the hidden layer, and the latent space.
  x_dim  = 784
  hidden_dim = 200
  latent_dim = 50

  mnist_transform = transforms.Compose([
          transforms.ToTensor(),
          BernoulliTransform(),
  ])


  if not os.path.exists(outputs_dir):
    # Create a new directory because it does not exist
    os.makedirs(outputs_dir)


  train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
  test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
  test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)



  for k in ks:
    print("Running k: ", k)
    ### IWAE
    if run_iwae:
      iwaeModel = IWAEModel(x_dim, hidden_dim, latent_dim, k=k)

      iwae_train_loss = iwaeModel.train(train_loader, max_i, batch_size,"iwaeImages")
      print("IWAE Training", iwae_train_loss)

      iwae_eval_nll, active_units = iwaeModel.evaluate(test_loader, batch_size, k=eval_k)
      print("IWAE Evaluation complete!", "\t NLL :",iwae_eval_nll, "\t Active units: ", active_units)

      if save_outputs:
        torch.save(iwae_eval_nll, f"{outputs_dir}/k{k}_iwae_eval_nll.pt")
        torch.save(iwae_train_loss, f"{outputs_dir}/k{k}_iwae_train_loss.pt")
        torch.save(iwaeModel.model.state_dict(), f"{outputs_dir}/k{k}_iwae_trained_model.pt")  


    ### VAE
    if run_vae:
      vaeModel = VAEModel(x_dim, hidden_dim, latent_dim, k=k)
      
      vae_train_loss = vaeModel.train(train_loader,test_loader, max_i, batch_size, "vaeImages")
      print("VAE Training", vae_train_loss)


      if save_outputs:
        torch.save(vae_eval_nll, f"{outputs_dir}/k{k}_vae_eval_nll.pt")
        torch.save(vae_train_loss, f"{outputs_dir}/k{k}_vae_train_loss.pt")
        torch.save(vaeModel.model.state_dict(), f"{outputs_dir}/k{k}_vae_trained_model.pt")      