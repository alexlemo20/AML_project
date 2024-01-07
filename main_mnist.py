import torch
import torch.nn as nn
import time
import numpy as np
import os
from tqdm import tqdm

from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST, Omniglot
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from VAE import VAEModel
from IWAE import IWAEModel

from binarizations import BernoulliTransform, AlternativeTransform, ThresholdTransform, CustomMNIST


if __name__ == '__main__':
#def runMNIST():
  dataset_path = '~/datasets'
  outputs_dir = "outputs/MNIST/new_loss_bs80" # MNIST
  save_outputs = True # If the program should save the losses to file
  run_iwae = True
  run_vae = True
  batch_size = 80
  eval_batch_size = 20

  # Max i value
  max_i = 7 
  ks = [1,5,50]
  eval_k = 5000

  # Dimensions of the input, the hidden layer, and the latent space.
  x_dim  = 784
  hidden_dim = 200
  latent_dim = 50


  if not os.path.exists(outputs_dir):
    # Create a new directory because it does not exist
    os.makedirs(outputs_dir)


  ## MNIST DATASET
  train_dataset = CustomMNIST(dataset_path, transform=transforms.ToTensor(), train=True, download=True)
  test_dataset  = CustomMNIST(dataset_path, transform=transforms.ToTensor(), train=False, download=True)  


  # Dataloaders
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4) # 2 fastest
  test_loader  = DataLoader(dataset=test_dataset,  batch_size=eval_batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)

  # Used for gpu, comment out if run on cpu
  torch.backends.cudnn.benchmark = True
  torch.set_float32_matmul_precision('high')

  print("MNIST")
  for k in ks:
    print("Running k: ", k, " GPU: ", torch.cuda.is_available())
    ### IWAE
    if run_iwae:
      print("Running iwae")
      iwaeModel = IWAEModel(x_dim, hidden_dim, latent_dim, k=k)

      iwae_train_loss = iwaeModel.train(train_loader, max_i, batch_size)
      print("IWAE Training", iwae_train_loss)

      iwae_eval_nll, active_units = iwaeModel.evaluate(test_loader, eval_batch_size, k=eval_k)
      print("IWAE Evaluation complete!", "\t NLL :",iwae_eval_nll, "\t Active units: ", active_units)

      if save_outputs:
        torch.save(iwae_eval_nll, f"{outputs_dir}/k{k}_iwae_eval_nll.pt")
        torch.save(iwae_train_loss, f"{outputs_dir}/k{k}_iwae_train_loss.pt")
        torch.save(active_units, f"{outputs_dir}/k{k}_iwae_active_units.pt")
        torch.save(iwaeModel.model.state_dict(), f"{outputs_dir}/k{k}_iwae_trained_model.pt")  


    ### VAE
    if run_vae:
      print("Running vae")
      vaeModel = VAEModel(x_dim, hidden_dim, latent_dim, k=k)
      
      vae_train_loss = vaeModel.train(train_loader, max_i, batch_size)
      print("VAE Training", vae_train_loss)

      vae_eval_nll, active_units = vaeModel.evaluate(test_loader, eval_batch_size, k=eval_k)
      print("VAE Evaluation complete!","\t NLL :",vae_eval_nll, "\t Active units: ", active_units)

      if save_outputs:
        torch.save(vae_eval_nll, f"{outputs_dir}/k{k}_vae_eval_nll.pt")
        torch.save(vae_train_loss, f"{outputs_dir}/k{k}_vae_train_loss.pt")
        torch.save(active_units, f"{outputs_dir}/k{k}_vae_active_units.pt")
        torch.save(vaeModel.model.state_dict(), f"{outputs_dir}/k{k}_vae_trained_model.pt")      
