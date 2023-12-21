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

from VAE import VAEModel
from IWAE import IWAEModel

class ThresholdTransform(object):
  def __call__(self, x):
    torch.manual_seed(30)
    return (x > torch.rand_like(x)).to(x.dtype)  # do not change the data type

class BernoulliTransform(object):
  def __call__(self, x):
    #print("SIZE OF X: ", x.size())
    return torch.bernoulli(x).to(x.dtype)
  
# binarization based on the code of the paper
class AlternativeTransform(object):
  def __call__(self, x):
    torch.manual_seed(30)
    #torch.manual_seed(int(time.time() * 100))
    num_cases, num_dims, num_batches = x.size()
    return (x > torch.rand(num_cases, num_dims, num_batches)).to(x.dtype)



dataset_path = '~/datasets'
outputs_dir = "outputs/L1"
save_outputs = False # If the program should save the losses to file
run_iwae = False
run_vae = True
batch_size = 20

# Max i value
max_i = 1 # 7
k = 10
eval_k = 500 # 5000

# Dimensions of the input, the hidden layer, and the latent space.
x_dim  = 784
hidden_dim = 200
latent_dim = 50

mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        #ThresholdTransform(),
        BernoulliTransform(),
        #AlternativeTransform(),
])


if not os.path.exists(outputs_dir):
   # Create a new directory because it does not exist
   os.makedirs(outputs_dir)


train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)



### IWAE
if run_iwae:
  iwaeModel = IWAEModel(x_dim, hidden_dim, latent_dim, k=k)

  iwae_train_loss, iwae_train_nll = iwaeModel.train(train_loader, max_i, batch_size)
  print("Training", iwae_train_loss, "\nNLL train", iwae_train_nll)

  iwae_eval_nll = iwaeModel.evaluate(test_loader, batch_size, k=eval_k)
  print("Evaluation complete!", "\t NLL :",iwae_eval_nll)

### VAE
if run_vae:
  vaeModel = VAEModel(x_dim, hidden_dim, latent_dim, k=k)

  #vae_eval_nll = vaeModel.evaluate(test_loader, batch_size, k=eval_k)
  #print("Evaluation complete!","\t NLL :",vae_eval_nll)
  
  vae_train_loss, vae_train_nll = vaeModel.train(train_loader, max_i, batch_size)
  print("Training", vae_train_loss, "\nNLL train", vae_train_nll)

  vae_eval_nll = vaeModel.evaluate(test_loader, batch_size, k=eval_k)
  print("Evaluation complete!","\t NLL :",vae_eval_nll)

if save_outputs:
  #IWAE
  if run_iwae:
    torch.save(iwae_eval_nll, f"{outputs_dir}/iwae_eval_nll.pt")
    torch.save(iwae_train_loss, f"{outputs_dir}/iwae_train_loss.pt")
    torch.save(iwae_train_nll, f"{outputs_dir}/iwae_train_nll.pt")

  # VAE
  if run_vae:
    torch.save(vae_eval_nll, f"{outputs_dir}/vae_eval_nll.pt")
    torch.save(vae_train_loss, f"{outputs_dir}/vae_train_loss.pt")
    torch.save(vae_train_nll, f"{outputs_dir}/vae_train_nll.pt")
