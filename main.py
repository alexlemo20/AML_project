import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from VAE import VAEModel
from IWAE import IWAEModel

class ThresholdTransform(object):
  def __call__(self, x):
    return (x > torch.rand_like(x)).to(x.dtype)  # do not change the data type


dataset_path = '~/datasets'
outputs_dir = "outputs"
save_outputs = False # If the program should save the losses to file

batch_size = 20

# Max i value
max_i = 3 # 7
k = 5

# Dimensions of the input, the hidden layer, and the latent space.
x_dim  = 784
hidden_dim = 200
latent_dim = 50

mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        ThresholdTransform(),
])


train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)


### IWAE
iwaeModel = IWAEModel(x_dim, hidden_dim, latent_dim, k=k)

iwae_train_loss, iwae_train_nll = iwaeModel.train(train_loader, max_i, batch_size)
print("Training", iwae_train_loss, "\nNLL train", iwae_train_nll)

iwae_eval_loss, iwae_eval_nll = iwaeModel.evaluate(test_loader, batch_size)
print("Evaluation complete!", "\tAverage Loss: ", iwae_eval_loss,"\t NLL :",iwae_eval_nll)

### VAE
vaeModel = VAEModel(x_dim, hidden_dim, latent_dim, k=k)

vae_train_loss, vae_train_nll = vaeModel.train(train_loader, max_i, batch_size)
print("Training", vae_train_loss, "\nNLL train", vae_train_nll)

vae_eval_loss, vae_eval_nll = vaeModel.evaluate(test_loader, batch_size)
print("Evaluation complete!", "\tAverage Loss: ", vae_eval_loss,"\t NLL :",vae_eval_nll)

if save_outputs:
  #IWAE
  torch.save(iwae_eval_loss, f"{outputs_dir}/iwae_eval_loss.pt")
  torch.save(iwae_eval_nll, f"{outputs_dir}/iwae_eval_nll.pt")
  torch.save(iwae_train_loss, f"{outputs_dir}/iwae_train_loss.pt")
  torch.save(iwae_train_nll, f"{outputs_dir}/iwae_train_nll.pt")

  # VAE
  torch.save(vae_eval_loss, f"{outputs_dir}/vae_eval_loss.pt")
  torch.save(vae_eval_nll, f"{outputs_dir}/vae_eval_nll.pt")
  torch.save(vae_train_loss, f"{outputs_dir}/vae_train_loss.pt")
  torch.save(vae_train_nll, f"{outputs_dir}/vae_train_nll.pt")
