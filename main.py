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
  def __init__(self, thr):
    self.thr = thr  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type


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
        ThresholdTransform(thr=0.5)
])


train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)


iwaeModel = IWAEModel(x_dim, hidden_dim, latent_dim, k=k)
vaeModel = VAEModel(x_dim, hidden_dim, latent_dim, k=k)

iwae_overall_loss, iwae_loss_epochs, iwae_nll_epochs = iwaeModel.train(train_loader, max_i, batch_size)
print("Training", iwae_loss_epochs, "Nll train", iwae_nll_epochs)

iwae_avg_eval_loss, iwae_avg_NLL, iwae_NLL = iwaeModel.evaluate(test_loader, batch_size)
print("Test", iwae_avg_eval_loss, "Test nll", iwae_avg_NLL, "Nll evaluation", iwae_NLL)

vae_overall_loss, vae_loss_epochs, vae_nll_epochs = vaeModel.train(train_loader, max_i, batch_size)
print("Training", vae_loss_epochs, "Nll train", vae_nll_epochs)

if save_outputs:
  torch.save(vae_loss_epochs, f"{outputs_dir}/vae_train_loss.pt")
  torch.save(vae_nll_epochs, f"{outputs_dir}/vae_train_nll.pt")

vae_avg_eval_loss, vae_avg_NLL, vae_NLL = vaeModel.evaluate(test_loader, batch_size)
print("Test", vae_avg_eval_loss, "Test nll", vae_avg_NLL, "Nll evaluation", vae_NLL)

if save_outputs:
  torch.save(vae_avg_eval_loss, f"{outputs_dir}/vae_eval_loss.pth")
  torch.save(vae_avg_NLL, f"{outputs_dir}/vae_eval_nll.pth")
