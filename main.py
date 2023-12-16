import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from VAE import VAEModel

class ThresholdTransform(object):
  def __init__(self, thr):
    self.thr = thr  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type


dataset_path = '~/datasets'

batch_size = 20

# Max i value
max_i = 3 # 7
k = 10

# Dimensions of the input, the hidden layer, and the latent space.
vae_x_dim  = 784
vae_hidden_dim = 200
vae_latent_dim = 50

mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        ThresholdTransform(thr=0.5)
])


train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)


vaeModel = VAEModel(vae_x_dim, vae_hidden_dim, vae_latent_dim, k=k)

vae_overall_loss, vae_loss_epochs = vaeModel.train(train_loader, max_i, batch_size)

print(vae_overall_loss)
print(vae_loss_epochs)

vae_avg_eval_loss = vaeModel.evaluate(test_loader, batch_size)
