import torch.distributions as dists
import torch
import scipy.io
from torch.utils.data import Dataset
from torchvision import datasets

class ThresholdTransform(object):
  def __call__(self, x):
    torch.manual_seed(30)
    return (x > torch.rand_like(x)).to(x.dtype)  # do not change the data type

class BernoulliTransform(object):
  def __call__(self, x):
    #print("SIZE OF X: ", x.size())
    return dists.Bernoulli(x).sample().type(torch.float32) #torch.bernoulli(x).to(x.dtype)
  
# binarization based on the code of the paper
class AlternativeTransform(object):
  def __call__(self, x):
    torch.manual_seed(30)
    #torch.manual_seed(int(time.time() * 100))
    num_cases, num_dims, num_batches = x.size()
    return (x > torch.rand(num_cases, num_dims, num_batches)).to(x.dtype)
  
class CustomOmniglot(Dataset):
    def __init__(self, train):
        omni_raw = scipy.io.loadmat('data/omniglot/chardata.mat')
        if train:
            self.data = torch.from_numpy(self.reshape_data(omni_raw['data'].T.astype('float32'))) # train_data
        else:
            self.data = torch.from_numpy(self.reshape_data(omni_raw['testdata'].T.astype('float32'))) #test_data

    def reshape_data(self, data):
        return data.reshape((-1, 1, 28, 28))#.reshape((-1, 28*28))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = 0 # label doesnt matter for autoencoders

        return dists.Bernoulli(self.data[idx]).sample().type(torch.float32), label

class CustomMNIST(datasets.MNIST):
    def __init__(self, root, train, transform=None, target_transform=None, download=False):
        super(CustomMNIST, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        return dists.Bernoulli(img).sample().type(torch.float32), target