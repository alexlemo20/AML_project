import torch.distributions as dists
import torch

class ThresholdTransform(object):
  def __call__(self, x):
    torch.manual_seed(30)
    return (x > torch.rand_like(x)).to(x.dtype)  # do not change the data type

class BernoulliTransform(object):
  def __call__(self, x):
    #print("SIZE OF X: ", x.size())
    return dists.Bernoulli(x).sample().type(x.dtype) #torch.bernoulli(x).to(x.dtype)
  
# binarization based on the code of the paper
class AlternativeTransform(object):
  def __call__(self, x):
    torch.manual_seed(30)
    #torch.manual_seed(int(time.time() * 100))
    num_cases, num_dims, num_batches = x.size()
    return (x > torch.rand(num_cases, num_dims, num_batches)).to(x.dtype)