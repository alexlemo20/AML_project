import torch
import os
from torch.utils.data import DataLoader

from VAE2 import VAEModel2
from IWAE2 import IWAEModel2
from binarizations import CustomOmniglot


if __name__ == '__main__':
  print("Omniglot")
  dataset_path = '~/datasets'
  outputs_dir = "outputs/omniglot/Run4_minus_k/L2"
  save_outputs = True # If the program should save the losses to file
  run_iwae = True
  run_vae = False  
  batch_size = 80
  compile_model = True

  # Max i value
  max_i = 7
  ks = [1,5,50]
  eval_k = 5000

  # Dimensions of the input, the hidden layer, and the latent space.
  x_dim  = 784
  hidden_dim_1 = 200
  latent_dim_1 = 100
  hidden_dim_2 = 100
  latent_dim_2 = 50

  if not os.path.exists(outputs_dir):
    # Create a new directory because it does not exist
    os.makedirs(outputs_dir)


  train_dataset = CustomOmniglot(train=True) # Same as in the paper
  test_dataset  = CustomOmniglot(train=False) 

  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=2 if torch.cuda.is_available() else 0) # num_workers=12
  test_loader  = DataLoader(dataset=test_dataset,  batch_size=20, shuffle=False, drop_last=True, pin_memory=True, num_workers=2 if torch.cuda.is_available() else 0) # num_workers=12

  if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


  for k in ks:
    print("Running k: ", k, " GPU: ", torch.cuda.is_available())
    ### IWAE
    if run_iwae:
      print("Running iwae")
      iwaeModel = IWAEModel2(x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2, k=k, compile_model=compile_model)

      iwae_train_loss = iwaeModel.train(train_loader, max_i, batch_size)
      print("IWAE Training", iwae_train_loss)

      iwae_eval_nll, active_units = iwaeModel.evaluate(test_loader, 20, k=eval_k)
      print("IWAE Evaluation complete!", "\t NLL: ", iwae_eval_nll, "\t Active Units: ", active_units)

      if save_outputs:
        torch.save(iwae_eval_nll, f"{outputs_dir}/k{k}_iwae_eval_nll.pt")
        torch.save(iwae_train_loss, f"{outputs_dir}/k{k}_iwae_train_loss.pt")
        torch.save(active_units, f"{outputs_dir}/k{k}_iwae_active_units.pt")
        torch.save(iwaeModel.model.state_dict(), f"{outputs_dir}/k{k}_iwae_trained_model.pt")  

    ### VAE
    if run_vae:
      print("Running vae")
      vaeModel = VAEModel2(x_dim, hidden_dim_1, latent_dim_1, hidden_dim_2, latent_dim_2, k=k, compile_model=compile_model)

      vae_train_loss = vaeModel.train(train_loader, max_i, batch_size)
      print("Training", vae_train_loss)

      vae_eval_nll, active_units = vaeModel.evaluate(test_loader, 20, k=eval_k)
      print("Evaluation complete!","\t NLL :",vae_eval_nll, "\t Active Units: ", active_units)

      if save_outputs:       
        torch.save(vae_eval_nll, f"{outputs_dir}/k{k}_vae_eval_nll.pt")
        torch.save(vae_train_loss, f"{outputs_dir}/k{k}_vae_train_loss.pt")
        torch.save(active_units, f"{outputs_dir}/k{k}_vae_active_units.pt")
        torch.save(vaeModel.model.state_dict(), f"{outputs_dir}/k{k}_vae_trained_model.pt")        
