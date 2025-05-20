import os
current_working_directory = os.getcwd()
print(current_working_directory)

import numpy as np 

import torch 
import torch.nn as nn 
import torch.optim as optim

# import torch_optimizer as optim_custom
from torch.utils.data import Dataset, DataLoader
from bernstein_torch import bernstein_coeff_ordern_new
import scipy.io as sio

# from models.mlp_qp_vis_aware_2 import MLP, vis_aware_track_net, PointNet
# import pol_matrix_comp
from tqdm import trange,tqdm

from mlp_manipulator import MLP, MLPProjectionFilter

import os

#Cell 2
t_fin = 20.0
num = 100
tot_time = torch.linspace(0, t_fin, num)
tot_time_copy = tot_time.reshape(num, 1)
P, Pdot, Pddot = bernstein_coeff_ordern_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)


# P_diag = torch.block_diag(P, P)
# Pdot_diag = torch.block_diag(Pdot, Pdot)

# Pddot_diag = torch.block_diag(Pddot, Pddot)
nvar_single = P.size(dim = 1) 
num_dof = 6
nvar = nvar_single*num_dof

#Cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

data = np.load("sample_dataset_final.npz")

xi_filtered = data['xi_filtered']
xi_samples = data['xi_samples']
state_terms = data['state_terms']

xi_samples = xi_samples[:, -1]
xi_filtered = xi_filtered[:, -1]

print("xi_filtered", xi_filtered.shape)
print("xi_samples", xi_samples.shape)
print("state_terms", state_terms.shape)

#Cell 4

#data = np.load("dataset")

#init_state = data['init_state_data']

init_state =state_terms

c_samples_input = xi_samples

#Only first time-step
init_state_ = init_state[0]
c_samples_input_ = c_samples_input[0]
inp = np.hstack((init_state_, c_samples_input_))

# #Full Time interval
# inp = np.concatenate((init_state, c_samples_input), axis=2)

inp_mean, inp_std = inp.mean(), inp.std()


print("c_samples_input", c_samples_input.shape)

#Cell 5

# Custom Dataset Loader 
class TrajDataset(Dataset):
	"""Expert Trajectory Dataset."""
	def __init__(self, inp, init_state, c_samples_input):
		
		# input
		self.inp = inp
		# State Data
		self.init_state = init_state
		
		self.c_samples_input = c_samples_input
	
	def __len__(self):
		return len(self.inp)    
			
	def __getitem__(self, idx):
		
		# Inputs
		init_state = self.init_state[idx]
		
		c_samples_input = self.c_samples_input[idx]
  
		inp = self.inp[idx]
		
				 
		return torch.tensor(inp).float(), torch.tensor(init_state).float(), torch.tensor(c_samples_input).float() 

# Batch Size For Training - 3k or 4k 
batch_size = 100 

# Using PyTorch Dataloader
train_dataset = TrajDataset(inp, init_state_, c_samples_input_)

# sample = train_dataset[0]

# print("Number of elements in sample:", len(sample))

# inp_tensor, init_state_tensor, c_samples_input_tensor = sample

# print("inp shape:", inp_tensor.shape)
# print("init_state shape:", init_state_tensor.shape)
# print("c_samples_input shape:", c_samples_input_tensor.shape)

# print(type(train_dataset))                    # Should be your custom class
# print(type(train_dataset[0]))                 # Should be tuple
# print(type(train_dataset[0][0]))              # Should be torch.Tensor

#print(train_dataset[0])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

# batch = next(iter(train_loader))

# inp_batch, init_state_batch, c_samples_batch = batch

# print("inp_batch shape:", inp_batch.shape)
# print("init_state_batch shape:", init_state_batch.shape)
# print("c_samples_batch shape:", c_samples_batch.shape)

# print("inp_batch type:", type(inp_batch))
# print("init_state_batch type:", type(init_state_batch))
# print("c_samples_batch type:", type(c_samples_batch))

    
    
#Cell 7

# Differentiable Layer
num_batch = train_loader.batch_size

P = P.to(device) 
Pdot = Pdot.to(device)
# P_diag = P_diag.to(device)
# Pdot_diag = Pdot_diag.to(device)

# Pddot_diag = Pddot_diag.to(device)



# num_dot = num 
# num_ddot = num_dot 
# num_constraint = 2*num+2*num_dot+2*num_ddot

# CVAE input
enc_inp_dim = np.shape(inp)[1] 
mlp_inp_dim = enc_inp_dim
hidden_dim = 1024
mlp_out_dim = 4*nvar#+3*num_constraint (lambda- 0:3*nvar, c_samples- 3*nvar:4*nvar)
#print(mlp_out_dim)


mlp =  MLP(mlp_inp_dim, hidden_dim, mlp_out_dim)

#print(mlp)
print("P", P.size())

model = MLPProjectionFilter(P, Pdot, Pddot, mlp, num_batch, inp_mean, inp_std, t_fin).to(device)

print(type(model))

#print(model)
# model.train()

#Cell 8

epochs = 50
step, beta = 0, 1.0 # 3.5
optimizer = optim.AdamW(model.parameters(), lr = 2e-4, weight_decay=6e-5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
losses = []
last_loss = torch.inf
avg_train_loss, avg_primal_loss, avg_fixed_point_loss = [], [], []
for epoch in range(epochs):
    
    # Train Loop
    losses_train, primal_losses, fixed_point_losses = [], [], []
    
    for (inp, init_state, c_samples_input) in tqdm(train_loader):
        
        # Input and Output 
        inp = inp.to(device)
        init_state = init_state.to(device)
        c_samples_input = c_samples_input.to(device)

        # print("inp shape:", inp.shape)
        # print("init_state shape:", init_state.shape)
        # print("c_samples_input shape:", c_samples_input.shape)

        
        
        c_samples, accumulated_res_fixed_point, accumulated_res_primal, accumulated_res_primal_temp, accumulated_res_fixed_point_temp = model(inp, init_state, c_samples_input)
        primal_loss, fixed_point_loss, loss = model.mlp_loss(accumulated_res_primal, accumulated_res_fixed_point, c_samples, c_samples_input)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses_train.append(loss.detach().cpu().numpy()) 
        primal_losses.append(primal_loss.detach().cpu().numpy())
        fixed_point_losses.append(fixed_point_loss.detach().cpu().numpy())
        

    if epoch % 2 == 0:    
        print(f"Epoch: {epoch + 1}, Train Loss: {np.average(losses_train):.3f}, primal: {np.average(primal_losses):.3f}, fixed_point: {np.average(fixed_point_losses):.3f} ")

    step += 0.07 #0.15
    # scheduler.step()
    
    os.makedirs("./training_scripts/weights", exist_ok=True)
    if loss <= last_loss:
            torch.save(model.state_dict(), f"./training_scripts/weights/mlp_learned_proj_manipulator.pth")
            last_loss = loss
    avg_train_loss.append(np.average(losses_train)), avg_primal_loss.append(np.average(primal_losses)), \
    avg_fixed_point_loss.append(np.average(fixed_point_losses))
    
