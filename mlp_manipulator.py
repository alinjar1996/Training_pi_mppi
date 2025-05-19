

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import os

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_default_dtype(torch.float32)

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



class MLP(nn.Module):
	def __init__(self, inp_dim, hidden_dim, out_dim):
		super(MLP, self).__init__()
		
		# MC Dropout Architecture
		self.mlp = nn.Sequential(
			nn.Linear(inp_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),

			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),


			nn.Linear(hidden_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			
			nn.Linear(256, out_dim),
		)
	
	def forward(self, x):
		out = self.mlp(x)
		return out


class mlp_projection_filter(nn.Module):
	
    def __init__(self, P, Pdot, Pddot, mlp, num_batch, inp_mean, inp_std, t_fin):
        super(mlp_projection_filter, self).__init__()
        
        # BayesMLP
        self.mlp = mlp
        
        # Normalizing Constants


        self.v_max = 0.8
        self.a_max = 1.8
        self.p_max = 180*np.pi/180


        # P Matrices
        self.P = P.to(device)
        self.Pdot = Pdot.to(device)
        self.Pddot = Pddot.to(device)

        # No. of Variables
        self.nvar_single = P.size(dim = 1)
        self.num = P.size(dim = 0)
        self.num_batch = num_batch
        self.num_dof = 6

        self.nvar = self.num_dof * self.nvar_single

        self.A_projection = torch.eye(self.nvar, device = device)


        self.rho_ineq = 1.0
        self.rho_projection = 1.0

        # A_v
        A_v_ineq, A_v = self.get_A_v()
        self.A_v_ineq = A_v_ineq.detach().to(device = device)  # or A_v_ineq if you need gradients
        self.A_v = A_v.detach().to(device = device)

        # A_a
        A_a_ineq, A_a = self.get_A_a()
        self.A_a_ineq = A_a_ineq.detach().to(device = device)
        self.A_a = A_a.detach().to(device = device)

        # A_p
        A_p_ineq, A_p = self.get_A_p()
        self.A_p_ineq = A_p_ineq.detach().to(device = device)
        self.A_p = A_p.detach().to(device = device)

        # A_eq
        A_eq = self.get_A_eq().to(device = device)
        self.A_eq = A_eq.detach().to(device = device)

        # Q_inv
        Q_inv = self.get_Q_inv().to(device = device)
        self.Q_inv = Q_inv.detach().to(device = device)

        # Trajectory matrices
        A_theta, A_thetadot, A_thetaddot = self.get_A_traj()
        self.A_theta = A_theta.detach().to(device = device)
        self.A_thetadot = A_thetadot.detach().to(device = device)
        self.A_thetaddot = A_thetaddot.detach().to(device = device)

    

        self.maxiter =  15

        self.t_fin = t_fin		

        self.tot_time = torch.linspace(0, t_fin, self.num, device=device)

        # self.A = torch.vstack([ self.P, -self.P  ])
        # self.A_dot = torch.vstack([ self.Pdot, -self.Pdot  ])
        # self.A_ddot = torch.vstack([self.Pddot, -self.Pddot  ])
        # self.A_control = torch.vstack([self.A, self.A_dot, self.A_ddot ])
        
        # self.A_eq_control = torch.vstack([self.P[0], self.Pdot[0] ])
        self.inp_mean = inp_mean
        self.inp_std = inp_std
        

        # self.num_dot = self.num
        # self.num_ddot = self.num_dot

        # self.num_constraint = 2*self.num+2*self.num_dot+2*self.num_ddot

        ########################################
        
        # RCL Loss
        self.rcl_loss = nn.MSELoss()


    def get_A_traj(self):
        I = torch.eye(self.num_dof, device=self.P.device)
        A_theta = torch.kron(I, self.P)
        A_thetadot = torch.kron(I, self.Pdot)
        A_thetaddot = torch.kron(I, self.Pddot)
        return A_theta, A_thetadot, A_thetaddot

    def get_A_p(self):
        A_p = torch.vstack((self.P, -self.P))
        A_p_ineq = torch.kron(torch.eye(self.num_dof, device=self.P.device), A_p)
        return A_p_ineq, A_p

    def get_A_v(self):
        A_v = torch.vstack((self.Pdot, -self.Pdot))
        A_v_ineq = torch.kron(torch.eye(self.num_dof, device=self.P.device), A_v)
        return A_v_ineq, A_v

    def get_A_a(self):
        A_a = torch.vstack((self.Pddot, -self.Pddot))
        A_a_ineq = torch.kron(torch.eye(self.num_dof, device=self.P.device), A_a)
        return A_a_ineq, A_a

    def get_A_eq(self):
        A_eq = torch.kron(torch.eye(self.num_dof, device=self.P.device), torch.vstack((
            self.P[0].unsqueeze(0),
            self.Pdot[0].unsqueeze(0),
            self.Pddot[0].unsqueeze(0),
            self.Pdot[-1].unsqueeze(0),
            self.Pddot[-1].unsqueeze(0)
        )))

        return A_eq
        

    def get_Q_inv(self):
        Q_top_left = (
            self.A_projection.T @ self.A_projection +
            self.rho_ineq * (self.A_v_ineq.T @ self.A_v_ineq) +
            self.rho_ineq * (self.A_a_ineq.T @ self.A_a_ineq) +
            self.rho_ineq * (self.A_p_ineq.T @ self.A_p_ineq)
        )
        Q_top = torch.hstack((Q_top_left, self.A_eq.T))
        Q_bottom = torch.hstack((self.A_eq, torch.zeros((self.A_eq.shape[0], self.A_eq.shape[0]), device=self.A_eq.device)))
        Q = torch.vstack((Q_top, Q_bottom))
        Q_inv = torch.inverse(Q)
        return Q_inv


  
  
    def compute_boundary_vec(self, state_term):
        
        # print("state_term", state_term.shape)
        # b_eq_term = state_term.reshape(5, self.num_dof).T
        # b_eq_term = b_eq_term.reshape(self.num_dof*5)

        b_eq_term = state_term
        
        return b_eq_term




    def compute_feasible_control(self,lamda_v, lamda_a, lamda_p, s_v, s_a, s_p, b_eq_term, xi_samples):
        
        device = lamda_v.device  # assumes all tensors are on same device (CPU or CUDA)

        # Build inequality bounds
        v_max_temp = torch.hstack((
            self.v_max * torch.ones((self.num_batch, self.num), device=device),
            self.v_max * torch.ones((self.num_batch, self.num), device=device)
        ))
        v_max_vec = v_max_temp.repeat(1, self.num_dof)

        a_max_temp = torch.hstack((
            self.a_max * torch.ones((self.num_batch, self.num), device=device),
            self.a_max * torch.ones((self.num_batch, self.num), device=device)
        ))
        a_max_vec = a_max_temp.repeat(1, self.num_dof)

        p_max_temp = torch.hstack((
            self.p_max * torch.ones((self.num_batch, self.num), device=device),
            self.p_max * torch.ones((self.num_batch, self.num), device=device)
        ))
        p_max_vec = p_max_temp.repeat(1, self.num_dof)

        b_v = v_max_vec
        b_a = a_max_vec
        b_p = p_max_vec

        b_v_aug = b_v - s_v
        b_a_aug = b_a - s_a
        b_p_aug = b_p - s_p

        # Compute linear cost
        lincost = (
            -lamda_v - lamda_a - lamda_p
            - self.rho_projection * torch.matmul(self.A_projection.T, xi_samples.T).T
            - self.rho_ineq * torch.matmul(self.A_v_ineq.T, b_v_aug.T).T
            - self.rho_ineq * torch.matmul(self.A_a_ineq.T, b_a_aug.T).T
            - self.rho_ineq * torch.matmul(self.A_p_ineq.T, b_p_aug.T).T
        )

        # Solve KKT system
        rhs = torch.hstack((-lincost, b_eq_term))
        sol = torch.matmul(self.Q_inv, rhs.T).T

        primal_sol = sol[:, :self.nvar]

        # Slack and residual updates
        s_v = torch.maximum(
            torch.zeros(self.num_batch, 2*self.num*self.num_dof).to(device = device),
            -torch.matmul(self.A_v_ineq, primal_sol.T).T + b_v
        )
        res_v = torch.matmul(self.A_v_ineq, primal_sol.T).T - b_v + s_v

        s_a = torch.maximum(
            torch.zeros(self.num_batch, 2*self.num*self.num_dof).to(device = device),
            -torch.matmul(self.A_a_ineq, primal_sol.T).T + b_a
        )
        res_a = torch.matmul(self.A_a_ineq, primal_sol.T).T - b_a + s_a

        s_p = torch.maximum(
            torch.zeros(self.num_batch, 2*self.num*self.num_dof).to(device = device),
            -torch.matmul(self.A_p_ineq, primal_sol.T).T + b_p
        )
        res_p = torch.matmul(self.A_p_ineq, primal_sol.T).T - b_p + s_p

        # Dual variable updates
        lamda_v = lamda_v - self.rho_ineq * torch.matmul(self.A_v_ineq.T, res_v.T).T
        lamda_a = lamda_a - self.rho_ineq * torch.matmul(self.A_a_ineq.T, res_a.T).T
        lamda_p = lamda_p - self.rho_ineq * torch.matmul(self.A_p_ineq.T, res_p.T).T

        # Residual norms (projection quality)
        res_v = torch.norm(res_v, dim=1)
        res_a = torch.norm(res_a, dim=1)
        res_p = torch.norm(res_p, dim=1)

        res_projection = res_v + res_a + res_p

        return primal_sol, s_v, s_a, s_p, lamda_v, lamda_a, lamda_p, res_projection



    def compute_s_init(self, primal_sol):
        
        device = primal_sol.device  # get the device of incoming tensor

        # Move all relevant tensors to the same device
        self.A_v_ineq.to(device)
        
        
        # Build inequality bounds
        v_max_temp = torch.hstack((
            self.v_max * torch.ones((self.num_batch, self.num), device=device),
            self.v_max * torch.ones((self.num_batch, self.num), device=device)
        )).to(device)
        v_max_vec = v_max_temp.repeat(1, self.num_dof).to(device)

        print("v_max_temp", v_max_temp.shape)
        print("v_max_vec", v_max_vec.shape)

        a_max_temp = torch.hstack((
            self.a_max * torch.ones((self.num_batch, self.num), device=device),
            self.a_max * torch.ones((self.num_batch, self.num), device=device)
        ))
        a_max_vec = a_max_temp.repeat(1, self.num_dof)

        p_max_temp = torch.hstack((
            self.p_max * torch.ones((self.num_batch, self.num), device=device),
            self.p_max * torch.ones((self.num_batch, self.num), device=device)
        ))
        p_max_vec = p_max_temp.repeat(1, self.num_dof)

        b_v = v_max_vec.to(device = device)
        b_a = a_max_vec.to(device = device)
        b_p = p_max_vec.to(device = device)
        
        print("self.A_v_ineq", self.A_v_ineq.shape)
        print("primal_sol", primal_sol.shape)
        print("b_v", b_v.shape)
        s_v = torch.maximum(
            torch.zeros(self.num_batch, 2*self.num*self.num_dof).to(device = device),
            -torch.matmul(self.A_v_ineq, primal_sol.T).T + b_v
        )
        

        s_a = torch.maximum(
            torch.zeros(self.num_batch, 2*self.num*self.num_dof).to(device = device),
            -torch.matmul(self.A_a_ineq, primal_sol.T).T + b_a
        )
        
        s_p = torch.maximum(
            torch.zeros(self.num_batch, 2*self.num*self.num_dof).to(device = device),
            -torch.matmul(self.A_p_ineq, primal_sol.T).T + b_p
        )
        
        
        return s_v, s_a, s_p



        # torch.linalg.norm(c_v_samples-c_v_samples_prev, dim = 1) + \
        # 					  torch.linalg.norm(c_pitch_samples-c_pitch_samples_prev, dim = 1) +\
        #                       torch.linalg.norm(c_roll_samples-c_roll_samples_prev, dim = 1) +\
                                    
                                    


    def custom_forward(self, lamda_samples, c_samples_input, c_samples, b_eq):
        

        cost_mat_inv_control = 	self.get_Q_inv()


        s_v, s_a, s_p = self.compute_s_init(c_samples)


        lamda_v = lamda_samples [:,0:self.nvar].to(device = device)    
        lamda_a = lamda_samples [:,self.nvar:2*self.nvar].to(device = device)   
        lamda_p = lamda_samples [:,2*self.nvar:3*self.nvar].to(device = device)

                                     
        # print(s_v)    
        
        accumulated_res_primal = []

        accumulated_res_fixed_point = []

        for i in range(0, self.maxiter):
        
   
            c_samples_prev = c_samples.clone()
            
            lamda_v_prev = lamda_v.clone() 
            lamda_a_prev = lamda_a.clone()
            lamda_p_prev = lamda_p.clone()
            s_v_prev = s_v.clone() 
            s_a_prev = s_a.clone()
            s_p_prev = s_p.clone()

            
            # print("in ",c_v_samples_input)

            c_samples, s_v, s_a, s_p, lamda_v, lamda_a, lamda_p, res_projection =  self.compute_feasible_control(lamda_v, lamda_a, lamda_p, s_v, s_a, s_p, b_eq, c_samples)
            

        
            accumulated_res_primal.append(res_projection)

            fixed_point_res = torch.linalg.norm(lamda_v-lamda_v_prev, dim = 1) +\
                                torch.linalg.norm(lamda_a-lamda_a_prev, dim = 1) + \
                                torch.linalg.norm(lamda_p-lamda_p_prev, dim = 1) + \
                                torch.linalg.norm(s_v-s_v_prev, dim = 1) + \
                                torch.linalg.norm(s_a-s_a_prev, dim = 1)  + \
                                torch.linalg.norm(s_p-s_p_prev, dim = 1)+\
                                torch.linalg.norm(c_samples-c_samples_prev, dim = 1)
                                		
                                
                                
            accumulated_res_fixed_point.append(fixed_point_res)


        accumulated_res_primal_temp = accumulated_res_primal
        accumulated_res_fixed_point_temp = accumulated_res_fixed_point
        
        res_primal_stack = torch.stack(accumulated_res_primal )
        res_fixed_stack = torch.stack(accumulated_res_fixed_point )

        accumulated_res_primal = torch.sum(res_primal_stack, axis = 0)/self.maxiter
        accumulated_res_fixed_point = torch.sum(res_fixed_stack, axis = 0)/self.maxiter

        return c_samples, accumulated_res_fixed_point, accumulated_res_primal, accumulated_res_primal_temp, accumulated_res_fixed_point_temp


    def decoder_function(self, inp_norm, init_state, c_samples_input):
        
        # v_init, v_dot_init, pitch_init, pitch_dot_init, roll_init, roll_dot_init  = init_stat

        # v_init = init_state[:, 0]
        # v_dot_init = init_state[:, 1]
        # pitch_init = init_state[:, 2]
        # pitch_dot_init = init_state[:, 3]
        # roll_init = init_state[:, 4]
        # roll_dot_init = init_state[:, 5]
        

        neural_output_batch = self.mlp(inp_norm)

        
        lamda_samples = neural_output_batch[:, 0:3*self.nvar  ].to(device = device)  
        c_samples = neural_output_batch[:, 3 *self.nvar:4 *self.nvar ].to(device = device)



        # print(lamda_v)
        # print(c_v_samples)

        # s_v = s_samples[:, 0 : self.num_constraint ]
        # s_pitch = s_samples[:, self.num_constraint : 2*self.num_constraint ]
        # s_roll = s_samples[:, 2*self.num_constraint : 3*self.num_constraint ]

        b_eq = self.compute_boundary_vec(init_state)


        # print(b_eq_v)

        c_samples, accumulated_res_fixed_point, accumulated_res_primal, accumulated_res_primal_temp, accumulated_res_fixed_point_temp = self.custom_forward(lamda_samples, c_samples_input, c_samples, b_eq)
        
        return c_samples, accumulated_res_fixed_point, accumulated_res_primal, accumulated_res_primal_temp, accumulated_res_fixed_point_temp


    def mlp_loss(self, accumulated_res_primal, accumulated_res_fixed_point, c_samples, c_samples_input):		
        # Aug loss
        primal_loss = 0.5 * (torch.mean(accumulated_res_primal))
        fixed_point_loss = 0.5 * (torch.mean(accumulated_res_fixed_point  ))


        proj_loss = self.rcl_loss(c_samples, c_samples_input)

        # acc_loss = 0.5 * (torch.mean(predict_acc))

        loss = primal_loss+fixed_point_loss+0.1*proj_loss

        # loss = fixed_point_loss+primal_loss

        return primal_loss, fixed_point_loss, loss


    def forward(self,  inp, init_state, c_samples_input):
        
        
        # Normalize input
        inp_norm = (inp - self.inp_mean) / self.inp_std

        # Decode y
        c_samples, accumulated_res_fixed_point, accumulated_res_primal, accumulated_res_primal_temp, accumulated_res_fixed_point_temp = self.decoder_function( inp_norm, init_state, c_samples_input)
        
            
        return c_samples, accumulated_res_fixed_point, accumulated_res_primal, accumulated_res_primal_temp, accumulated_res_fixed_point_temp

	
		
  
							   
								
   
   
		

	 
	

  
