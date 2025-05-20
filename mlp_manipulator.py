import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        
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
        return self.mlp(x)


class MLPProjectionFilter(nn.Module):
    
    def __init__(self, P, Pdot, Pddot, mlp, num_batch, inp_mean, inp_std, t_fin):
        super(MLPProjectionFilter, self).__init__()
        
        # MLP Model
        self.mlp = mlp
        
        # Constraint parameters
        self.v_max = 0.8
        self.a_max = 1.8
        self.p_max = 180 * np.pi/180
        
        # P Matrices
        self.P = P.to(device)
        self.Pdot = Pdot.to(device)
        self.Pddot = Pddot.to(device)
        
        # Dimensions
        self.nvar_single = P.size(dim=1)
        self.num = P.size(dim=0)
        self.num_batch = num_batch
        self.num_dof = 6
        self.nvar = self.num_dof * self.nvar_single
        
        # Projection parameters
        self.A_projection = torch.eye(self.nvar, device=device)
        self.rho_ineq = 1.0
        self.rho_projection = 1.0
        
        # Compute constraint matrices
        self.A_v_ineq, self.A_v = self._get_A_v()
        self.A_a_ineq, self.A_a = self._get_A_a()
        self.A_p_ineq, self.A_p = self._get_A_p()
        self.A_eq = self._get_A_eq()
        self.Q_inv = self._get_Q_inv()
        
        # Trajectory matrices
        self.A_theta, self.A_thetadot, self.A_thetaddot = self._get_A_traj()
        
        # Solver parameters
        self.maxiter = 15
        self.t_fin = t_fin
        self.tot_time = torch.linspace(0, t_fin, self.num, device=device)
        
        # Normalization parameters
        self.inp_mean = inp_mean
        self.inp_std = inp_std
        
        # Loss function
        self.rcl_loss = nn.MSELoss()

    def _get_A_traj(self):
        """Compute trajectory matrices"""
        I = torch.eye(self.num_dof, device=self.P.device)
        A_theta = torch.kron(I, self.P)
        A_thetadot = torch.kron(I, self.Pdot)
        A_thetaddot = torch.kron(I, self.Pddot)
        return A_theta, A_thetadot, A_thetaddot

    def _get_A_p(self):
        """Compute position constraint matrices"""
        A_p = torch.vstack((self.P, -self.P))
        A_p_ineq = torch.kron(torch.eye(self.num_dof, device=self.P.device), A_p)
        return A_p_ineq, A_p

    def _get_A_v(self):
        """Compute velocity constraint matrices"""
        A_v = torch.vstack((self.Pdot, -self.Pdot))
        A_v_ineq = torch.kron(torch.eye(self.num_dof, device=self.P.device), A_v)
        return A_v_ineq, A_v

    def _get_A_a(self):
        """Compute acceleration constraint matrices"""
        A_a = torch.vstack((self.Pddot, -self.Pddot))
        A_a_ineq = torch.kron(torch.eye(self.num_dof, device=self.P.device), A_a)
        return A_a_ineq, A_a

    def _get_A_eq(self):
        """Compute equality constraint matrices"""
        A_eq = torch.kron(torch.eye(self.num_dof, device=self.P.device), torch.vstack((
            self.P[0].unsqueeze(0),
            self.Pdot[0].unsqueeze(0),
            self.Pddot[0].unsqueeze(0),
            self.Pdot[-1].unsqueeze(0),
            self.Pddot[-1].unsqueeze(0)
        )))
        return A_eq

    def _get_Q_inv(self):
        """Compute inverse of Q matrix for the KKT system"""
        Q_top_left = (
            self.A_projection.T @ self.A_projection +
            self.rho_ineq * (self.A_v_ineq.T @ self.A_v_ineq) +
            self.rho_ineq * (self.A_a_ineq.T @ self.A_a_ineq) +
            self.rho_ineq * (self.A_p_ineq.T @ self.A_p_ineq)
        )
        Q_top = torch.hstack((Q_top_left, self.A_eq.T))
        Q_bottom = torch.hstack((self.A_eq, torch.zeros((self.A_eq.shape[0], self.A_eq.shape[0]), device=self.A_eq.device)))
        Q = torch.vstack((Q_top, Q_bottom))
        return torch.inverse(Q)

    def compute_boundary_vec(self, state_term):
        """Compute boundary condition vector"""
        return state_term

    def compute_feasible_control(self, lamda_v, lamda_a, lamda_p, s_v, s_a, s_p, b_eq_term, xi_samples):
        """Compute feasible control via optimization"""
        
        
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

        # Set bounds
        b_v = v_max_vec
        b_a = a_max_vec
        b_p = p_max_vec

        # Augmented bounds
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

        # Update slack variables
        s_v = torch.maximum(
            torch.zeros(self.num_batch, 2*self.num*self.num_dof, device=device),
            -torch.matmul(self.A_v_ineq, primal_sol.T).T + b_v
        )
        res_v = torch.matmul(self.A_v_ineq, primal_sol.T).T - b_v + s_v

        s_a = torch.maximum(
            torch.zeros(self.num_batch, 2*self.num*self.num_dof, device=device),
            -torch.matmul(self.A_a_ineq, primal_sol.T).T + b_a
        )
        res_a = torch.matmul(self.A_a_ineq, primal_sol.T).T - b_a + s_a

        s_p = torch.maximum(
            torch.zeros(self.num_batch, 2*self.num*self.num_dof, device=device),
            -torch.matmul(self.A_p_ineq, primal_sol.T).T + b_p
        )
        res_p = torch.matmul(self.A_p_ineq, primal_sol.T).T - b_p + s_p

        # Update dual variables
        lamda_v = lamda_v - self.rho_ineq * torch.matmul(self.A_v_ineq.T, res_v.T).T
        lamda_a = lamda_a - self.rho_ineq * torch.matmul(self.A_a_ineq.T, res_a.T).T
        lamda_p = lamda_p - self.rho_ineq * torch.matmul(self.A_p_ineq.T, res_p.T).T

        # Compute residual norms
        res_v = torch.norm(res_v, dim=1)
        res_a = torch.norm(res_a, dim=1)
        res_p = torch.norm(res_p, dim=1)
        res_projection = res_v + res_a + res_p

        return primal_sol, s_v, s_a, s_p, lamda_v, lamda_a, lamda_p, res_projection

    def initialize_slack_variables(self, primal_sol):
        """Initialize slack variables based on primal solution"""
        device = primal_sol.device
        
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
        
        # Initialize slack variables
        s_v = torch.maximum(
            torch.zeros(self.num_batch, 2*self.num*self.num_dof, device=device),
            -torch.matmul(self.A_v_ineq, primal_sol.T).T + b_v
        )
        
        s_a = torch.maximum(
            torch.zeros(self.num_batch, 2*self.num*self.num_dof, device=device),
            -torch.matmul(self.A_a_ineq, primal_sol.T).T + b_a
        )
        
        s_p = torch.maximum(
            torch.zeros(self.num_batch, 2*self.num*self.num_dof, device=device),
            -torch.matmul(self.A_p_ineq, primal_sol.T).T + b_p
        )
        
        return s_v, s_a, s_p

    def custom_forward(self, lamda_samples, c_samples_input, c_samples, b_eq):
        """Custom forward pass with optimization loop"""
        # Initialize slack variables
        s_v, s_a, s_p = self.initialize_slack_variables(c_samples)

        # Extract dual variables
        lamda_v = lamda_samples[:, 0:self.nvar].to(device)    
        lamda_a = lamda_samples[:, self.nvar:2*self.nvar].to(device)   
        lamda_p = lamda_samples[:, 2*self.nvar:3*self.nvar].to(device)
        
        # Track residuals
        accumulated_res_primal = []
        accumulated_res_fixed_point = []

        # Optimization loop
        for _ in range(self.maxiter):
            # Store previous values
            c_samples_prev = c_samples.clone()
            lamda_v_prev = lamda_v.clone() 
            lamda_a_prev = lamda_a.clone()
            lamda_p_prev = lamda_p.clone()
            s_v_prev = s_v.clone() 
            s_a_prev = s_a.clone()
            s_p_prev = s_p.clone()

            # Compute feasible control
            c_samples, s_v, s_a, s_p, lamda_v, lamda_a, lamda_p, res_projection = self.compute_feasible_control(
                lamda_v, lamda_a, lamda_p, s_v, s_a, s_p, b_eq, c_samples
            )
            
            # Track residuals
            accumulated_res_primal.append(res_projection)

            # Compute fixed point residual
            fixed_point_res = (
                torch.linalg.norm(lamda_v - lamda_v_prev, dim=1) +
                torch.linalg.norm(lamda_a - lamda_a_prev, dim=1) + 
                torch.linalg.norm(lamda_p - lamda_p_prev, dim=1) + 
                torch.linalg.norm(s_v - s_v_prev, dim=1) + 
                torch.linalg.norm(s_a - s_a_prev, dim=1) + 
                torch.linalg.norm(s_p - s_p_prev, dim=1) +
                torch.linalg.norm(c_samples - c_samples_prev, dim=1)
            )
                                
            accumulated_res_fixed_point.append(fixed_point_res)

        # Compute average residuals
        res_primal_stack = torch.stack(accumulated_res_primal)
        res_fixed_stack = torch.stack(accumulated_res_fixed_point)
        avg_res_primal = torch.sum(res_primal_stack, axis=0) / self.maxiter
        avg_res_fixed_point = torch.sum(res_fixed_stack, axis=0) / self.maxiter

        return c_samples, avg_res_fixed_point, avg_res_primal, accumulated_res_primal, accumulated_res_fixed_point

    def decoder_function(self, inp_norm, init_state, c_samples_input):
        """Decoder function to compute control from normalized input"""
        # Get neural network output
        neural_output_batch = self.mlp(inp_norm)
        
        # Extract components from output
        lamda_samples = neural_output_batch[:, 0:3*self.nvar].to(device)  
        c_samples = neural_output_batch[:, 3*self.nvar:4*self.nvar].to(device)

        # Compute boundary conditions
        b_eq = self.compute_boundary_vec(init_state)

        # Run optimization
        c_samples, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history = self.custom_forward(
            lamda_samples, c_samples_input, c_samples, b_eq
        )
        
        return c_samples, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history

    def mlp_loss(self, avg_res_primal, avg_res_fixed_point, c_samples, c_samples_input):
        """Compute loss for optimization"""
        # Component losses
        primal_loss = 0.5 * torch.mean(avg_res_primal)
        fixed_point_loss = 0.5 * torch.mean(avg_res_fixed_point)
        proj_loss = self.rcl_loss(c_samples, c_samples_input)

        # Total loss
        loss = primal_loss + fixed_point_loss + 0.1 * proj_loss

        return primal_loss, fixed_point_loss, loss

    def forward(self, inp, init_state, c_samples_input):
        """Forward pass through the model"""
        # Normalize input
        inp_norm = (inp - self.inp_mean) / self.inp_std

        # Decode input to get control
        c_samples, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history = self.decoder_function(
            inp_norm, init_state, c_samples_input
        )
            
        return c_samples, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history