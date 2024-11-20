import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
from scipy.optimize import minimize

class DB_MTL:
    def __init__(self, beta=0.5, beta_sigma=0.5):
        self.beta = beta
        self.beta_sigma = beta_sigma
        self.step = 0
        self.grad_buffer = None  

    def apply(self, components):
        self.step += 1
        components_ = torch.stack(components) 

        if self.grad_buffer is None or self.grad_buffer.shape != components_.shape:
            self.grad_buffer = torch.zeros_like(components_)

        self.grad_buffer = components_ + (self.beta / self.step ** self.beta_sigma) * (self.grad_buffer - components_)

        u_grad = self.grad_buffer.norm(dim=-1)  
        alpha = u_grad.max() / (u_grad + 1e-8) 

        adjusted_grad = torch.sum(alpha.view(-1, 1) * self.grad_buffer, dim=0)

        return adjusted_grad, alpha