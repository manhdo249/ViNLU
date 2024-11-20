import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
from scipy.optimize import minimize

class Gram_Schmidt:
    def __init__(self, model, device, buffer_size=3):
        self.model = model
        self.device = device
        self.buffer_size = buffer_size
        self.grad_buffer = []  
        self.grad_dim = self._get_grad_dim()  
    
    def _get_grad_dim(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _get_total_grad(self, total_loss):
        self.model.zero_grad()
        total_loss.backward(retain_graph=True)
        total_grad = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None])
        return total_grad.detach()
    
    def update_grad_buffer(self, total_grad):
        if len(self.grad_buffer) >= self.buffer_size:
            self.grad_buffer.pop(0)
        self.grad_buffer.append(total_grad)
    
    def decompose_grad(self, total_grad):
        if len(self.grad_buffer) == 0:
            return [total_grad]
        else:
            orthonormal_basis = []
            for v in self.grad_buffer:
                w = v.clone()
                for u in orthonormal_basis:
                    projection = torch.dot(w, u) * u
                    w = w - projection
                norm_w = w.norm()
                if norm_w > 1e-8:
                    w = w / norm_w
                    orthonormal_basis.append(w)
            components = []
            residual = total_grad.clone()
            for u in orthonormal_basis:
                coeff = torch.dot(residual, u)
                component = coeff * u
                components.append(component)
                residual = residual - component
            components.append(residual)
            return components