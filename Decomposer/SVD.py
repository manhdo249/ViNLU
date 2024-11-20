import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
from scipy.optimize import minimize

class SVD:
    def __init__(self, model, device, buffer_size=4):
        self.model = model
        self.device = device
        self.buffer_size = buffer_size
        self.grad_buffer = []
        self.grad_dim = self._get_grad_dim()
    
    def _get_grad_dim(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _get_total_grad(self, total_loss):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = None  
        total_loss.backward(retain_graph=True)
        total_grad = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None])
        return total_grad.to(self.device)
    
    def remove_grad_buffer(self):
        while len(self.grad_buffer) > 0:
            self.grad_buffer.pop(0)
    
    def update_grad_buffer(self, total_grad):
        #if len(self.grad_buffer) >= self.buffer_size:
        #    self.grad_buffer.pop(0)
        self.grad_buffer.append(total_grad.detach())
    
    def decompose_grad(self, total_grad):
        if len(self.grad_buffer) < 2:
            return [total_grad]
        else:
            grad_matrix = torch.stack(self.grad_buffer)  
            grad_mean = grad_matrix.mean(dim=0)
            grad_matrix_centered = grad_matrix - grad_mean
            U, S, Vh = torch.linalg.svd(grad_matrix_centered, full_matrices=False)
            V = Vh.T
            top_k = min(self.buffer_size, len(S))  
            top_k_indices = torch.argsort(S, descending=True)[:top_k]
            top_k_components = V[:, top_k_indices]
            components = []
            for i in range(top_k_components.shape[1]):
                pc = top_k_components[:, i]
                coeff = torch.dot(total_grad, pc)
                component = coeff * pc
                components.append(component)
            '''for i in range(V.shape[1]):
                pc = V[:, i]
                coeff = torch.dot(total_grad, pc)
                component = coeff * pc
                components.append(component)'''
            return components
        