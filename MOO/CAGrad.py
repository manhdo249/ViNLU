import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
from scipy.optimize import minimize

class CAGrad:
    def __init__(self, calpha=0.5, rescale=1, device='cuda'):
        self.calpha = calpha
        self.rescale = rescale
        self.device = device

    def apply(self, components):
        task_num = len(components)
        grads = torch.stack(components)  
        grads = grads.to(self.device)

        GG = torch.matmul(grads, grads.t()).cpu()  
        g0_norm = (GG.mean() + 1e-8).sqrt()  

        x_start = np.ones(task_num) / task_num
        bounds = [(0, 1) for _ in x_start]
        constraints = {'type': 'eq', 'fun': lambda x: 1 - sum(x)}
        A = GG.numpy()
        c = (self.calpha * g0_norm + 1e-8).item()

        def objfn(x):
            x = x.reshape(-1, 1)
            xT = x.T
            xAx = xT @ A @ x
            obj = (xT @ A @ x_start.reshape(-1, 1) + c * np.sqrt(xAx + 1e-8)).sum()
            return obj

        res = minimize(objfn, x_start, method='SLSQP', bounds=bounds, constraints=constraints)
        if not res.success:
            raise ValueError(f"Optimization failed: {res.message}")

        alpha = torch.tensor(res.x, device=self.device, dtype=grads.dtype)
    
        gw = (grads * alpha.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(0) + lmbda * gw

        if self.rescale == 0:
            new_grads = g
        elif self.rescale == 1:
            new_grads = g / (1 + self.calpha ** 2)
        elif self.rescale == 2:
            new_grads = g / (1 + self.calpha)
        else:
            raise ValueError(f'No support rescale type {self.rescale}')

        adjusted_grad = new_grads

        return adjusted_grad, alpha