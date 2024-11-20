import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
from scipy.optimize import minimize

class PCGrad:
    def __init__(self):
        pass

    def apply(self, components):
        task_num = len(components)
        pc_grads = [comp.clone() for comp in components]
        grads = components  

        batch_weight = torch.ones(task_num, device=components[0].device)
        
        for tn_i in range(task_num):
            task_indices = list(range(task_num))
            random.shuffle(task_indices)
            for tn_j in task_indices:
                if tn_i != tn_j:
                    g_ij = torch.dot(pc_grads[tn_i], grads[tn_j])
                    if g_ij < 0:
                        adjustment = (g_ij / (grads[tn_j].norm().pow(2) + 1e-8)) * grads[tn_j]
                        pc_grads[tn_i] -= adjustment
                        batch_weight[tn_j] -= (g_ij / (grads[tn_j].norm().pow(2) + 1e-8))

        adjusted_grad = torch.stack(pc_grads).sum(dim=0)

        return adjusted_grad, batch_weight