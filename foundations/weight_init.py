import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std=math.sqrt(2.0/(fan_in+fan_out))
        W=torch.randn(fan_out,fan_in)*std
        return [[round(float(val),4) for val in row] for row in W]

        pass

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std=math.sqrt(2.0/fan_in)
        W=torch.randn(fan_out,fan_in)*std

        return [[round(float(val),4) for val in row] for row in W]
        
        pass

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        weights=[]
        for i in range(num_layers):
            fan_in = input_dim if i==0 else hidden_dim
            fan_out=hidden_dim
            if init_type =="xavier":
                std=math.sqrt(2.0/(fan_in+fan_out))
                W= torch.randn(fan_out,fan_in)*std
                #x= torch.tanh(torch.matmul(W,x))
            elif init_type=="kaiming":
                std=math.sqrt(2.0/fan_in)
                W=torch.randn(fan_out,fan_in)*std
                #x=torch.relu(torch.matmul(W,x))
            else:
                W=torch.randn(fan_out,fan_in)
                #x=torch.relu(torch.matmul(W,x))
            weights.append(W)
        x= torch.randn(input_dim)
        stds=[]
        for i in range(num_layers):
            if init_type=="xavier":
                x=torch.tanh(torch.matmul(weights[i],x))
            else:
                x=torch.relu(torch.matmul(weights[i],x))
            stds.append(round(float(x.std()),2))
        return stds
        pass

