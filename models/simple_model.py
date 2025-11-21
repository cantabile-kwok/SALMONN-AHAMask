import torch.nn as nn
import torch

class SimpleTensorModel(nn.Module):
    def __init__(self, tensor_length=1056):
        super().__init__()
        self.tensor = nn.Parameter(torch.zeros(tensor_length))
        nn.init.normal_(self.tensor, mean=4, std=0.02)

    def forward(self, x=None):
        return self.tensor