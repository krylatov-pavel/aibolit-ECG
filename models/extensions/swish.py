import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()