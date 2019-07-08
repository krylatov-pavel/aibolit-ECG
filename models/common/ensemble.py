import torch
import torch.nn as nn
import torch.nn.functional as f

class Ensemble(nn.Module):
    def __init__(self, **kwargs):
        super(Ensemble, self).__init__()

        self.models = nn.ModuleList()

        for name, model in kwargs.items():
            self.models.add_module(name, model)

    def forward(self, x):
        i = 0
        for model in self.models:
            if i == 0:
                predictions = model(x)
            else:
                predictions = predictions.add(model(x))
            i += 1
        
        res = f.softmax(predictions, dim=1)

        return res