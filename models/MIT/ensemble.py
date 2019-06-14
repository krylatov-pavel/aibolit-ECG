import torch
import torch.nn as nn
import torch.nn.functional as f

class Ensemble(nn.Module):
    def __init__(self, model0, model1, model2, model3):
        super(Ensemble, self).__init__()
        self.model0 = model0
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, x):
        pred0 = self.model0(x)
        pred1 = self.model1(x)
        pred2 = self.model2(x)
        pred3 = self.model3(x)

        res = torch.add(pred0, pred1)
        res = torch.add(res, pred2)
        res = torch.add(res, pred3)

        res = f.softmax(res, dim=1)

        return res