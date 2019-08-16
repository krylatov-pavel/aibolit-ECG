import torch
import torch.nn as nn

def create(classifier_type, classifier_params):
    classifier_type = classifier_type or "index"
    classifier_params = classifier_params or {}

    if classifier_type == "index":
        return IndexClassifier(**classifier_params)
    if classifier_type == "oh_zero_based":
        return OHZeroBasedClassifier(**classifier_params)
    else:
        raise ValueError("Unknown classifier type")

def to_oh_zero_base(y, class_num):
    batch_size = y.shape[0]
    oh = torch.zeros(y.shape[0], class_num - 1, device=y.device)
    for i in range(batch_size):
        if y[i] > 0:
            oh[i][y[i] - 1] = 1

    return oh

class IndexClassifier(nn.Module):
    def __init__(self):
        super(IndexClassifier, self).__init__()

    def forward(self, x):
        _, predictions = torch.max(x, 1)

        return predictions

class OHZeroBasedClassifier(nn.Module):
    def __init__(self, label_map, threshold=None):
        super(OHZeroBasedClassifier, self).__init__()

        self.class_num = len(label_map)
        self.threshold = threshold or 0.24
        self.classes = to_oh_zero_base(torch.tensor([idx for key, idx in label_map.items() if idx], dtype=torch.long), self.class_num)

    def forward(self, x):
        self.classes = self.classes.to(x.device)

        predictions = x.unsqueeze(dim=1)
        predictions = predictions.repeat_interleave(self.class_num - 1, dim=1)
        predictions = torch.abs(predictions - self.classes)
        predictions = torch.sum(predictions, dim=2)
        _, predictions = torch.min(predictions, dim=1)
        predictions += 1

        for i in range(x.shape[0]):
            if x[i].shape[0] == (x[i] < self.threshold).nonzero().shape[0]:
                predictions[i] = 0

        return predictions