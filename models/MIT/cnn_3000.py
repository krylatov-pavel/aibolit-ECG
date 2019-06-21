import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        class_num = len(config.dataset.params["label_map"])
        input_size = config.dataset.params["slice_window"]

        self.conv1 = nn.Conv1d(1, 32, 7)
        self.pool1 = nn.MaxPool1d(3, 3)
        self.conv2 = nn.Conv1d(32, 64, 7)
        self.pool2 = nn.AvgPool1d(8, 8)
        self.fc1 = nn.Linear(7936, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num = 1
        for s in size:
            num *= s
        return num