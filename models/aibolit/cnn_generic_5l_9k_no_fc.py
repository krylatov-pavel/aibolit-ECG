
import torch.nn as nn
import math
import torch.functional as F
import utils.helpers as helpers
from utils.data_shape_1d import DataShape1d

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        class_num = len(config.dataset.params["class_settings"])
        input_size = config.dataset.params["example_duration"] * config.dataset.params["example_fs"]

        filters_num = int(config.model.hparams["filters_num"])
        dropout = config.model.hparams["dropout"]
        activation = helpers.get_class(config.model.hparams["activation_fn"])
        
        shape = DataShape1d(1, input_size)
        kernel_size = 9

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        #layer1-5
        for _ in range(1, 6):  
            conv = nn.Conv1d(shape.channels, filters_num, kernel_size)
            shape.conv(filters_num, kernel_size)
            self.conv_layers.append(conv)
            self.conv_layers.append(nn.BatchNorm1d(filters_num))
            self.conv_layers.append(activation())

            pad = shape.suggest_padding(2, 2)
            shape.pool(2, 2, pad)
            if pad:
                self.conv_layers.append(nn.ConstantPad1d(pad, 0))
            self.conv_layers.append(nn.MaxPool1d(2, 2))

            filters_num *= 2

        pool_size = math.ceil(shape.shape[1] / 2)

        pad = shape.suggest_padding(pool_size, pool_size)
        shape.pool(pool_size, pool_size, pad)
        if pad:
            self.conv_layers.append(nn.ConstantPad1d(pad, 0))
        self.conv_layers.append(nn.AvgPool1d(pool_size, pool_size)) #2

        self.flatten_size = shape.size

        self.fc_layers.append(nn.Linear(self.flatten_size, class_num))
        self.fc_layers.append(nn.Dropout(dropout))
        shape.fc(class_num)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(-1, self.flatten_size)

        for layer in self.fc_layers:
            x = layer(x)

        return x