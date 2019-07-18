import torch.nn as nn
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
        fc_units = int(config.model.hparams["fc_units"])
        activation = helpers.get_class(config.model.hparams["activation_fn"])

        fc_layers = 3
        kernel_size = 9
        
        shape = DataShape1d(1, input_size)

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        #layer1
        conv = nn.Conv1d(shape.channels, filters_num, kernel_size)
        shape.conv(filters_num, kernel_size)
        self.conv_layers.append(conv)
        self.conv_layers.append(nn.BatchNorm1d(filters_num))
        self.conv_layers.append(activation())

        #layer2-6
        for _ in range(4):  
            conv = nn.Conv1d(shape.channels, filters_num, kernel_size)
            shape.conv(filters_num, kernel_size)
            self.conv_layers.append(conv)
            self.conv_layers.append(nn.BatchNorm1d(filters_num))
            self.conv_layers.append(activation())

            pad = shape.suggest_padding(3, 3)
            shape.pool(3, 3, pad)
            if pad:
                self.conv_layers.append(nn.ConstantPad1d(pad, 0))
            self.conv_layers.append(nn.MaxPool1d(3, 3))

            filters_num *= 2

        conv = nn.Conv1d(shape.channels, filters_num, kernel_size)
        shape.conv(filters_num, kernel_size)
        self.conv_layers.append(conv)
        self.conv_layers.append(nn.BatchNorm1d(filters_num))
        self.conv_layers.append(activation())   #71

        pad = (1,0)
        shape.pool(18, 18, pad)
        self.conv_layers.append(nn.ConstantPad1d(pad, 0))
        self.conv_layers.append(nn.AvgPool1d(18, 18)) #2

        self.flatten_size = shape.size

        self.fc_layers.append(nn.Linear(self.flatten_size, fc_units))
        self.fc_layers.append(nn.BatchNorm1d(fc_units))
        self.fc_layers.append(nn.ReLU())
        self.fc_layers.append(nn.Dropout(dropout))
        shape.fc(fc_units)

        for i in range(max(fc_layers - 2, 0)):
            self.fc_layers.append(nn.Linear(fc_units, fc_units))
            self.fc_layers.append(nn.BatchNorm1d(fc_units))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            shape.fc(fc_units)

        self.fc_layers.append(nn.Linear(shape.size, class_num))
        shape.fc(class_num) 

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(-1, self.flatten_size)

        for layer in self.fc_layers:
            x = layer(x)

        return x