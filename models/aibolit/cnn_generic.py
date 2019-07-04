import torch.nn as nn
import torch.functional as F
from utils.data_shape_1d import DataShape1d

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        class_num = len(config.dataset.params["label_map"])
        input_size = config.dataset.params["example_duration"] * config.dataset.params["resample_fs"]

        filters_num = config.model.hparams["filters_num"]
        filters_step = config.model.hparams["filters_step"]
        conv_kernel_size = config.model.hparams["conv_kernel_size"]
        conv_layer_num = config.model.hparams["conv_layer_num"]
        dropout = config.model.hparams["dropout"]
        fc_units = config.model.hparams["fc_units"]
        fc_layers = config.model.hparams["fc_layers"]

        shape = DataShape1d(1, input_size)

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        for i in range (1, conv_layer_num + 1):
            pad = shape.suggest_padding(conv_kernel_size, 1)
            if pad:
                self.conv_layers.append(nn.ConstantPad1d(pad, 0))

            conv = nn.Conv1d(shape.channels, filters_num, conv_kernel_size)
            shape.conv(filters_num, conv_kernel_size, padding=pad)
            self.conv_layers.append(conv)
            self.conv_layers.append(nn.BatchNorm1d(filters_num))
            self.conv_layers.append(nn.ReLU())

            if i % 2 == 0 and i != conv_layer_num:
                pad = shape.suggest_padding(3, 3)
                if pad:
                    self.conv_layers.append(nn.ConstantPad1d(pad, 0))
                shape.pool(3, 3, padding=pad)
                self.conv_layers.append(nn.MaxPool1d(3, 3))
                
                filters_num += filters_step

        pad = shape.suggest_padding(4, 4)
        if pad:
            self.conv_layers.append(nn.ConstantPad1d(pad, 0))
        shape.pool(4, 4, padding=pad)
        self.conv_layers.append(nn.AvgPool1d(4, 4))

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