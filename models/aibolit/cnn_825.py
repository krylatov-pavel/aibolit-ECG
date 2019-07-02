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
        dropout = config.model.hparams["dropout"]
        fc_units = config.model.hparams["fc_units"]

        shape = DataShape1d(1, input_size)

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv1d(1, filters_num, 9)) #817 
        self.conv_layers.append(nn.BatchNorm1d(filters_num))
        self.conv_layers.append(nn.ReLU())
        shape.conv(filters_num, 9)

        self.conv_layers.append(nn.Conv1d(filters_num, filters_num, 8)) #810 
        self.conv_layers.append(nn.BatchNorm1d(filters_num))
        self.conv_layers.append(nn.ReLU())
        shape.conv(filters_num, 8)

        self.conv_layers.append(nn.MaxPool1d(2, 2)) #405
        shape.pool(2, 2)

        self.conv_layers.append(nn.Conv1d(filters_num, filters_num + filters_step, 7)) #399
        filters_num = filters_num + filters_step
        self.conv_layers.append(nn.BatchNorm1d(filters_num))
        self.conv_layers.append(nn.ReLU())
        shape.conv(filters_num, 7)

        self.conv_layers.append(nn.Conv1d(filters_num, filters_num, 6)) #394
        self.conv_layers.append(nn.BatchNorm1d(filters_num))
        self.conv_layers.append(nn.ReLU())
        shape.conv(filters_num, 6)

        self.conv_layers.append(nn.MaxPool1d(2, 2)) #197
        shape.pool(2, 2)
        
        self.conv_layers.append(nn.Conv1d(filters_num, filters_num + filters_step, 5)) #193
        filters_num += filters_step
        self.conv_layers.append(nn.BatchNorm1d(filters_num))
        self.conv_layers.append(nn.ReLU())
        shape.conv(filters_num, 5)

        self.conv_layers.append(nn.ConstantPad1d((1, 0), 0))
        self.conv_layers.append(nn.MaxPool1d(2, 2)) #97
        shape.pool(2, 2, (1, 0))

        self.conv_layers.append(nn.Conv1d(filters_num, filters_num + filters_step, 4)) #94
        filters_num += filters_step
        self.conv_layers.append(nn.BatchNorm1d(filters_num))
        self.conv_layers.append(nn.ReLU())
        shape.conv(filters_num, 4)

        self.conv_layers.append(nn.MaxPool1d(2, 2)) #47
        shape.pool(2, 2)

        self.conv_layers.append(nn.Conv1d(filters_num, filters_num + filters_step, 3)) #93
        filters_num += filters_step
        self.conv_layers.append(nn.BatchNorm1d(filters_num))
        self.conv_layers.append(nn.ReLU())
        shape.conv(filters_num, 3)

        self.conv_layers.append(nn.AvgPool1d(7, 7))#13
        shape.pool(7, 7)

        self.flatten_size = shape.size

        self.fc_layers.append(nn.Linear(self.flatten_size, fc_units))
        self.fc_layers.append(nn.BatchNorm1d(fc_units))
        self.fc_layers.append(nn.ReLU())
        self.fc_layers.append(nn.Dropout(dropout))
        shape.fc(fc_units)

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