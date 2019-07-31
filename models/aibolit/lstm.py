import torch
import torch.nn as nn
from utils.data_shape_1d import DataShape1d

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()

        class_num = len(config.dataset.params["class_settings"])
        input_size = config.dataset.params["example_duration"] * config.dataset.params["example_fs"]
        
        filters_num = 44
        hidden_size = 256
        kernel_size = 9
        
        shape = DataShape1d(1, input_size)

        self.conv_layers = nn.ModuleList()

        #layer1-3
        for _ in range(1, 3):  
            conv = nn.Conv1d(shape.channels, filters_num, kernel_size)
            shape.conv(filters_num, kernel_size)
            self.conv_layers.append(conv)
            self.conv_layers.append(nn.BatchNorm1d(filters_num))
            self.conv_layers.append(nn.ReLU())

            pad = shape.suggest_padding(2, 2)
            shape.pool(2, 2, pad)
            if pad:
                self.conv_layers.append(nn.ConstantPad1d(pad, 0))
            self.conv_layers.append(nn.MaxPool1d(2, 2))

            filters_num *= 2
        
        self.lstm = nn.LSTM(shape.channels, hidden_size)
        self.fc = nn.Linear(hidden_size, class_num)
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.permute(2, 0, 1)

        out, _ = self.lstm(x)
        out = out[-1]
        out = self.fc(out)

        return out