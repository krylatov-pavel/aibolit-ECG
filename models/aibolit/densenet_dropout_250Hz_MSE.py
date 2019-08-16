import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from utils.data_shape_1d import DataShape1d

bn_epsilon = 2e-05

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        #data params
        class_num = len(config.dataset.params["class_settings"])
        input_size = config.dataset.params["example_duration"] * config.dataset.params["example_fs"]

        #hparams
        growth_rate = int(config.model.hparams["growth_rate"])
        kernel_size = int(config.model.hparams["kernel_size"])
        compression = config.model.hparams["compression"]
        depth = int(config.model.hparams.get("depth"))
        dropout = config.model.hparams.get("dropout")
        bottleneck = config.model.hparams.get("bottleneck") or False

        #layers
        shape = DataShape1d(1, input_size)
        dense_depth = [depth] * 3
        self.layers = nn.ModuleList()
                
        #initial convolution
        filters = 2 * growth_rate
        padding = (kernel_size - 1) // 2
        self.layers.append(nn.Conv1d(1, filters, kernel_size, padding=padding, bias=False))
        self.layers.append(nn.BatchNorm1d(filters, eps=bn_epsilon))
        self.layers.append(nn.ReLU())
        shape.conv(filters, kernel_size, padding=(padding, padding))

        padding = shape.suggest_padding(2, 2)
        if padding:
            self.layers.append(nn.ConstantPad1d(padding, 0))    
        self.layers.append(nn.MaxPool1d(2, 2))
        shape.pool(2, 2, padding)
        
        #dense blocks + transition layers
        last_idx = len(dense_depth) - 1
        for i, depth in enumerate(dense_depth):
            self.layers.append(DenseBlock(depth, shape.channels, growth_rate,
                kernel_size=kernel_size, dropout=dropout, bottlenek=bottleneck))
            shape.dense_block(depth, growth_rate)

            if i != last_idx:
                output_channels = math.floor(compression * shape.channels)
                padding = shape.suggest_padding(2, 2)
                self.layers.append(TransitionLayer(shape.channels, output_channels, pool_pad=padding))
                shape.transition_layer(output_channels, padding)
        
        #final pool
        self.layers.append(nn.BatchNorm1d(shape.channels, eps=bn_epsilon))
        self.layers.append(nn.ReLU())
        pool_size = shape.length
        self.layers.append(nn.AvgPool1d(pool_size, pool_size))
        shape.pool(pool_size, pool_size)

        self.flatten_size = shape.size
        self.fc = nn.Linear(shape.size, class_num - 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = x.view(-1, self.flatten_size)

        x = self.fc(x)
        x = torch.sigmoid(x)

        return x

class BottleneckLayer(nn.Module):
    def __init__(self, input_channels, growth_rate, kernel_size=9, dropout=0):
        super(BottleneckLayer, self).__init__()

        intermediate_channels = 4 * growth_rate
        padding = (kernel_size - 1) // 2
        
        self.conv1_bn = nn.BatchNorm1d(input_channels, eps=bn_epsilon)
        self.conv1 = nn.Conv1d(input_channels, intermediate_channels, 1, bias=False)
        self.conv1_drop = nn.Dropout(dropout) if dropout else None

        self.conv2_bn = nn.BatchNorm1d(intermediate_channels, eps=bn_epsilon)
        self.conv2 = nn.Conv1d(intermediate_channels, growth_rate, kernel_size, padding=padding, bias=False)
        self.conv2_drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        out = self.conv1_bn(x)
        out = F.relu(out)
        out = self.conv1(out)
        if self.conv1_drop:
            out = self.conv1_drop(out)

        out = self.conv2_bn(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.conv2_drop:
            out = self.conv2_drop(out)

        out = torch.cat((x, out), 1)

        return out

class SingleLayer(nn.Module):
    def __init__(self, input_channels, growth_rate, kernel_size=9, dropout=0):
        super(SingleLayer, self).__init__()

        padding = (kernel_size - 1) // 2
        
        self.conv_bn = nn.BatchNorm1d(input_channels, eps=bn_epsilon)
        self.conv = nn.Conv1d(input_channels, growth_rate, kernel_size, padding=padding, bias=False)
        self.conv_drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        out = self.conv_bn(x)
        out = F.relu(out)
        out = self.conv(out)
        if self.conv_drop:
            out = self.conv_drop(out)

        out = torch.cat((x, out), 1)

        return out

class DenseBlock(nn.Module):
    def __init__(self, depth, input_channels, growth_rate, kernel_size=9, dropout=0, bottlenek=False):
        super(DenseBlock, self).__init__()

        self.layers = nn.ModuleList()

        for _ in range(depth):
            if bottlenek:
                self.layers.append(BottleneckLayer(input_channels, growth_rate, kernel_size, dropout))
            else:
                self.layers.append(SingleLayer(input_channels, growth_rate, kernel_size, dropout))
            input_channels += growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, input_channels, out_channels, pool_pad=None):
        super(TransitionLayer, self).__init__()

        self.bn = nn.BatchNorm1d(input_channels, eps=bn_epsilon)
        self.conv = nn.Conv1d(input_channels, out_channels, 1, bias=False)
        self.pool_pad = nn.ConstantPad1d(pool_pad, 0) if pool_pad else None
        self.pool = nn.AvgPool1d(2, 2)
    
    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        if self.pool_pad:
            x = self.pool_pad(x)
        x = self.pool(x)

        return x   