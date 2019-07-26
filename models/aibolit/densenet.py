import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from utils.data_shape_1d import DataShape1d

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

        #layers
        shape = DataShape1d(1, input_size)
        
        padding = shape.suggest_padding(kernel_size, 2, output_length=shape.shape[1] // 2)
        self.initial_pad = nn.ConstantPad1d(padding, 0)
        self.initial_conv = nn.Conv1d(1, 2 * growth_rate, kernel_size, stride=2, bias=False)
        shape.conv(2 * growth_rate, kernel_size, stride=2, padding=padding)
        self.initial_bn = nn.BatchNorm1d(2 * growth_rate)
        
        self.initial_pool = nn.MaxPool1d(2, 2)
        shape.pool(2, 2)

        self.dense_1 = DenseBlock(6, shape.channels, growth_rate, kernel_size)
        shape.dense_block(6, growth_rate)
        
        output_channels = math.floor(compression * shape.channels)
        pad = shape.suggest_padding(2, 2)
        self.transition_1 = TransitionLayer(shape.channels, output_channels, pool_pad=pad)
        shape.transition_layer(output_channels, pad)

        self.dense_2 = DenseBlock(12, shape.channels, growth_rate, kernel_size)
        shape.dense_block(12, growth_rate)

        output_channels = math.floor(compression * shape.channels)
        pad = shape.suggest_padding(2, 2)
        self.transition_2 = TransitionLayer(shape.channels, output_channels, pool_pad=pad)
        shape.transition_layer(output_channels, pad)

        self.dense_3 = DenseBlock(8, shape.channels, growth_rate, kernel_size)
        shape.dense_block(8, growth_rate)
        
        self.final_bn = nn.BatchNorm1d(shape.channels)
        pool_size = math.ceil(shape.shape[1] / 2)
        pad = shape.suggest_padding(pool_size, pool_size)
        self.global_avg_pool_pad = nn.ConstantPad1d(pad, 0) if pad else None
        self.global_avg_pool = nn.AvgPool1d(pool_size, pool_size)
        shape.pool(pool_size, pool_size, pad)

        self.flatten_size = shape.size
        self.fc = nn.Linear(shape.size, class_num)

    def forward(self, x):
        x = self.initial_pad(x)
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x)
        x = self.initial_pool(x)

        x = self.dense_1(x)
        x = self.transition_1(x)

        x = self.dense_2(x)
        x = self.transition_2(x)

        x = self.dense_3(x)

        x = self.final_bn(x)
        x = F.relu(x)
        if self.global_avg_pool_pad:
            x = self.global_avg_pool_pad(x)
        x = self.global_avg_pool(x)

        x = x.view(-1, self.flatten_size)

        x = self.fc(x)

        return x

class BottleneckLayer(nn.Module):
    def __init__(self, input_channels, growth_rate, kernel_size=9):
        super(BottleneckLayer, self).__init__()

        intermediate_channels = 4 * growth_rate
        padding = (kernel_size - 1) // 2
        
        self.conv1_bn = nn.BatchNorm1d(input_channels)
        self.conv1 = nn.Conv1d(input_channels, intermediate_channels, 1, bias=False)

        self.conv2_bn = nn.BatchNorm1d(intermediate_channels)
        self.conv2 = nn.Conv1d(intermediate_channels, growth_rate, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        out = self.conv1_bn(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.conv2_bn(out)
        out = F.relu(out)
        out = self.conv2(out)

        out = torch.cat((x, out), 1)

        return out

class DenseBlock(nn.Module):
    def __init__(self, depth, input_channels, growth_rate, kernel_size=9):
        super(DenseBlock, self).__init__()

        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(BottleneckLayer(input_channels, growth_rate, kernel_size))
            input_channels += growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, input_channels, out_channels, pool_pad=None):
        super(TransitionLayer, self).__init__()

        self.bn = nn.BatchNorm1d(input_channels)
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