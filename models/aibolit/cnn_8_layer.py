import torch.nn as nn
import torch.nn.functional as F
from utils.data_shape_1d import DataShape1d

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        class_num = len(config.dataset.params["label_map"])
        input_size = config.dataset.params["example_duration"] * config.dataset.params["resample_fs"]

        shape = DataShape1d(1, input_size)

        filters_num = config.model.hparams["filters_num"]
        filters_step = config.model.hparams["filters_step"]
        kernel_size = 7
        
        dropout = config.model.hparams["dropout"]
        fc_units = config.model.hparams["fc_units"]

        self.conv1 = nn.Conv1d(1, filters_num, kernel_size)
        self.conv1_bn = nn.BatchNorm1d(filters_num)
        shape.conv(filters_num, kernel_size)

        self.conv2 = nn.Conv1d(filters_num, filters_num, kernel_size)
        self.conv2_bn = nn.BatchNorm1d(filters_num)
        shape.conv(filters_num, kernel_size)

        self.maxpool1 = nn.MaxPool1d(2, 2)
        shape.pool(2, 2)

        self.conv3 = nn.Conv1d(filters_num, filters_num + filters_step, kernel_size)
        filters_num += filters_step
        self.conv3_bn = nn.BatchNorm1d(filters_num)
        shape.conv(filters_num, kernel_size)

        self.conv4 = nn.Conv1d(filters_num, filters_num, kernel_size)
        self.conv4_bn = nn.BatchNorm1d(filters_num)
        shape.conv(filters_num, kernel_size)

        self.maxpool2_pad = nn.ConstantPad1d((1, 0), 0)
        self.maxpool2 = nn.MaxPool1d(2, 2)
        shape.pool(2, 2, padding=[1,0])

        self.conv5 = nn.Conv1d(filters_num, filters_num + filters_step, kernel_size)
        filters_num += filters_step
        self.conv5_bn = nn.BatchNorm1d(filters_num)
        shape.conv(filters_num, kernel_size)

        self.conv6 = nn.Conv1d(filters_num, filters_num, kernel_size)
        self.conv6_bn = nn.BatchNorm1d(filters_num)
        shape.conv(filters_num, kernel_size)

        self.maxpool3_pad = nn.ConstantPad1d((1, 0), 0)
        self.maxpool3 = nn.MaxPool1d(2, 2)
        shape.pool(2, 2, padding=[1,0])

        self.conv7 = nn.Conv1d(filters_num, filters_num + filters_step, kernel_size)
        filters_num += filters_step
        self.conv7_bn = nn.BatchNorm1d(filters_num)
        shape.conv(filters_num, kernel_size)

        self.avg_pool1 = nn.AvgPool1d(2, 2)
        shape.pool(2, 2)

        self.conv8 = nn.Conv1d(filters_num, filters_num + filters_step, 4)
        filters_num += filters_step
        self.conv8_bn = nn.BatchNorm1d(filters_num)
        shape.conv(filters_num, 4)

        self.avg_pool2 = nn.AvgPool1d(4, 4)
        shape.pool(4, 4)

        self.flatten_size = shape.size

        self.fc1 =  nn.Linear(self.flatten_size, fc_units)
        self.fc1_bn = nn.BatchNorm1d(fc_units)
        self.fc1_drop = nn.Dropout(dropout)
        shape.fc(fc_units)

        self.fc2 =  nn.Linear(fc_units, fc_units)
        self.fc2_bn = nn.BatchNorm1d(fc_units)
        self.fc2_drop = nn.Dropout(dropout)
        shape.fc(fc_units)

        self.fc3 = nn.Linear(shape.size, class_num)
        shape.fc(class_num) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)

        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.relu(x)

        x = self.maxpool2_pad(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = F.relu(x)

        x = self.maxpool3_pad(x)
        x = self.maxpool3(x)

        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = F.relu(x)

        x = self.avg_pool1(x)
        
        x = self.conv8(x)
        x = self.conv8_bn(x)
        x = F.relu(x)

        x = self.avg_pool2(x)

        x = x.view(-1, self.flatten_size)

        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc1_drop(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.fc2_drop(x)

        x = self.fc3(x)

        return x