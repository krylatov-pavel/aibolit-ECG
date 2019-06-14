import torch.nn as nn
import torch.nn.functional as f

class ArrhythmiaCNN(nn.Module):
    def __init__(self, config):
        super(ArrhythmiaCNN, self).__init__()

        dropout_rate = config.model.hparams["dropout_rate"]
        class_num = len(config.dataset.params["label_map"])

        self.conv1 = nn.Conv1d(1, 64, 7)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 7)
        self.conv2_bn = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(3, 3)
        self.conv3 = nn.Conv1d(64, 128, 7)
        self.conv3_bn = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, 7)
        self.conv4_bn = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(3, 3)
        self.conv5 = nn.Conv1d(128, 192, 7)
        self.conv5_bn = nn.BatchNorm1d(192)
        self.conv6 = nn.Conv1d(192, 192, 7)
        self.conv6_bn = nn.BatchNorm1d(192)
        self.maxpool3 = nn.MaxPool1d(3, 3)
        self.conv7 = nn.Conv1d(192, 256, 7)
        self.conv7_bn = nn.BatchNorm1d(256)
        self.conv8 = nn.Conv1d(256, 256, 7)
        self.conv8_bn = nn.BatchNorm1d(256)
        self.maxpool4 = nn.MaxPool1d(3, 3)
        self.conv9 = nn.Conv1d(256, 320, 5)
        self.conv9_bn = nn.BatchNorm1d(320)
        self.conv10 = nn.Conv1d(320, 320, 5)
        self.conv10_bn = nn.BatchNorm1d(320)
        self.avgpool1 = nn.AvgPool1d(4, 4)
        self.fc1 = nn.Linear(320, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_drop = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = f.relu(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = f.relu(x)

        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = f.relu(x)

        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = f.relu(x)

        x = f.pad(x, (0,1))
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = f.relu(x)

        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = f.relu(x)

        x = f.pad(x, (0,1))
        x = self.maxpool3(x)

        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = f.relu(x)

        x = self.conv8(x)
        x = self.conv8_bn(x)
        x = f.relu(x)

        x = self.maxpool4(x)

        x = self.conv9(x)
        x = self.conv9_bn(x)
        x = f.relu(x)

        x = self.conv10(x)
        x = self.conv10_bn(x)
        x = f.relu(x)

        x = self.avgpool1(x)
        
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = f.relu(x)
        x = self.fc1_drop(x)

        x = self.fc2(x)

        return x