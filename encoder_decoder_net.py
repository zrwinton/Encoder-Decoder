import scipy.io as scio
import h5py
import torch
import torch.nn as nn
import time
import numpy as np
import random
import torch.optim as optim
from torch.nn import init


class My_NN(nn.Module):
    def __init__(self):
        super(My_NN, self).__init__()

        self.his_up = nn.Sequential(
            nn.Upsample(scale_factor=12, mode='bicubic'),
        )
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=69,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.encoder_pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.encoder_pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.encoder_pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )
        self.decoder_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
        )
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decoder_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.decoder_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
        )
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=68,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, his, pan):
        time_start = time.time()
        his_upsample = self.his_up(his)
        time_end = time.time()
        print(time_end-time_start)
        input = torch.cat((pan, his_upsample), 1)
        encod_conv1 = self.encoder_conv1(input)
        encod_pool1 = self.encoder_pool1(encod_conv1)
        encod_conv2 = self.encoder_conv2(encod_pool1)
        encod_pool2 = self.encoder_pool2(encod_conv2)
        encod_conv3 = self.encoder_conv3(encod_pool2)
        encod_pool3 = self.encoder_pool3(encod_conv3)
        decod_up1 = self.decoder_up1(encod_pool3)
        decod_up1 = decod_up1 + encod_conv3
        decod_conv1 = self.decoder_conv1(decod_up1)
        # decod_conv1 = decod_conv1 + encod_pool2
        decod_up2 = self.decoder_up2(decod_conv1)
        decod_up2 = decod_up2 + encod_conv2
        decod_conv2 = self.decoder_conv2(decod_up2)
        # decod_conv2 = decod_conv2 + encod_pool1
        decod_up3 = self.decoder_up3(decod_conv2)
        decod_up3 = decod_up3 + encod_conv1
        output = self.decoder_conv3(decod_up3)
        output = output + his_upsample
        return output

