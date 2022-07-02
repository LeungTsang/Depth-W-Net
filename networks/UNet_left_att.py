# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

from collections import OrderedDict
import torch
import torch.nn as nn
		
class UNet_left(nn.Module):

    def __init__(self, in_channels=3, init_features=64):
        super(UNet_left, self).__init__()

        features = init_features
        self.encoder1 = UNet_left._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet_left._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet_left._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet_left._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.neck = UNet_left._block(features * 8, features * 16, name="neck")


    def forward(self, x, att):
        # encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
       # neck
        neck = self.neck(self.pool4(enc4))
        att = att[:,:,::4,::4]
        #print("==")
        #print(att.shape)
        #print(neck.shape)
        neck_ = neck.flatten(-2)
        #print(neck_.shape)
        neck_ = neck_.unsqueeze(-1).expand(neck_.shape[0],neck_.shape[1],neck_.shape[2],neck_.shape[2])
        #print(neck_.shape)
        neck_att = (neck_*att).sum(-1).view(neck.shape[0],neck.shape[1],neck.shape[2],neck.shape[3])
        #print(neck_att.shape)
        return [enc1,enc2,enc3,enc4,neck_att]

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

