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
		
class UNet_right(nn.Module):

    def __init__(self, out_channels=1, scales = [0], activate = 'softmax', init_features=64):
        super(UNet_right, self).__init__()

        features = init_features

        self.scales = scales
        out_channels = out_channels

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet_right._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_right._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_right._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_right._block(features * 2, features, name="dec1")
        
        self.convs = OrderedDict()
        for scale in self.scales:
            divisor = (2 ** scale)
            self.convs[scale] = nn.Conv2d(in_channels=features*divisor, out_channels=out_channels, kernel_size=1)
        
        self.decoder = nn.ModuleList(list(self.convs.values()))

        if activate == 'softmax':
            self.activate = nn.Softmax(dim=1)
        elif activate == 'sigmoid':
            self.activate = nn.Sigmoid()

    def forward(self, enc):
        
        outputs = {}
        # encoder

        # decoder
        dec4 = self.upconv4(enc[4])
        dec4 = torch.cat((dec4, enc[3]), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc[2]), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc[1]), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc[0]), dim=1)
        dec1 = self.decoder1(dec1)

        dec = [dec1,dec2,dec3,dec4]
		
        for scale in self.scales:
            outputs[("output", scale)] = self.activate(self.convs[scale](dec[scale]))
        return outputs

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

