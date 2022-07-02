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
		
class LinearHead(nn.Module):

    def __init__(self, out_channels=1, activate = 'softmax', in_channels=[64, 64, 128, 256, 512]):
        super(LinearHead, self).__init__()


        #self.upconv = nn.ConvTranspose2d(
        #    in_channels[1], in_channels[0], kernel_size=2, stride=2
        #)
        #self.decoder = Head._block(in_channels[0] * 2, in_channels[0], name="dec1")
        
        self.convs = nn.Conv2d(in_channels[2], out_channels=out_channels[2], kernel_size=1, padding=0, bias = True)
        
        if activate == 'softmax':
            self.activate = nn.Softmax(dim=1)
        elif activate == 'sigmoid':
            self.activate = nn.Sigmoid()

    def forward(self, representation):
        
        outputs = {}
        # encoder

        outputs[("output", 0)] = self.activate(self.convs(representation))
        return outputs

class FullHead(nn.Module):

    def __init__(self, out_channels=1, activate = 'softmax', in_channels=[64, 64, 128, 256, 512]):
        super(FullHead, self).__init__()


        #self.upconv = nn.ConvTranspose2d(
        #    in_channels[1], in_channels[0], kernel_size=2, stride=2
        #)
        #self.decoder = Head._block(in_channels[0] * 2, in_channels[0], name="dec1")
        self.upconv2 = nn.ConvTranspose2d(
            in_channels[2], in_channels[1], kernel_size=2, stride=2
        )
        self.decoder2 = FullHead._block(in_channels[1] * 2, in_channels[1], name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            in_channels[1], in_channels[0], kernel_size=2, stride=2
        )
        self.decoder1 = FullHead._block(in_channels[0] * 2, in_channels[0], name="dec1")

        self.convs = nn.Conv2d(in_channels[0], out_channels=out_channels, kernel_size=1, padding=0, bias = True)
        
        if activate == 'softmax':
            self.activate = nn.Softmax(dim=1)
        elif activate == 'sigmoid':
            self.activate = nn.Sigmoid()

    def forward(self, representation, enc):
        
        outputs = {}
        # encoder
        dec2 = self.upconv2(representation)
        dec2 = torch.cat((dec2, enc[1]), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc[0]), dim=1)
        dec1 = self.decoder1(dec1)

        outputs[("output", 0)] = self.convs(dec1)
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

