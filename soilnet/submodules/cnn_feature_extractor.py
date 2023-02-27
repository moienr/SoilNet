import torch
import torch.nn as nn
from channel_attention import ChannelAttention
from cnn_block import CNNBlock

class CNNFlattener(nn.Module):
    def __init__(self, in_channels=12, features=[64, 128, 256, 512, 1024]):
        super().__init__()

        #===========================================
        self.conv_sep = nn.Conv2d(in_channels,
                                out_channels = in_channels * 8,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1,
                                groups=in_channels)

        self.ca = ChannelAttention(in_channels * 8,
                                ratio = 8
                                )

        in_channels = in_channels * 8
        #===========================================

        layers = []
        in_channels = in_channels # Input is for example 7 bands of S2 and output is 1 bands S1 so the total in of Disc is 7 + 1
        for feature in features:
            layers.append(
                CNNBlock(in_channels,
                        feature,
                        kernel_size = 4,
                        stride = 2,
                        padding = 0,
                        norm = False if feature == features[0] else True # we don't want batch/instance norm in the first layer.
                        )
            )
            in_channels = feature # setting the in_channels to the last layer created

        # layers.append(
        #     nn.Conv2d(in_channels,1,4,stride=1, padding=1, padding_mode='reflect') # A the last layer to turn 512 channels into 1
        # )

        self.model = nn.Sequential(*layers)

        self.last_conv = nn.Conv2d(in_channels,
                            out_channels = 1024,
                            kernel_size = 2,
                            stride = 1,
                            padding = 0,
                            )


    def forward(self, x):
        x = self.conv_sep(x)
        #print(x.shape)
        x = self.ca(x)
        x = self.model(x)
        x = self.last_conv(x).squeeze(-1).squeeze(-1)
        return x
