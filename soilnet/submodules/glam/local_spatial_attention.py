import torch
from torch import nn

class LocalSpatialAttention(nn.Module):
    def __init__(self, in_channels, num_reduced_channels):
        super().__init__()
        
        self.conv1x1_1 = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_2 = nn.Conv2d(int(num_reduced_channels*4), 1, 1, 1)
        
        self.dilated_conv3x3 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=1)
        self.dilated_conv5x5 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=2, dilation=2)
        self.dilated_conv7x7 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=3, dilation=3)
        
    def forward(self, feature_maps, local_channel_output):
        att = self.conv1x1_1(feature_maps)
        d1 = self.dilated_conv3x3(att)
        d2 = self.dilated_conv5x5(att)
        d3 = self.dilated_conv7x7(att)
        att = torch.cat((att, d1, d2, d3), dim=1)
        att = self.conv1x1_2(att)
        return (local_channel_output * att) + local_channel_output
