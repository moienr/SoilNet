import torch
from torch import nn

class LocalChannelAttention(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"
        
        self.conv = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.GAP = nn.AvgPool2d(feature_map_size)
    
    def forward(self, x):
        N, C, H, W = x.shape
        att = self.GAP(x).reshape(N, 1, C)
        att = self.conv(att).sigmoid()
        att =  att.reshape(N, C, 1, 1)
        return (x * att) + x
