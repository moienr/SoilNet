import torch
from torch import nn

class GlobalChannelAttention(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"
        
        self.conv_q = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.conv_k = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.GAP = nn.AvgPool2d(feature_map_size)
        
    def forward(self, x):
        N, C, H, W = x.shape
        
        query = key = self.GAP(x).reshape(N, 1, C)
        query = self.conv_q(query).sigmoid()
        key = self.conv_q(key).sigmoid().permute(0, 2, 1)
        query_key = torch.bmm(key, query).reshape(N, -1)
        query_key = query_key.softmax(-1).reshape(N, C, C)
        
        value = x.permute(0, 2, 3, 1).reshape(N, -1, C)
        att = torch.bmm(value, query_key).permute(0, 2, 1)
        att = att.reshape(N, C, H, W)
        return x * att
