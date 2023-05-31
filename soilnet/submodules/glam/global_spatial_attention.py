import torch
from torch import nn

class GlobalSpatialAttention(nn.Module):
    def __init__(self, in_channels, num_reduced_channels):
        super().__init__()
        
        self.conv1x1_q = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_k = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_v = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_att = nn.Conv2d(num_reduced_channels, in_channels, 1, 1)
        
        self.att = None # for visualization later in the notebook
        
    def forward(self, feature_maps, global_channel_output):
        query = self.conv1x1_q(feature_maps)
        N, C, H, W = query.shape
        query = query.reshape(N, C, -1)
        key = self.conv1x1_k(feature_maps).reshape(N, C, -1)
        
        query_key = torch.bmm(key.permute(0, 2, 1), query)
        query_key = query_key.reshape(N, -1).softmax(-1)
        query_key = query_key.reshape(N, int(H*W), int(H*W))
        value = self.conv1x1_v(feature_maps).reshape(N, C, -1)
        self.att = torch.bmm(value, query_key).reshape(N, C, H, W)
        self.att = self.conv1x1_att(self.att)
        
        return (global_channel_output * self.att) + global_channel_output



import unittest
class TestGlobalSpatialAttention(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 2
        self.in_channels = 3
        self.num_reduced_channels = 2
        self.height = 4
        self.width = 4
        self.attention = GlobalSpatialAttention(self.in_channels, self.num_reduced_channels)
        self.feature_maps = torch.randn(self.batch_size, self.in_channels, self.height, self.width)
        self.global_channel_output = torch.randn(self.batch_size, self.in_channels, 1, 1)
        
    def test_forward(self):
        output = self.attention(self.feature_maps, self.global_channel_output)
        self.assertEqual(output.shape, self.global_channel_output.shape)
            
if __name__ == '__main__':
    unittest.main()