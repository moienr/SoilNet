import torch
import torch.nn as nn
import os
import sys

script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
script_dir = os.path.dirname(script_path) #i.e. /path/to/dir/
sys.path.append(script_dir + '\submodules') # add submodules to path so that CNNFlattener can import ChannelAttention from within CNNFlattener

from submodules.cnn_feature_extractor import CNNFlattener
from submodules.regressor import Regressor



class SoilNet(nn.Module):
    def __init__(self, input_size=1024, aux_size=12, hidden_size=128):
        super().__init__()
        self.cnn = CNNFlattener()
        self.reg = Regressor()
        
    def forward(self, raster_stack, auxillary_data):
        """
        Forward pass of the SoilNet module.
        
        Args:
            raster_stack (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            auxillary_data (torch.Tensor): Auxiliary input tensor of shape (batch_size, aux_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        flat_raster = self.cnn(raster_stack)
        output = self.reg(flat_raster,auxillary_data)
        return output
    
    
if __name__ == "__main__":
    x = torch.randn((32,12,128,128))
    y = torch.rand((32,12))
    model = SoilNet()
    z = model(x,y)
    print(z.detach().shape)