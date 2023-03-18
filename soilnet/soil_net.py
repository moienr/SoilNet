import torch
import torch.nn as nn
from submodules.cnn_feature_extractor import CNNFlattener64, CNNFlattener128
from submodules.regressor import Regressor, MultiHeadRegressor



class SoilNet(nn.Module):
    def __init__(self, input_size=1024, aux_size=12, hidden_size=128):
        super().__init__()
        self.cnn = CNNFlattener128()
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
    
    
class SoilNetFC(nn.Module):
    def __init__(self, regresor_input = 1024, hidden_size=128,device='cpu'):
        super().__init__()
        self.cnn = CNNFlattener64().to(device)
        self.reg = MultiHeadRegressor(1024).to(device)
        
    def forward(self, raster_stack):
        """
        Forward pass of the SoilNet module.
        
        Args:
            raster_stack (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            auxillary_data (torch.Tensor): Auxiliary input tensor of shape (batch_size, aux_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        flat_raster = self.cnn(raster_stack)
        output = self.reg(flat_raster)
        return output
       
    
    
    
    
    
if __name__ == "__main__":
    print("Testing SoilNet...")
    x = torch.randn((32,12,128,128))
    y = torch.rand((32,12))
    model = SoilNet()
    z = model(x,y)
    print(z.detach().shape)
    print('Testing SoilNetFC...')
    x = torch.randn((32,12,64,64))
    model = SoilNetFC()
    y = model(x)
    print(y.detach().shape)
    
    
    
    