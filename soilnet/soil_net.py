import torch
import torch.nn as nn
from submodules.cnn_feature_extractor import CNNFlattener64, CNNFlattener128
from submodules.regressor import Regressor, MultiHeadRegressor
from submodules import rnn


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
    def __init__(self, cnn_in_channels = 14 ,regresor_input = 1024, hidden_size=128):
        super().__init__()
        self.cnn = CNNFlattener64()
        self.reg = MultiHeadRegressor(regresor_input, hidden_size= hidden_size)
        
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
    

class SoilNetMonoLSTM(nn.Module):
    def __init__(self, cnn_in_channels = 14 ,lstm_n_features = 10,lstm_n_layers =2, lstm_out = 128, cnn_out = 1024, hidden_size=128):
        super().__init__()
        self.cnn = CNNFlattener64(cnn_in_channels) 
        self.lstm = rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        self.reg = MultiHeadRegressor(cnn_out, lstm_out, hidden_size= hidden_size)
        
    def forward(self, raster_stack, ts_features):
        """
        Forward pass of the SoilNet module.
        
        Args:
            raster_stack (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            auxillary_data (torch.Tensor): Auxiliary input tensor of shape (batch_size, aux_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        flat_raster = self.cnn(raster_stack)
        lstm_output = self.lstm(ts_features)
        output = self.reg(flat_raster, lstm_output)
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
    model = SoilNetFC(cnn_in_channels=12)
    y = model(x)
    print(y.detach().shape)
    
    print("Testing SoilNetMonoLSTM...")
    x_cnn = torch.randn((32,12,64,64))
    x_lstm = torch.randn((32, 12, 10))
    model = SoilNetMonoLSTM(cnn_in_channels=12, lstm_n_features=10)
    y= model(x_cnn, x_lstm)
    print(y.detach().shape)
    
    
    
    