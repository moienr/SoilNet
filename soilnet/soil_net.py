import torch
import torch.nn as nn
from submodules.cnn_feature_extractor import CNNFlattener64, CNNFlattener128, ResNet101
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
    def __init__(self, cnn_in_channels = 14 ,regresor_input_from_cnn = 1024, hidden_size=128):
        super().__init__()
        self.cnn = CNNFlattener64(in_channels=cnn_in_channels)
        self.reg = MultiHeadRegressor(regresor_input_from_cnn, hidden_size= hidden_size)
        
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
    """
    SoilNetMonoLSTM: A PyTorch module for multi-task regression on soil data using a combination of a convolutional neural
    network and a long short-term memory (LSTM) network.
    """
    def __init__(self, cnn_in_channels = 14 ,lstm_n_features = 10,lstm_n_layers =2, lstm_out = 128, cnn_out = 1024, hidden_size=128):
        """
        SoilNetMonoLSTM: A PyTorch module for multi-task regression on soil data using a combination of a convolutional neural
        network and a long short-term memory (LSTM) network.

        Args
        ----
            `cnn_in_channels` (int): The number of input image channels for the CNNFlattener Module (default: 14).
            `lstm_n_features` (int): The number of features in the input to the LSTM (default: 10).
            `lstm_n_layers` (int): The number of layers in the LSTM (default: 2).
            `lstm_out` (int): The number of output features in the LSTM (default: 128).
            `cnn_out` (int): The number of output features from the CNNFlatnner (default: 1024).
            `hidden_size` (int): The size of the hidden state in the LSTM (default: 128).

        Outputs
        -------
            - output (torch.Tensor): A tensor of shape `(batch_size, 1)` representing the predicted output for each
            task in the multi-task regression problem.
        """
        super().__init__()
        self.cnn = CNNFlattener64(cnn_in_channels) 
        self.lstm = rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        self.reg = MultiHeadRegressor(cnn_out, lstm_out, hidden_size= hidden_size)
        
    def forward(self, raster_stack, ts_features):
        """
        Inputs
        ------
            - `raster_stack` (torch.Tensor): A 4D tensor of shape `(batch_size, channels, height, width)` representing a stack of
            raster images.
            - `ts_features` (torch.Tensor): A 3D tensor of shape `(batch_size, seq_length, , n_features)` representing a sequence
            of time-series features. | `seq_length` is the number of time steps in the sequence. e.g. months in our climate data
            
        Outputs
        -------
            - output (torch.Tensor): A tensor of shape `(batch_size, 1)` representing the predicted output of regression.
        """

        flat_raster = self.cnn(raster_stack)
        lstm_output = self.lstm(ts_features)
        output = self.reg(flat_raster, lstm_output)
        return output
    

class ResNet(nn.Module):
    def __init__(self, resnet_architecture = "101" ,cnn_in_channels = 14 ,regresor_input_from_cnn = 1024, hidden_size=128):
        super().__init__()
        if resnet_architecture == "101":
            self.cnn = ResNet101(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            self.reg = MultiHeadRegressor(regresor_input_from_cnn, hidden_size= hidden_size)
        
    def forward(self, raster_stack):
        """
        Forward pass of the Resnet module.
        
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
    model = SoilNetFC(cnn_in_channels=12)
    y = model(x)
    print(y.detach().shape)
    
    print("Testing SoilNetMonoLSTM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    x_cnn = torch.randn((32,12,64,64)).to(device)
    x_lstm = torch.randn((32, 12, 10)).to(device)
    model = SoilNetMonoLSTM(cnn_in_channels=12, lstm_n_features=10).to(device)
    y= model(x_cnn, x_lstm)
    print(y.detach().shape)
    
    print("Testing Resnet...")
    x = torch.randn((32,12,64,64))
    model = ResNet(cnn_in_channels=12)
    y = model(x)
    print(y.detach().shape)
    
    