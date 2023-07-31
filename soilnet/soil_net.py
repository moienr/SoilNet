import torch
import torch.nn as nn
from submodules.cnn_feature_extractor import CNNFlattener64, CNNFlattener128,\
                                                ResNet101, ResNet101GLAM,\
                                                    VGG16, VGG16GLAM
from submodules.regressor import Regressor, MultiHeadRegressor
from submodules import rnn
from typing import Tuple

class SoilNet(nn.Module):
    def __init__(self, use_glam = False , cnn_arch = "resnet101", reg_version = 1,
                 cnn_in_channels = ... ,regresor_input_from_cnn = ..., hidden_size=128):
        super().__init__()
        if use_glam:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet' or 'vgg16'.")

        else:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet' or 'vgg16'.")
            

        
        self.reg = MultiHeadRegressor(regresor_input_from_cnn, hidden_size= hidden_size, version=reg_version)
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
        
class SoilNetLSTM(nn.Module):
    def __init__(self, use_glam = False  , cnn_arch = "resnet101", reg_version = 1,
                 cnn_in_channels = ... ,regresor_input_from_cnn = ..., 
                 lstm_n_features = ...,lstm_n_layers =..., lstm_out = 128, hidden_size=128):
        
        super().__init__()
        
        if use_glam:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet' or 'vgg16'.")

        else:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet' or 'vgg16'.")
            

        
        self.lstm = rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        
        self.reg = MultiHeadRegressor(regresor_input_from_cnn, lstm_out, hidden_size= hidden_size, version=reg_version)
        
    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Inputs
        ------
        input_raster_ts : A tupple containing the following two tensors:
            * raster_stack (torch.Tensor): A 4D tensor of shape `(batch_size, channels, height, width)` representing a stack of raster images.
            * ts_features (torch.Tensor): A 3D tensor of shape `(batch_size, seq_length, , n_features)` representing a sequence of time-series features. | `seq_length` is the number of time steps in the sequence. e.g. months in our climate data
            
        Outputs
        -------
            - output (torch.Tensor): A tensor of shape `(batch_size, 1)` representing the predicted output of regression.
        """
        raster_stack, ts_features = input_raster_ts
        flat_raster = self.cnn(raster_stack)
        lstm_output = self.lstm(ts_features)
        output = self.reg(flat_raster, lstm_output)
        return output
            
 
    
    
    
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    # print("Testing SoilNet...")
    # x = torch.randn((32,12,128,128))
    # y = torch.rand((32,12))
    # model = SoilNet()
    # z = model(x,y)
    # print(z.detach().shape)
    # print('Testing SoilNetFC...')
    # x = torch.randn((32,12,64,64))
    # model = SoilNetFC(cnn_in_channels=12)
    # y = model(x)
    # print(y.detach().shape)
    
    # print("Testing SoilNetMonoLSTM...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    # x_cnn = torch.randn((32,12,64,64)).to(device)
    # x_lstm = torch.randn((32, 12, 10)).to(device)
    # model = SoilNetMonoLSTM(cnn_in_channels=12, lstm_n_features=10).to(device)
    # y= model(x_cnn, x_lstm)
    # print(y.detach().shape)
    
    print("Testing SoilNet...")
    x = torch.randn((32,12,64,64))
    model = SoilNet(cnn_in_channels=12, resnet_architecture="101+GLAM")
    y = model(x)
    print(y.detach().shape)
    
    print('Testing SoilNetLSTM...')
    x_cnn = torch.randn((32,12,64,64)).to(device)
    x_lstm = torch.randn((32, 60, 10)).to(device)
    model = SoilNetLSTM(resnet_architecture="101+GLAM", cnn_in_channels= 12, regresor_input_from_cnn=1024,
                       lstm_n_features= 10, lstm_n_layers=2, lstm_out=128, hidden_size=128).to(device)
    y= model((x_cnn, x_lstm))
    print(y.detach().shape)