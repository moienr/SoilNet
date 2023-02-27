import torch
import torch.nn as nn
from submodules.cnn_feature_extractor import CNNFlattener
from submodules.regressor import Regressor

class SoilNet(nn.Module):
    def __init__(self, input_size=1024, aux_size=12, hidden_size=128):
        super().__init__()
        self.cnn = CNNFlattener()
        self.reg = Regressor()
        
    def forward(self, raster_stack, auxillary_data):
        flat_raster = self.cnn(raster_stack)
        output = self.reg(flat_raster,auxillary_data)
        return output
    
if __name__ == "__main__":
    x = torch.randn((32,12,128,128))
    y = torch.rand((32,12))
    model = SoilNet()
    z = model(x,y)
    print(z.detach())