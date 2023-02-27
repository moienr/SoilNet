import torch
import torch.nn as nn

class CNNBlock(nn.Module):
  """
  A Class to create a CNN Block
  ...

  Attributes
  ----------
  `in_channels` : number of input channels
  `out_channels` : number  of output channels
  `stride` : stride
  `norm` : adds a InstanceNorm layer to the block if `True`
  """
  def __init__(self, in_channels, out_channels, kernel_size =4, stride=2, padding =0, norm=True):
    super().__init__()

    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='reflect'))
    if norm: 
      layers.append(nn.InstanceNorm2d(out_channels)) 
    layers.append(nn.LeakyReLU(0.2))

    self.conv = nn.Sequential(*layers)
  
  def forward(self, x):
    return self.conv(x)

if __name__ == "__main__":
  block = CNNBlock(
      in_channels = 12,
      out_channels = 64,
      kernel_size = 4,
      padding= 1,
      stride = 2,
      norm = True
  )

  x = torch.randn((1, 12, 128, 128))
  y = block(x)
  print(y.shape)