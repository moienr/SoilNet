import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    """
    A building block for a Convolutional Neural Network (CNN) that consists of a convolutional layer followed
    by an optional instance normalization layer and a leaky ReLU activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel (default: 4).
        stride (int): Stride of the convolution (default: 2).
        padding (int): Padding of the convolution (default: 0).
        norm (bool): Whether to apply instance normalization (default: True).

    Input:
        x (tensor): Input tensor of shape (batch_size, in_channels, height, width).

    Output:
        output (tensor): Output tensor of shape (batch_size, out_channels, height', width').
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, norm=True):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='reflect'))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """
        Applies the CNN block to the input tensor.

        Args:
            x (tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            output (tensor): Output tensor of shape (batch_size, out_channels, height', width').
        """
        output = self.conv(x)
        return output


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