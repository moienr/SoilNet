import torch
import torch.nn as nn
from channel_attention import ChannelAttention
from cnn_block import CNNBlock

class CNNFlattener(nn.Module):
    """
    A convolutional neural network (CNN) flattener that extracts features from an input image and flattens them
    into a 1D feature vector.

    Args:
        in_channels (int): The number of input channels in the image.
        features (list of int): The number of output channels in each convolutional layer.

    Attributes:
        conv_sep (nn.Conv2d): A convolutional layer that separates the input into different channels.
        ca (ChannelAttention): A channel attention module that helps the network focus on important channels.
        model (nn.Sequential): The convolutional neural network that extracts features from the input image.
        last_conv (nn.Conv2d): A convolutional layer that further processes the output of the convolutional neural network
                               to produce a 1D feature vector.

    Methods:
        forward(x): Computes a forward pass through the network and returns the resulting 1D feature vector.

    """

    def __init__(self, in_channels=12, features=[64, 128, 256, 512, 1024]):
        """
        Initializes the CNNFlattener module.

        Args:
            in_channels (int): The number of input channels in the image.
            features (list of int): The number of output channels in each convolutional layer.
        """
        super().__init__()

        # Separable convolutional layer to split input into different channels
        self.conv_sep = nn.Conv2d(in_channels,
                                  out_channels=in_channels * 8,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  groups=in_channels)

        # Channel attention module to help focus on important channels
        self.ca = ChannelAttention(in_channels * 8, ratio=8)

        # Construct the CNN using CNNBlock modules
        layers = []
        in_channels = in_channels  # input channels for the first CNN block
        for feature in features:
            layers.append(CNNBlock(in_channels,
                                   feature,
                                   kernel_size=4,
                                   stride=2,
                                   padding=0,
                                   norm=False if feature == features[0] else True))
            in_channels = feature  # set the input channels for the next CNN block

        self.model = nn.Sequential(*layers)

        # Final convolutional layer to flatten output into a 1D feature vector
        self.last_conv = nn.Conv2d(in_channels,
                                   out_channels=1024,
                                   kernel_size=2,
                                   stride=1,
                                   padding=0)

    def forward(self, x):
        """
        Computes a forward pass through the CNNFlattener module and returns the resulting 1D feature vector.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The 1D feature vector obtained by flattening the output of the convolutional neural network.
        """
        x = self.conv_sep(x)
        x = self.ca(x)
        x = self.model(x)
        x = self.last_conv(x).squeeze(-1).squeeze(-1)
        return x
