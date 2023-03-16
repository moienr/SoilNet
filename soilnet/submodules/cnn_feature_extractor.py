import torch
import torch.nn as nn
from channel_attention import ChannelAttention

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
    
def test_cnn_block():
    """
    Tests the CNNBlock class.
    """
    x = torch.randn((32, 12, 128, 128))
    cnn_block = CNNBlock(
      in_channels = 12,
      out_channels = 64,
      kernel_size = 4,
      padding= 1,
      stride = 2,
      norm = True)
    
    output = cnn_block(x)
    print(output.shape)   
    
    
    
    
    

class CNNFlattener128(nn.Module):
    """
    # FOR 128x128 images
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
        
        in_channels = in_channels * 8 # set the input channels for the first CNN block
        
        # Construct the CNN using CNNBlock modules
        layers = []
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
        x = self.last_conv(x).squeeze(-1).squeeze(-1) # flatten the output, don't just use squeeze() because it will remove the batch dimension if it is 1
        return x
    
    
class CNNFlattener64(nn.Module):
    """
    # FOR 64x64 images
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
        
        in_channels = in_channels * 8 # set the input channels for the first CNN block
        
        # Construct the CNN using CNNBlock modules
        layers = []
        for feature in features:
            layers.append(CNNBlock(in_channels,
                                   feature,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
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
        x = self.last_conv(x).squeeze(-1).squeeze(-1) # flatten the output, don't just use squeeze() because it will remove the batch dimension if it is 1
        return x
    
    
def test_cnn_flattener(ClassToTest=CNNFlattener64, image_size=64):
    """
    Tests the CNNFlattener class.
    """
    x = torch.randn((16, 12, image_size, image_size))
    cnn_flattener = ClassToTest()
    output = cnn_flattener(x)
    print(output.shape)



   
if __name__ == "__main__": # testing the model
    print("Testing CNNBlock...")
    test_cnn_block()
    print("Testing CNNFlattener...")
    test_cnn_flattener()