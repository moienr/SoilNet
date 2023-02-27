import torch
import torch.nn as nn

class Regressor(nn.Module):
    """A simple regressor that takes in a 1024 dimensional vector and a 12 dimensional vector and outputs a single value.
        
        Args:
        ---
        input_size (int): Size of the input tensor. Default is 1024.
        aux_size (int): Size of the auxiliary tensor. Default is 12.
        hidden_size (int): Size of the hidden layer. Default is 128.

        Attributes:
        ---
        input_fc (nn.Linear): Linear layer that maps input tensor to hidden layer.
        aux_fc (nn.Linear): Linear layer that maps auxiliary tensor to hidden layer.
        concat_fc (nn.Linear): Linear layer that concatenates output of input_fc and aux_fc and maps it to hidden layer.
        output_fc (nn.Linear): Linear layer that maps hidden layer to output scalar.
        relu (nn.ReLU): Rectified Linear Unit activation function.

        Methods:
        ---
        forward(x, aux): Defines the computation performed at every forward pass of the network. 
                 Takes input tensor x and auxiliary tensor aux as inputs and returns a scalar output tensor.
    
    
    """
    def __init__(self, input_size=1024, aux_size=12, hidden_size=128):
        super().__init__()
        # Initialize a linear layer for processing input data
        self.input_fc = nn.Linear(input_size, hidden_size)
        # Initialize a linear layer for processing auxiliary data
        self.aux_fc = nn.Linear(aux_size, hidden_size)
        # A layer that takes the concatenation of the output of input_fc and aux_fc as input and maps it to hidden_size
        self.concat_fc = nn.Linear(2*hidden_size, hidden_size)
        # Takes the final output of concat_fc and maps it to a scalar output
        self.output_fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU() # Initialize a ReLU activation function
        
    def forward(self, x, aux):
        # Process input data and auxiliary data
        x = self.input_fc(x)
        aux = self.aux_fc(aux)
        # Apply ReLU activation function
        x = self.relu(x)
        aux = self.relu(aux)
        # Concatenate the output of input_fc and aux_fc
        x = torch.cat((x, aux),dim=-1)
        # Process the concatenated output
        x = self.concat_fc(x)
        x = self.relu(x)
        # Map the output of concat_fc to a scalar output
        output = self.output_fc(x)
        return output


if __name__ == "__main__":
    # Test the regressor
    x = torch.randn((32,1024))
    y = torch.rand((32,12))
    model = Regressor()
    z = model(x,y)
    print(z.shape)