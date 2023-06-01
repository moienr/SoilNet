import torch
import torch.nn as nn
from torch.nn import ModuleList

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


    
# Moduel Lists: FIX : https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
    
class MultiHeadRegressorBase(nn.Module):
    """
    A multi-head regressor that can take any number of input from branchesa and encodes each input to a common space of size `hidden_size` and maps the concatenated output to a scalar output.

    """

    def __init__(self, *input_sizes:int, hidden_size=128, activation = "sigmoid" ):
        """ Initializes the MultiHeadRegressor, with with a list that contains the output_size of the branches that are being inputed into the model.
        Args:
        -----
        `*input_sizes` (int): The output_size of each input branch.
        `hidden_size` (int): The size of the common space to which each input is encoded. Default: 128.
        `activation` (str): The activation function to be used. Default: "sigmoid". | NOTE: Only `"sigmoid"` and `"relu"` are supported.
        
        Example:
        --------
        ```
        # 32 is the barch size.
        x = torch.randn((32,1024))
        y = torch.rand((32,12))
        z = torch.rand((32, 5))
        model = MultiHeadRegressor(1024, 12, 5, hidden_size=128)
        w = model(x,y,z)
        print(w.shape)
        ```
        
        """
        super().__init__()
        # Creating a List of encoders for each input, this will encode each input to a common space of size hidden_size
        # NOTE: We should not use normal list to hold pytorch modules as it will not be registered as a submodule of the model.
        # This is the reason we were getting error when setting the modle to device, the models inside the list were not registered as submodules,
        # therfore they were not moved to the device.
        self.encoders = ModuleList([nn.Linear(in_size, hidden_size) for in_size in input_sizes])
        # concatenating the output of all the encoders    
        self.concat_fc = nn.Linear(len(input_sizes)*hidden_size, hidden_size)
    
        # Takes the final output of concat_fc and maps it to a scalar output
        self.output_fc = nn.Linear(hidden_size, 1)
        if activation == "relu":
            self.activ = nn.ReLU() # Initialize a ReLU activation function
        elif activation == "sigmoid":
            self.activ = nn.Sigmoid()
        else:
            raise ValueError("Activation function is invalid, please choose from 'relu' or 'sigmoid'.")
        
    def forward(self, *inputs:torch.Tensor) -> torch.Tensor:
        """ Forward pass of the MultiHeadRegressor.
        Args:
        `*inputs` (torch.Tensor): The inputs to the model, should be of the same length as the number of input_sizes.
        Returns:
            torch.Tensor: The output of the model, which is a scalar value with shape (batch_size, 1).
        """
        # loop over all the inputs and encode them to a common space | by using list comprehension we use less memory
        x = torch.cat([self.activ(self.encoders[i](inputs[i])) for i in range(len(inputs))], dim=-1)
        # Process the concatenated output into a common space
        x = self.concat_fc(x)
        # Apply ReLU activation function
        x = self.activ(x)
        # Map the output of concat_fc to a scalar output
        x = self.output_fc(x)
        return x

class MultiHeadRegressor(MultiHeadRegressorBase):
    def __init__(self, *input_sizes:int, hidden_size=128, activation="sigmoid",version = 1, dropout_prob=0.5):
        super().__init__(*input_sizes, hidden_size=hidden_size, activation=activation)
        
        if version not in [1, 2]:
            raise ValueError("Version is invalid, please choose from 1 or 2.")
                
        
        self.version = version
        if version == 2:
            self.encoders = self.encoders = nn.ModuleList([nn.Sequential(
                nn.Linear(in_size, hidden_size),
                #nn.Dropout(dropout_prob)
            ) for in_size in input_sizes])


            self.concat_fc = nn.Sequential(
                nn.Linear(len(input_sizes)*hidden_size, hidden_size),
                nn.Dropout(dropout_prob//2),
                self.activ,
                nn.Linear(hidden_size, hidden_size//4)
            )
            
            self.output_fc = nn.Sequential(
                nn.Linear(hidden_size//4, 1),
                nn.ReLU())


    def forward(self, *inputs:torch.Tensor) -> torch.Tensor:
        x = super().forward(*inputs)
        return x


if __name__ == "__main__":
    print('Checking device: ')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing Regressor...")
    x = torch.randn((32,1024)).to(device)
    y = torch.rand((32,12)).to(device)
    model = Regressor().to(device)
    z = model(x,y)
    print(z.shape)
    
    print("Testing MultiHeadRegressor...")
    x = torch.randn((32,1024)).to(device)
    y = torch.rand((32,12)).to(device)
    z = torch.rand((32, 5)).to(device)
    model = MultiHeadRegressor(1024, 12, 5, hidden_size=128).to(device)

    w = model(x,y,z)
    print(w.shape)
    
    
    # model = MultiHeadRegressor(1024, 12, 5, hidden_size=128).to(device)
    # for param in model.parameters():
    #     print(param.name, param.shape, param.device)
    # print("---------------------")
    # for buffer in model.buffers():
    #     print(buffer.name, buffer.shape, buffer.device)
    