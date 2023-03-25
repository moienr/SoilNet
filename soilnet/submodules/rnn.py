import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the RNN module.

        Args:
        - input_size (int): the size of the input tensor
        - hidden_size (int): the number of hidden units in the RNN layer
        - num_layers (int): the number of RNN layers in the network
        - output_size (int): the number of output units in the linear layer
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Define the forward pass of the RNN module.

        Args:
        - x (torch.Tensor): the input tensor of shape (batch_size, seq_length, input_size)

        Returns:
        - out (torch.Tensor): the output tensor of shape (batch_size, num_classes)
        """
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) .to(device)
        # x: (n, seq, input), 
        # h0: (rnn_layers, n, hiden_size)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        return out
    
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the GRU module.

        Args:
        - input_size (int): the size of the input tensor
        - hidden_size (int): the number of hidden units in the GRU layer
        - num_layers (int): the number of GRU layers in the network
        - output_size (int): the number of output units in the linear layer
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Define the forward pass of the GRU module.

        Args:
        - x (torch.Tensor): the input tensor of shape (batch_size, seq_length, input_size)

        Returns:
        - out (torch.Tensor): the output tensor of shape (batch_size, num_classes)
        """
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) .to(device)
        # x: (n, seq, input), 
        # h0: (gru_layers, n, hiden_size)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)  
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        return out
    
    
    
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the LSTM model.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers. E.g., setting num_layers=2 would mean stacking
                two LSTMs together to form a `stacked LSTM`, with the second LSTM taking in outputs of the
                first LSTM and producing the final results.
            output_size (int): The size of the output.

        """
   
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward propagate the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)

        """

        # Set initial hidden states and cell states for LSTM | the deufault is zero so we could skip this step
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # x: (n, seq, input)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # h0 & c0: (lstm_layers, n, hiden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x)   # out: tensor of shape (batch_size, seq_length, hidden_size)
        #out, _ = self.lstm(x, (h0,c0))  # if h0 and c0 are defined, we need to pass them to the forward function
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :] # out: (n, 128)
        
        out = self.fc(out) # out: (n, 10)
        
        return out

if __name__ == '__main__':
    print("Testing RNN...")
    model = RNN(1, 128, 2, 16)
    x = torch.randn(32, 60, 1)
    y = model(x)
    print(y.shape)
    #summary(model, input_size=(32, 60, 1), device='cpu')
    
    print("Testing GRU...")
    model = GRU(1, 128, 2, 16)
    x = torch.randn(32, 60, 1)
    y = model(x)
    print(y.shape)
    
    
    print("Testing LSTM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    model = LSTM(10, 128, 2, 64).to(device)  # Move the model to the device
    x = torch.randn(32, 12, 10).to(device)  # Move the input tensor to the device
    y = model(x)
    print(y.shape)