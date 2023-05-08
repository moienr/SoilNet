import torch
from torch import nn
from torch.utils.data import DataLoader
# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
from dataset.utils.utils import TextColors as tc
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchmetrics import R2Score

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    
class R2Loss(nn.Module):
    """
    Calculates the R2 loss for regression problems.

    The R2 loss measures the proportion of variance in the dependent variable that can be explained by the independent
    variable. It is also known as the coefficient of determination.

    Args:
        None

    Shape:
        - Input: (batch_size, *)
        - Target: (batch_size, *)
        - Output: scalar value

    Attributes:
        mse (nn.MSELoss): Mean squared error loss

    Examples::
        >>> loss = R2Loss()
        >>> yhat = torch.tensor([1, 2, 3, 4])
        >>> y = torch.tensor([2, 4, 6, 8])
        >>> r2 = loss(yhat, y)
    """

    def __init__(self):
        """
        Initializes the R2Loss module.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        """
        Calculates the R2 loss for the given predictions and targets.

        Args:
            yhat (torch.Tensor): Predictions tensor of shape (batch_size, *)
            y (torch.Tensor): Targets tensor of shape (batch_size, *)

        Returns:
            torch.Tensor: Scalar tensor representing the R2 loss
        """
        ones = torch.ones_like(y)
        return 1 - (self.mse(yhat,y)/self.mse(y,ones*y.mean()))

def train_step(model:nn.Module, data_loader:DataLoader, loss_fn:nn.Module, optimizer:torch.optim.Optimizer):
    model.train()
    # Setup train loss and train accuracy values
    train_loss = 0
    loop = tqdm(data_loader, leave=True)
    for batch, (X, y) in enumerate(loop):
        # Send data to target device
        if isinstance(X, tuple) or isinstance(X, list): # if its a tuple it has the climate data in it
            X = [tensor.to(device) for tensor in list(X)]
            y = y.to(device)
        elif isinstance(X, torch.Tensor): # if its a tensor then its only the Image data
            X, y = X.to(device), y.to(device)
        else:
            raise ValueError(f"Input of the netowrk must be either a Tensor or a Tuple of Tensors but it is: {type(X)}")
        # 1. Forward pass
        y_pred = model(X)


        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y.unsqueeze(1))
        train_loss += loss.item() 


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0 or batch == len(data_loader) - 1:
            loss= loss.item()
            loop.set_postfix(Train_Loss=train_loss / (batch+1))
            
    train_loss = train_loss / len(data_loader)
    return train_loss


# Test step function
def test_step(model:nn.Module, data_loader:DataLoader, loss_fn:nn.Module, verbose = False):
    size = len(data_loader.dataset)
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            # Send data to target device
            if isinstance(X, tuple) or isinstance(X, list): # if its a tuple it has the climate data in it
                X = [tensor.to(device) for tensor in list(X)]
                y = y.to(device)
            elif isinstance(X, torch.Tensor): # if its a tensor then its only the Image data
                X, y = X.to(device), y.to(device)
            else:
                raise ValueError(f"Input of the netowrk must be either a Tensor or a Tuple of Tensors but it is: {type(X)}")
            y_pred = model(X)
            loss = loss_fn(y_pred, y.unsqueeze(1))
            test_loss += loss.item()
            
            # if batch % 2 == 0:
            #     loss, current = loss.item(), batch * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    test_loss /= len(data_loader)
    if verbose:
        print(f"Test Loss: {test_loss:>8f}%")
        print(y_pred.shape, y.shape)
    return test_loss


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = RMSELoss(),
          epochs: int = 5,
          lr_scheduler: bool = None,
          save_model_path = None
          ):
    """_summary_

    Args:
        model (torch.nn.Module): Pytorch model
        train_dataloader (torch.utils.data.DataLoader): train dataloader
        test_dataloader (torch.utils.data.DataLoader): test dataloader
        optimizer (torch.optim.Optimizer): optimizer
        loss_fn (torch.nn.Module, optional): Loss funciton. Defaults to RMSELoss().
        epochs (int, optional): Number of Epochs. Defaults to 5.
        lr_scheduler (bool, optional): Use LR scheduler. Defaults to None, Options are "plateau" or "step" . Defaults to None. / plateau or step
        save_model_path (str, optional): If given, saves the model with the given name and path. Defaults to None | Example: "my_checkpoint.pth.tar".

    Returns:
        _type_: _description_
    """
    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    elif lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2, verbose=True)
    else:
        pass
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "test_loss": [],
               "MAE": [],
               "R2": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(1, epochs+1):
        print(tc.OKGREEN,f"Epoch {epoch}\n-------------------------------",tc.ENDC)
        train_loss = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss = test_step(model=model,
            data_loader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            tc.OKCYAN,
            f"Epoch {epoch} Results: | ",
            f"train_loss: {train_loss} | ",
            f"test_loss: {test_loss} ",
            tc.ENDC
        )
        print("")

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        if lr_scheduler == "step":
            scheduler.step()
        elif lr_scheduler == "plateau":
            scheduler.step(train_loss)
        else:
            pass
    results["MAE"].append(test_step(model=model, data_loader=test_dataloader, loss_fn=nn.L1Loss(), verbose=False)) 
    results["R2"].append(test_step(model=model, data_loader=test_dataloader, loss_fn=R2Score(), verbose=False)) 
    # Save the model
    if save_model_path:
        save_checkpoint(model, optimizer, filename=save_model_path)
    # 6. Return the filled results at the end of the epochs
    return results




def plot_losses(loss_dict):
    train_losses = loss_dict["train_loss"]
    test_losses = loss_dict["test_loss"]
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.title("Training and Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()