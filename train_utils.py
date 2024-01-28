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

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

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
    

class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    def forward(self, predictions, actuals):
        """
        Compute the Root Mean Squared Logarithmic Error.
        
        Args:
            predictions (torch.Tensor): The predicted values.
            actuals (torch.Tensor): The actual values.
        
        Returns:
            torch.Tensor: The computed RMSLE value.
        """
        # Ensure predictions are greater than -1, as log(0) and negative values are undefined
        predictions = torch.clamp(predictions, min=-1 + 1e-9)
        actuals = torch.clamp(actuals, min=-1 + 1e-9)

        # Calculate the log loss
        log_diff = torch.log(predictions + 1) - torch.log(actuals + 1)
        squared_log_diff = torch.square(log_diff)

        # Return the square root of the mean of squared log differences
        return torch.sqrt(torch.mean(squared_log_diff))


    

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
            loss = loss_fn(y_pred, y.unsqueeze(1)) # y_pred is of shape (batch_size, 1) and y is of shape (batch_size) -> unsqueeze y to (batch_size, 1)
            test_loss += loss.item()
            
            # if batch % 2 == 0:
            #     loss, current = loss.item(), batch * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    test_loss /= len(data_loader)
    if verbose:
        print(f"Test Loss: {test_loss:>8f}%")
        print(y_pred.shape, y.shape)
    return test_loss



import pandas as pd

def test_step_w_id(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, csv_file: str = "test.csv", verbose: bool = False):
    size = len(data_loader.dataset)
    model.eval()
    test_loss = 0
    results = []  # Store results for CSV

    with torch.inference_mode():
        for batch, (X, y, point_id) in enumerate(data_loader):
            # Send data to target device
            if isinstance(X, tuple) or isinstance(X, list): # if it's a tuple, it has the climate data in it
                X = [tensor.to(device) for tensor in list(X)]
                y = y.to(device)
            elif isinstance(X, torch.Tensor): # if it's a tensor, it's only the Image data
                X, y = X.to(device), y.to(device)
            else:
                raise ValueError(f"Input of the network must be either a Tensor or a Tuple of Tensors but it is: {type(X)}")

            y_pred = model(X)
            loss = loss_fn(y_pred, y.unsqueeze(1))
            test_loss += loss.item()

            # Save results for CSV
            if csv_file:
                y_pred = y_pred.squeeze(1)  # Remove the extra dimension from y_pred
                for i in range(len(point_id)):
                    results.append({'point_id': point_id[i], 'y_real': y[i].item(), 'y_pred': y_pred[i].item()})

    test_loss /= len(data_loader)
    if verbose:
        print(f"Test Loss: {test_loss:>8f}%")
        print(y_pred.shape, y.shape)

    # Save CSV
    if csv_file:
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)

    #return test_loss


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint=> ", end="")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print("Done!")
    
def load_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("Loading checkpoint=> ", end="")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Done!")

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = RMSELoss(),
          epochs: int = 5,
          lr_scheduler: bool = None,
          save_model_path = None,
          save_model_if_mae_lower_than = None,
          save_train_data_metrics = False
          ):
    """ Train the model and test it on the test set
    Note: If you don't have diffrent validation and test sets, just pass the same dataloader for both test and val

    Args:
        model (torch.nn.Module): Pytorch model
        train_dataloader (torch.utils.data.DataLoader): train dataloader
        test_dataloader (torch.utils.data.DataLoader): test dataloader
        val_dataloader (torch.utils.data.DataLoader): validation dataloader
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
               "val_loss": [],
               "MAE": [],
               "RMSE": [],
               "R2": [],
               "train_MAE": [],
                "train_RMSE": [],
                "train_R2": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(1, epochs+1):
        print(tc.OKGREEN,f"Epoch {epoch}\n-------------------------------",tc.ENDC)
        train_loss = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        val_loss = test_step(model=model,
            data_loader=val_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            tc.OKCYAN,
            f"Epoch {epoch} Results: | ",
            f"train_loss: {train_loss} | ",
            f"val_loss: {val_loss} ",
            tc.ENDC
        )
        print("")

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        if lr_scheduler == "step":
            scheduler.step()
        elif lr_scheduler == "plateau":
            scheduler.step(train_loss)
        else:
            pass
    results["MAE"].append(test_step(model=model, data_loader=test_dataloader, loss_fn=nn.L1Loss(), verbose=False))
    results["RMSE"].append(test_step(model=model, data_loader=test_dataloader, loss_fn=RMSELoss(), verbose=False))
    results["R2"].append(test_step(model=model, data_loader=test_dataloader, loss_fn=R2Score().to(device), verbose=False)) 
    if save_train_data_metrics:
        results["train_MAE"].append(test_step(model=model, data_loader=train_dataloader, loss_fn=nn.L1Loss(), verbose=False))
        results["train_RMSE"].append(test_step(model=model, data_loader=train_dataloader, loss_fn=RMSELoss(), verbose=False))
        results["train_R2"].append(test_step(model=model, data_loader=train_dataloader, loss_fn=R2Score().to(device), verbose=False))
    # Save the model
    if save_model_path:
        if save_model_if_mae_lower_than:
            if results["MAE"][-1] < save_model_if_mae_lower_than:
                save_checkpoint(model, optimizer, filename=save_model_path)
        else:
            save_checkpoint(model, optimizer, filename=save_model_path)
    # 6. Return the filled results at the end of the epochs
    return results




def plot_losses(loss_dict):
    train_losses = loss_dict["train_loss"]
    val_losses = loss_dict["val_loss"]
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    
class BatchLoader(torch.utils.data.Dataset): 
    """ Takes in a Pytorch DataLoader and returns any batch using index
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __len__(self):
        return len(self.dataloader)

    def __call__(self, index):
        for i, batch in enumerate(self.dataloader):
            if i == index:
                return batch
        raise IndexError("Index out of range")
    



def evaluate_regression_metrics(y_true, y_pred):
    """Calculate multiple regression evaluation metrics."""
    # y_true = y_true * 87  # Multiply y_true by 87
    # y_pred = y_pred * 87  # Multiply y_pred by 87
    
    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate R2 (R-squared)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate RPIQ (Relative Prediction Interval Quality)
    y_std = np.std(y_true)
    rpiq = 1 - (rmse / y_std)
    
    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate MEC (Mean Error Correction)
    mec = np.mean(y_true - y_pred)
    
    # Calculate CCC (Concordance Correlation Coefficient)
    def concordance_correlation_coefficient(y_real, y_pred):
        # Raw data
        dct = {
            'y_real': y_real,
            'y_pred': y_pred
        }
        df = pd.DataFrame(dct)
        # Remove NaNs
        df = df.dropna()
        # Pearson product-moment correlation coefficients
        y_real = df['y_real']
        y_pred = df['y_pred']
        cor = np.corrcoef(y_real, y_pred)[0][1]
        # Means
        mean_real = np.mean(y_real)
        mean_pred = np.mean(y_pred)
        # Population variances
        var_real = np.var(y_real)
        var_pred = np.var(y_pred)
        # Population standard deviations
        sd_real = np.std(y_real)
        sd_pred = np.std(y_pred)
        # Calculate CCC
        numerator = 2 * cor * sd_real * sd_pred
        denominator = var_real + var_pred + (mean_real - mean_pred)**2

        return numerator / denominator
    
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    
    return rmse, r2, rpiq, mae, mec, ccc