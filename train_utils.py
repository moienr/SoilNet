import torch
from torch import nn
from torch.utils.data import DataLoader
# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
from dataset.utils.utils import TextColors as tc
from tqdm import tqdm
import matplotlib.pyplot as plt

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

def train_step(model:nn.Module, data_loader:DataLoader, loss_fn:nn.Module, optimizer:torch.optim.Optimizer):
    model.train()
    # Setup train loss and train accuracy values
    train_loss = 0
    loop = tqdm(data_loader, leave=True)
    for batch, (X, y) in enumerate(loop):
        # Send data to target device
        X, y = X.to(device), y.to(device)
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
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y.unsqueeze(1))
            test_loss += loss.item()
            
            # if batch % 2 == 0:
            #     loss, current = loss.item(), batch * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    test_loss /= len(data_loader)
    if verbose:print(f"Test Loss: {test_loss:>8f}%")
    return test_loss


# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = RMSELoss(),
          epochs: int = 5,
          lr_scheduler: bool = None):
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        train_dataloader (torch.utils.data.DataLoader): _description_
        test_dataloader (torch.utils.data.DataLoader): _description_
        optimizer (torch.optim.Optimizer): _description_
        loss_fn (torch.nn.Module, optional): _description_. Defaults to RMSELoss().
        epochs (int, optional): _description_. Defaults to 5.
        lr_scheduler (bool, optional): _description_. Defaults to None. / plateau or step

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
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(1, epochs):
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