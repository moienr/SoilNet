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
import torch.nn.functional as F

    
# SimCLR loss for contrastive learning -> SimCLR paper: https://arxiv.org/pdf/2002.05709.pdf   
class SimCLR(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        assert temperature > 0.0, "The temperature must be a positive float!"
        self.temperature = temperature

    def forward(self, feats1, feats2):
        # Concatenate two batches of features
        feats = torch.cat([feats1, feats2], dim=0)

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        
        # Ranking metrics
        acc_top1 = (sim_argsort == 0).float().mean()
        acc_top5 = (sim_argsort < 5).float().mean()
        acc_mean_pos = 1 + sim_argsort.float().mean()
        
        return nll, acc_top1, acc_top5, acc_mean_pos    
    
    
def test_SimCLR():
    from torch.distributions import uniform
    # Create instance of SimCLR model
    model = SimCLR(temperature=0.5)

    # Generate dummy features with uniform distribution
    dummy_feats1 = uniform.Uniform(0, 1).rsample((64, 128)) # 64 samples, 128 features
    dummy_feats2 = uniform.Uniform(0, 1).rsample((64, 128))

    # Forward pass
    nll, acc_top1, acc_top5, acc_mean_pos = model(dummy_feats1, dummy_feats2)

    # Validate outputs
    assert isinstance(nll, torch.Tensor)
    assert isinstance(acc_top1, torch.Tensor)
    assert isinstance(acc_top5, torch.Tensor)
    assert isinstance(acc_mean_pos, torch.Tensor)

    print("nll: ", nll, "\n", "acc_top1: ", acc_top1, "\n", "acc_top5: ", acc_top5, "\n", "acc_mean_pos: ", acc_mean_pos)
    # Print test passed
    print("Test passed!")
        

def train_step(model:nn.Module, data_loader:DataLoader, loss_fn:nn.Module, optimizer:torch.optim.Optimizer):
    model.train()
    # Setup train loss and train accuracy values
    train_loss = 0
    train_top1 = 0
    train_top5 = 0
    train_mean_pos = 0
    
    
    
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
        z_img, z_clim = model(X)


        # 2. Calculate  and accumulate loss
        loss, acc_top1, acc_top5, acc_mean_pos = loss_fn(z_img, z_clim)
        train_loss += loss.item() 
        train_top1 += acc_top1.item()
        train_top5 += acc_top5.item()
        train_mean_pos += acc_mean_pos.item()
        


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0 or batch == len(data_loader) - 1:
            loss= loss.item()
            loop.set_postfix(Train_Loss=train_loss / (batch+1))
            
    train_loss = train_loss / len(data_loader)
    train_top1 = train_top1 / len(data_loader)
    train_top5 = train_top5 / len(data_loader)
    train_mean_pos = train_mean_pos / len(data_loader)
    
    return train_loss, train_top1, train_top5, train_mean_pos


# Test step function
def test_step(model:nn.Module, data_loader:DataLoader, loss_fn:nn.Module, verbose = False):
    size = len(data_loader.dataset)
    model.eval()
    test_loss = 0
    test_top1 = 0
    test_top5 = 0
    test_mean_pos = 0
    
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
            z_img, z_clim = model(X)
            loss, acc_top1, acc_top5, acc_mean_pos = loss_fn(z_img, z_clim)# y_pred is of shape (batch_size, 1) and y is of shape (batch_size) -> unsqueeze y to (batch_size, 1)
            test_loss += loss.item()
            test_top1 += acc_top1.item()
            test_top5 += acc_top5.item()
            test_mean_pos += acc_mean_pos.item()
            
            
            # if batch % 2 == 0:
            #     loss, current = loss.item(), batch * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    test_loss /= len(data_loader)
    test_top1 /= len(data_loader)
    test_top5 /= len(data_loader)
    test_mean_pos /= len(data_loader)
    
    if verbose:
        print(f"Test Loss: {test_loss:>8f}%")
        print(z_img.shape, z_clim.shape)
    return test_loss, test_top1, test_top5, test_mean_pos


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
          loss_fn: torch.nn.Module = SimCLR(temperature=0.5),
          epochs: int = 5,
          lr_scheduler: bool = None,
          save_model_path = None,
          save_model_if_mae_lower_than = None,
          ):
    """ Train the model and test it on the test set
    Note: If you don't have diffrent validation and test sets, just pass the same dataloader for both test and val

    Args:
        model (torch.nn.Module): Pytorch model
        train_dataloader (torch.utils.data.DataLoader): train dataloader
        test_dataloader (torch.utils.data.DataLoader): test dataloader
        val_dataloader (torch.utils.data.DataLoader): validation dataloader
        optimizer (torch.optim.Optimizer): optimizer
        loss_fn (torch.nn.Module, optional): Loss funciton. Defaults to SimCLR(temperature=0.5).
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
                "train_acc_top1": [],
                "train_acc_top5": [],
                "train_acc_mean_pos": [],
                "val_loss": [],
                "val_acc_top1": [],
                "val_acc_top5": [],
                "val_acc_mean_pos": [],
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(1, epochs+1):
        print(tc.OKGREEN,f"Epoch {epoch}\n-------------------------------",tc.ENDC)
        train_loss,train_acc_top1, train_acc_top5, train_acc_mean_pos = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        val_loss,val_acc_top1, val_acc_top5, val_acc_mean_pos = test_step(model=model,
            data_loader=val_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            tc.OKCYAN,
            f"Epoch {epoch} Results: | ",
            f"train_loss: {train_loss} | ",
            f"val_loss: {val_loss} ",
            f"train_acc_top1: {train_acc_top1} | ",
            f"val_acc_top1: {val_acc_top1} ",
            f"train_acc_top5: {train_acc_top5} | ",
            f"val_acc_top5: {val_acc_top5} ",
            f"train_acc_mean_pos: {train_acc_mean_pos} | ",
            f"val_acc_mean_pos: {val_acc_mean_pos} ",
            tc.ENDC
        )
        print("")

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc_top1"].append(train_acc_top1)
        results["train_acc_top5"].append(train_acc_top5)
        results["train_acc_mean_pos"].append(train_acc_mean_pos)
        results["val_loss"].append(val_loss)
        results["val_acc_top1"].append(val_acc_top1)
        results["val_acc_top5"].append(val_acc_top5)
        results["val_acc_mean_pos"].append(val_acc_mean_pos)
        # 6. Update LR scheduler
        
        if lr_scheduler == "step":
            scheduler.step()
        elif lr_scheduler == "plateau":
            scheduler.step(train_loss)
        else:
            pass
    # results["MAE"].append(test_step(model=model, data_loader=test_dataloader, loss_fn=nn.L1Loss(), verbose=False))
    # results["RMSE"].append(test_step(model=model, data_loader=test_dataloader, loss_fn=RMSELoss(), verbose=False))
    # results["R2"].append(test_step(model=model, data_loader=test_dataloader, loss_fn=R2Score().to(device), verbose=False)) 
    # Save the model
    # if save_model_path:
    #     if save_model_if_mae_lower_than:
    #         if results["MAE"][-1] < save_model_if_mae_lower_than:
    #             save_checkpoint(model, optimizer, filename=save_model_path)
    #     else:
            # save_checkpoint(model, optimizer, filename=save_model_path)
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
    

if __name__ == "__main__":
    test_SimCLR()