import torch.optim
from src.library import *
from src.data_handling.data_handler import DataHandler


class Trainer:
    """Trainer class for the transformer model.

    Args:
        model: The model to train.
        dh: The data handler object.
        batch_size: The batch size.
        lr: The learning rate.
        betas: The betas for the Adam optimiser.
        eps: The epsilon for the Adam optimiser.
        epochs: The number of epochs to train for.
    """

    def __init__(
        self,
        dh: DataHandler,
        epochs: int = 10,
    ):
        self.criterion = nn.CrossEntropyLoss()

        self.dh = dh
        self.dataset = None
        self.train_data = None
        self.test_data = None
        self.n_epochs = epochs
        cuda_dev = "0"
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")

    def fit(self, dataloader: torch.utils.data.DataLoader, model: nn.Module, optimiser: torch.optim.Optimizer):
        losses = []
        model = model.to(self.device)
        model.train()
        model.double()
        for epoch in range(self.n_epochs):
            losses = self.train_one_epoch(dataloader=dataloader, epoch_no=epoch, losses=losses, optimiser=optimiser, model=model)

    def train_one_epoch(self, dataloader, epoch_no, losses, optimiser, model, disable_tqdm=False):
        epoch_loss = 0
        i = 0
        with tqdm(dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for idx, data in enumerate(tepoch):
                i += 1
                loss, losses = self._train_one_loop(data=data, losses=losses, model=model, optimiser=optimiser)
                epoch_loss += loss.detach()
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=epoch_loss.item() / i)
        return losses


    def _train_one_loop(
        self, data: torch.utils.data.DataLoader, losses: List[float], model: nn.Module, optimiser: torch.optim.Optimizer
    ) -> Tuple[float, List[float]]:

        optimiser.zero_grad()
        data[0] = data[0].double()
        padding_mask = torch.ones((data[0].shape[0], data[0].shape[1])) > 0
        output = model(data[0].to(self.device), padding_mask.to(self.device))
        loss = self.criterion(output, data[1].type(torch.DoubleTensor).to(self.device))
        loss.backward()
        optimiser.step()
        losses.append(loss.detach())
        return loss.detach(), losses

    def evaluate(self, dataloader: torch.utils.data.DataLoader, model: nn.Module):
        """Run the model on the test set and return the accuracy."""
        model.eval()
        n_correct = 0
        n_incorrect = 0
        for idx, data in enumerate(dataloader):
            padding_mask = torch.ones((data[0].shape[0], data[0].shape[1])) > 0

            output = model(data[0].to(self.device), padding_mask.to(self.device))
            predictions = torch.argmax(output, dim=1)
            target = torch.argmax(data[1], dim=1).to(self.device)
            incorrect = torch.count_nonzero(predictions - target)
            n_incorrect += incorrect.detach()
            n_correct += (len(target) - incorrect).detach()
        accuracy = n_correct / (n_correct + n_incorrect)
        return accuracy
