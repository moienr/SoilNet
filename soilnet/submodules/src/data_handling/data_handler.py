import torch.utils.data

from src.library import *


class DataHandler:
    def __init__(self):
        """DataHandler class.

        Args:
            data_folder (str): Path to the folder containing the data.
            frame_subfolder (str): Name of the feature to extract from the data.

        Attributes:
            dataset (ndarray): Data loaded from the folder.
        """

        self.dataset = None
        self.dataset_x = None
        self.dataset_y = None
        self.train_data = None
        self.test_data = None
        self.torch_dataset = None
        random.seed = 1

    def create_dataset(self):
        self.torch_dataset = torch.utils.data.TensorDataset(
            self.dataset_x, self.dataset_y
        )

    def split_data(self, train_split: float = 0.8):
        self.train_data, self.test_data = torch.utils.data.random_split(
            self.torch_dataset, [train_split, 1 - train_split]
        )

    @staticmethod
    def create_dataloader(dataset: torch.utils.data.TensorDataset,
                          batch_size: int) -> torch.utils.data.DataLoader:
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=True, batch_size=batch_size
        )
        return dataloader

