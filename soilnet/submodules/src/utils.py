from src.library import *
from src.data_handling.data_handler import DataHandler


def get_data(train_path, test_path):
    torch.cuda.empty_cache()
    train_x, train_y = load_from_tsfile(
        train_path, return_data_type='numpy3d')
    train_x = torch.tensor(train_x)
    train_y_orig, train_y = np.unique(train_y, return_inverse=True)
    n_values = np.max(train_y) + 1
    train_y = np.eye(n_values)[train_y]

    test_x, test_y = load_from_tsfile(
        test_path, return_data_type='numpy3d')
    test_x = torch.tensor(test_x)
    test_y_orig, test_y = np.unique(test_y, return_inverse=True)
    n_values = np.max(test_y) + 1
    test_y = np.eye(n_values)[test_y]
    train_y = torch.tensor(train_y)
    test_y = torch.tensor(test_y)

    dh = DataHandler()
    data_x = torch.concat((train_x, test_x), dim=0).permute(0, 2, 1)
    data_y = torch.concat((train_y, test_y), dim=0)
    dh.dataset_x = data_x
    dh.dataset_y = data_y
    return dh

def get_activation_fn(activation: str):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise ValueError(f"Activation should be relu/gelu, not {activation}.")
