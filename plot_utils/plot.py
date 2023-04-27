import numpy as np
import matplotlib.pyplot as plt


def plot_train_test_losses(train_losses:np.array, test_losses:np.array, title="Train Test Loss",
                           x_label="Epochs", y_label="RMSE", min_max_bounds= False,
                           y_lim=None, save_path=None)->None:
    """
    This function takes in train and test losses as inputs and plots them using matplotlib.

    Parameters:
    ---
    train_losses (numpy array): Array of train losses for each epoch. The shape of the array should be (num_runs, num_epochs)
    test_losses (numpy array): Array of test losses for each epoch. The shape of the array should be (num_runs, num_epochs)
    title (str): Title of the plot (default is "Train Test Loss")
    x_label (str): Label for the x-axis (default is "Epochs")
    y_label (str): Label for the y-axis (default is "RMSE")
    min_max_bounds (bool): If True, the plot shows minimum and maximum values of losses, if False, the plot shows mean and standard deviation of losses (default is False)
    y_lim (tuple): Limits for the y-axis (default is None)
    save_path (str): If provided, saves the plot at the given path (default is None)

    Returns:
    ---
    None

    Example Usage:
    ---
    plot_train_test_losses(train_losses, test_losses, title="Train Test Losses", x_label="Epochs", y_label="RMSE")
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 12
    mean_train_losses = np.mean(train_losses, axis=0)
    std_train_losses = np.std(train_losses, axis=0)
    mean_test_losses = np.mean(test_losses, axis=0)
    std_test_losses = np.std(test_losses, axis=0)
    if min_max_bounds:
        lower_train_losses = np.min(train_losses, axis=0)
        upper_train_losses = np.max(train_losses, axis=0)
        lower_test_losses = np.min(test_losses, axis=0)
        upper_test_losses = np.max(test_losses, axis=0)
    else:
        lower_train_losses = mean_train_losses - std_train_losses
        upper_train_losses = mean_train_losses + std_train_losses
        lower_test_losses = mean_test_losses - std_test_losses
        upper_test_losses = mean_test_losses + std_test_losses

    
    plt.plot(mean_train_losses, color='#33a9a5', linewidth=2, label='Train loss')
    plt.fill_between(range(train_losses.shape[1]), lower_train_losses, upper_train_losses, alpha=0.2, color='#33a9a5', edgecolor='none')

    plt.plot(mean_test_losses, color='#f27085', linewidth=2, label='Test loss')
    plt.fill_between(range(train_losses.shape[1]), lower_test_losses, upper_test_losses, alpha=0.2, color='#f27085', edgecolor='none')

    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_lim is not None:
        plt.ylim(y_lim)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    
    plt.show()


if __name__ == "__main__":
    train_losses = np.random.random((10, 100))
    test_losses = np.random.random((10, 100)) * 3

    plot_train_test_losses(train_losses, test_losses,y_lim=[0,3])
