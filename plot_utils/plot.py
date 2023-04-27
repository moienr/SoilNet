import numpy as np
import matplotlib.pyplot as plt

train_losses = np.random.random((10, 100))
test_losses = np.random.random((10, 100)) * 3

mean_train_losses = np.mean(train_losses, axis=0)
std_train_losses = np.std(train_losses, axis=0)
mean_test_losses = np.mean(test_losses, axis=0)
std_test_losses = np.std(test_losses, axis=0)


lower_train_losses = mean_train_losses - std_train_losses
upper_train_losses = mean_train_losses + std_train_losses
lower_test_losses = mean_test_losses - std_test_losses
upper_test_losses = mean_test_losses + std_test_losses

plt.plot(mean_train_losses, color='#33a9a5', linewidth=2)
plt.fill_between(range(train_losses.shape[1]), lower_train_losses, upper_train_losses, alpha=0.2, color='#33a9a5', edgecolor='none')

plt.plot(mean_test_losses, color='#f27085', linewidth=2)
plt.fill_between(range(train_losses.shape[1]), lower_test_losses, upper_test_losses, alpha=0.2, color='#f27085', edgecolor='none')

plt.show()
