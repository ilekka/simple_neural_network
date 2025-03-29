import numpy as np
from network import Network

# Generate random data for training
N_samples = 10000
x = 1 - 2 * np.random.rand(2, N_samples)
y = np.sum(1/2 * x**2, axis=0, keepdims=True)

# Divide the data into training and testing data
N_train = 8000

idx_train = np.random.choice(range(N_samples), N_train, replace=False)
x_train = x[:, idx_train]
y_train = y[:, idx_train]

idx_test = [i for i in range(N_samples) if i not in idx_train]
x_test = x[:, idx_test]
y_test = y[:, idx_test]

# Initialize the network
n = Network(loss='mean_square')
n.add_layer((2, 10), function='tanh')
n.add_layer((10, 1), function='identity')

# Train the network. The value of the loss function on the training data is
# printed after each training cycle.
n.train(x_train, y_train, N=1000, r=0.1, batch_size=100, batches=100)

# Value of the loss function on the testing data
y_pred = n.transform(x_test)
print(n.loss(y_pred, y_test))

# Compare predictions and correct values for 50 randomly chosen examples of the
# testing data
idx = np.random.choice(y_pred.shape[1], 50, replace=False)
sample = np.concatenate((y_pred[:, idx], y_test[:, idx])).T
print(np.round(sample, 4))

# Visualize the parameters
n.plot_weights(layer=0)
n.plot_weights(layer=1)
