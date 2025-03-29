import numpy as np
from network import Network
from utils import load_data, show_image

# Load the data
train_images, train_labels = load_data('mnist_train.csv')
test_images, test_labels = load_data('mnist_test.csv')

# Initialize the network. The optional argument `drop` activates dropout for
# training.
n = Network(loss='log')
n.add_layer((784, 100), function='elu', drop=0.1)
n.add_layer((100, 10), function='exp', drop=0.1)

# Look at 10 examples of the training data
for i in np.random.choice(60000, 10, replace=False):
    show_image(i, train_images, train_labels)

# Five rounds of 10 training cycles. The value of the loss function on the
# training data is printed after each cycle. After each round of 10 cycles the
# network's accuracy on the training and testing data is printed.
for _ in range(5):
    n.train(train_images, train_labels,
            N=10, r=0.1, batch_size=60, batches=1000)
    accuracy_train, _, _ = n.test(train_images, train_labels)
    accuracy_test, correct, incorrect = n.test(test_images, test_labels)
    print('\n', f'{accuracy_train:.4f}', f'{accuracy_test:.4f}', '\n')

# Value of the loss function on the testing data
prediction = n.transform(test_images, drop=False)
print(n.loss(prediction, test_labels))

# 10 examples of correct and incorrect predictions for the testing data
for i in np.random.choice(correct, 10, replace=False):
    show_image(i, test_images, test_labels, network=n)

for i in np.random.choice(incorrect, 10, replace=False):
    show_image(i, test_images, test_labels, network=n)
