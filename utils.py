import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', dtype=int)
    N = data.shape[0]
    images = 1/255 * data[:, 1:].T
    labels = np.zeros((10, N), dtype=int)
    labels[data[:, 0], np.arange(N)] = 1
    return images, labels


def show_image(i, images, labels, network=None):
    img = images[:, i].reshape(28, 28)
    label = np.argmax(labels[:, i])
    text = f'Number: {label}.'
    if network is not None:
        prediction, confidence = network.predict(images[:, i].reshape(784, 1))
        text += f'  Prediction: {prediction[0]} ({np.round(confidence[0], 3)})'
    fig = plt.figure()
    plt.imshow(img, cmap='Greys')
    fig.suptitle(text, fontsize=16)
    plt.axis('off')
    plt.show()
