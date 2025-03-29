import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from functions import functions
# ruff: noqa: E741


class Layer:
    def __init__(self, shape, function, drop=0):
        self.W = 1 - 2 * np.random.rand(*shape).T
        self.b = 1 - 2 * np.random.rand(shape[1], 1)
        self.f, self.J = functions[function]
        self.drop = drop

    def transform(self, x, drop=True):
        self.x = x
        if drop and self.drop > 0:
            N = x.shape[0]
            n_drop = int(self.drop * N)
            idx = random.sample(range(N), n_drop)
            x[idx, :] = 0
            x *= 1 / (1 - n_drop / N)
        self.z = self.W @ x + self.b
        return self.f(self.z)


class Network:
    def __init__(self, loss):
        self.layers = []
        self.loss, self.dloss = functions[loss]

    def add_layer(self, shape, function, drop=0):
        self.layers.append(Layer(shape, function, drop))

    def transform(self, x, drop=True):
        for l in self.layers:
            x = l.transform(x, drop)
        return x

    def predict(self, x):
        y = self.transform(x, drop=False)
        prediction = np.argmax(y, axis=0)
        confidence = np.max(y, axis=0)
        return prediction, confidence

    def gradient(self, x, y_t):
        y = self.transform(x)
        dL = self.dloss(y, y_t)
        dW, db = [], []
        for l in reversed(self.layers):
            if (J := l.J(l.z)).ndim == 2:
                D = dL * J
            else:
                D = np.einsum('in,ijn->jn', dL, J)
            dW.append(D @ l.x.T)
            db.append(np.sum(D, axis=-1)[:, np.newaxis])
            dL = l.W.T @ D
        return dW[::-1], db[::-1]

    def train(self, X, Y, N, r, batch_size, batches):
        for i in range(N):
            for _ in range(batches):
                idx = random.sample(range(X.shape[1]), batch_size)
                x, y_t = X[:, idx], Y[:, idx]
                dW, db = self.gradient(x, y_t)
                for n, l in enumerate(self.layers):
                    l.W -= r * dW[n]
                    l.b -= r * db[n]
            Y_pred = self.transform(X, drop=False)
            print(i+1, self.loss(Y_pred, Y))

    def test(self, x, labels):
        prediction, _ = self.predict(x)
        correct = np.argmax(labels, axis=0)
        accuracy = np.sum(prediction == correct) / len(prediction)
        correct_idx = np.argwhere(prediction == correct).flatten()
        incorrect_idx = np.argwhere(prediction != correct).flatten()
        return accuracy, correct_idx, incorrect_idx

    def plot_weights(self, layer):
        W = self.layers[layer].W
        b = self.layers[layer].b
        img = np.concatenate((W, 100 * np.ones_like(b), b), axis=1)
        masked = np.ma.masked_where(img == 100, img)

        cmap = mpl.cm.bwr
        cmap.set_bad(color='tab:gray')
        vmax = max(np.max(np.abs(W)), np.max(np.abs(b)))

        plt.figure(facecolor='tab:gray')
        plt.imshow(masked, cmap=cmap, vmax=vmax, vmin=-vmax, aspect='equal')
        plt.axis('off')
        plt.colorbar()
        plt.show()
