import numpy as np
from numba import njit


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return np.exp(-x) / (1 + np.exp(-x))**2


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1 / np.cosh(x)**2


def identity(x):
    return x


def didentity(x):
    return np.ones(x.shape)


def relu(x):
    return np.fmax(x, 0)


def drelu(x):
    return np.sign(relu(x))


def elu(x, alpha=1):
    return x * (x > 0) + alpha * (np.exp(x) - 1) * (x < 0)


def delu(x, alpha=1):
    return (x > 0) + alpha * np.exp(x) * (x < 0)


@njit
def exp(x):
    n, N = x.shape
    ans = np.zeros((n, N))
    for k in range(N):
        y = x[:, k]
        ey = np.exp(y - np.max(y))
        S = np.sum(ey)
        for i in range(n):
            ans[i, k] = ey[i] / S
    return ans


@njit
def J_exp(x):
    n, N = x.shape
    ans = np.zeros((n, n, N))
    s = exp(x)
    for i in range(n):
        for j in range(n):
            for k in range(N):
                ans[i, j, k] = s[i, k] * (int(i == j) - s[j, k])
    return ans


def mean_square(y, y_t):
    return 1/2 * np.sum((y - y_t)**2) / y.shape[1]


def D_mean_square(y, y_t):
    return (y - y_t) / y.shape[1]


@njit
def log(y, y_t):
    N = y.shape[1]
    ans = 0
    for k in range(N):
        i = np.argmax(y_t[:, k])
        if y[i, k] != 0:
            ans -= np.log(y[i, k])
    return ans / N


@njit
def Dlog(y, y_t):
    n, N = y.shape
    ans = np.zeros((n, N))
    for k in range(N):
        i = np.argmax(y_t[:, k])
        ans[i, k] = -1 / y[i, k]
    return ans / N


functions = {
    'sigmoid': (sigmoid, dsigmoid),
    'tanh': (tanh, dtanh),
    'identity': (identity, didentity),
    'relu': (relu, drelu),
    'elu': (elu, delu),
    'exp': (exp, J_exp),    # also known as softmax
    'mean_square': (mean_square, D_mean_square),
    'log': (log, Dlog),     # also known as cross-entropy
}
