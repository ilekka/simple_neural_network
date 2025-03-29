### A simple handmade neural network

Dependencies: `numpy`, `numba` and `matplotlib`. Usage is illustrated by the
example scripts `oscillator.py` and `digits.py`.

### Contents

* **`network.py`**\
Defines a `Network` class implementing the neural network.
* **`functions.py`**\
Collection of activation functions and loss functions.
* **`oscillator.py`**\
A very simple example. Given $p$ and $q$ (each in the interval $[-1, 1]$),
predicts $\tfrac{1}{2}(p^2 + q^2)$.
* **`digits.py`**\
Handwritten digit recognition using the MNIST dataset. Before running this
script you must download the dataset in .csv format (for example from
https://pjreddie.com/projects/mnist-in-csv/).
* **`utils.py`**\
Utility functions for processing the handwritten digits data.
