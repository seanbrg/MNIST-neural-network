import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def add_noise(X, factor=0.2):
    """
    Adds Gaussian noise to the input data.

    Args:
        X: Original input data.
        factor: Noise level to apply.
    """
    return np.clip(X + factor * np.random.randn(*X.shape), 0., 1.)

class Autoencoder:
    """
    A basic feedforward autoencoder for denoising input data.
    
    Args:
        layers (list of int): Number of units per layer, including input and output.
        learning_rate (float): Learning rate for gradient descent.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each batch for gradient descent.
    """

    def __init__(self, layers, lr=0.01, epochs=20, batch_size=64):
        self.lr = lr
        self.layers = layers
        self.weights = []
        self.biases  = []
        self.epochs = epochs
        self.batch_size = batch_size
        self._init_weights()

    def _init_weights(self):
        '''Initialize weights to small random values and biases to zero.'''
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            fan_out = self.layers[i + 1]
            w = np.random.randn(fan_in, fan_out) * 0.01 # Small random weights
            b = np.zeros((1, fan_out)) # Biases initialized to zero
            self.weights.append(w)
            self.biases.append(b)

    def _relu(self, x):
        '''ReLU activation.'''
        return np.maximum(0, x)     # Returns x for positive inputs, 0 otherwise

    def _relu_derivative(self, x):
        '''Derivative of ReLU.'''
        return (x > 0).astype(float)    # Returns 1 for positive inputs, 0 otherwise

    def _sigmoid(self, x):
        """sigmoid activation."""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        '''Derivative of sigmoid.'''
        s = self._sigmoid(x)
        return s * (1 - s)

    def _forward(self, X):
        '''Perform forward pass through the network.

        Returns:
            activations (list of np.ndarray): Activations at each layer.
            z_values (list of np.ndarray): Linear combinations before activation.
        '''
        z_values, activations = [], [X]
        for i in range(len(self.weights)-1):
            z_values.append(activations[-1] @ self.weights[i] + self.biases[i])
            activations.append(self._relu(z_values[-1]))
        z_values.append(activations[-1] @ self.weights[-1] + self.biases[-1])
        activations.append(self._sigmoid(z_values[-1]))
        return activations, z_values

    def _backward(self, y, z_values, activations):
        """
        Performs the backward pass and computes gradients.

        Returns:
            grads_w: Gradients of the weights.
            grads_b: Gradients of the biases.
        """
        grads_w, grads_b = [], []
        delta = (activations[-1] - y) * self._sigmoid_derivative(z_values[-1])
        for i in reversed(range(len(self.weights))):
            grads_w.insert(0, activations[i].T @ delta)
            grads_b.insert(0, np.sum(delta, axis=0, keepdims=True))
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._relu_derivative(Z[i-1])
        return grads_w, grads_b

    def fit(self, X_clean, X_noisy):
        """
        Trains the autoencoder using mini-batch gradient descent.

        Parameters:
        - X_clean: Clean input data (ground truth).
        - X_noisy: Noisy version of the input data.
        """
        n = X_clean.shape[0]
        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            X_clean, X_noisy = X_clean[idx], X_noisy[idx]
            for i in range(0, n, self.batch_size):
                xb, yb = X_noisy[i:i+self.batch_size], X_clean[i:i+self.batch_size]
                A, Z = self._forward(xb)
                dW, dB = self._backward(y=yb, z_values=Z, activations=A)
                for j in range(len(self.weights)):
                    self.weights[j] -= self.lr * dW[j]
                    self.biases[j]  -= self.lr * dB[j]
            print(f"Epoch {epoch+1}/{self.epochs} - MSE: {mean_squared_error(X_clean, self.predict(X_noisy)):.6f}")

    def predict(self, X):
        """ Predicts the output of the autoencoder for given input data."""
        activatios, _ = self._forward(X)
        return activatios[-1]


if __name__ == '__main__':
    # Load MNIST dataset
    data_train = pd.read_csv(r"data\MNIST-train.csv").to_numpy()
    data_test = pd.read_csv(r"data\MNIST-test.csv").to_numpy()

    # Split features and labels
    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test = data_test[:, :-1], data_test[:, -1]

    # Normalize data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Add noise
    X_train_noisy = add_noise(X_train, 0.2)
    X_test_noisy  = add_noise(X_test, 0.2)

    # Train Autoencoder
    autoencoder = Autoencoder(layers=[784, 128, 64, 32, 64, 128, 784], lr=0.01)
    autoencoder.fit(X_train, X_train_noisy, epochs=20)

    # Evaluate on test set
    reconstructed = autoencoder.predict(X_test_noisy)
    print(f"\nTest MSE: {mean_squared_error(X_test, reconstructed):.6f}")

