import numpy as np


class NeuralNetwork:
    '''
    A simple feedforward neural network with ReLU activation for hidden layers
    and softmax activation for the output layer. It supports multi-class classification.
    This implementation uses batch gradient descent for training.
    Args:
        layers (list of int): Number of units per layer, including input and output.
        learning_rate (float): Learning rate for gradient descent.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each batch for gradient descent.
    '''
    def __init__(self, layers, learning_rate=0.01, epochs=30, batch_size=64):
        self.layers = layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = []  # List to store weight matrices
        self.biases = []   # List to store bias vectors
        self._init_weights()

    def _init_weights(self):
        '''Initialize weights with He initialization and biases to zero.'''
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            fan_out = self.layers[i + 1]
            # Kaiming initialization for ReLU activations: w ~ N(0, sqrt(2/fan_in))
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            # Biases initialized to zero
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    def _relu(self, x):
        '''ReLU activation.'''
        return np.maximum(0, x)     # Returns x for positive inputs, 0 otherwise

    def _relu_derivative(self, x):
        '''Derivative of ReLU for backpropagation.'''
        return (x > 0).astype(float)    # Returns 1 for positive inputs, 0 otherwise

    def _softmax(self, x):
        '''Softmax activation for multi-class output.'''
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _forward(self, X):
        '''Perform forward pass through the network.

        Returns:
            activations (list of np.ndarray): Activations at each layer.
            z_values (list of np.ndarray): Linear combinations before activation.
        '''
        activations = [X]
        z_values = []

        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            # Compute linear combination and apply activation
            z = activations[-1] @ self.weights[i] + self.biases[i]
            a = self._relu(z)   # ReLU activation for hidden layers
            z_values.append(z)
            activations.append(a)

        # Output layer with Softmax
        z = (activations[-1] @ self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        a = self._softmax(z)     # Softmax activation for output layer (multi-class classification)
        activations.append(a)

        return activations, z_values

    def _backward(self, X, y, activations, z_values):
        '''Perform backpropagation and update weights and biases.'''
        m = X.shape[0]

        # One-hot encode labels
        y_one_hot = np.zeros_like(activations[-1])
        y_one_hot[np.arange(m), y] = 1

        # Error at output layer
        delta = activations[-1] - y_one_hot

        # Initialize gradients
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            grads_w[i] = (activations[i].T @ delta) / m             # Gradient for weights
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / m   # Gradient for biases
            if i > 0:
                # Compute delta for previous layer if not the input layer
                delta = (delta @ self.weights[i].T) * self._relu_derivative(z_values[i - 1])

        # Update parameters
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_w[i]   # Update weights
            self.biases[i] -= self.lr * grads_b[i]    # Update biases

    def fit(self, X, y):
        '''Train the network on data X with labels y.'''
        for epoch in range(1, self.epochs + 1):
            # Shuffle data at each epoch
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)   # Shuffle indices to randomize data order
            X_shuffled, y_shuffled = X[indices], y[indices]

            # Batch gradient descent
            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                batch_X = X_shuffled[start:end]     # Get the current batch
                batch_y = y_shuffled[start:end]     # Get the corresponding labels
                # Forward pass to compute activations and z-values
                activations, z_values = self._forward(batch_X)
                # Backward pass to compute gradients and update weights
                self._backward(batch_X, batch_y, activations, z_values)

    def predict(self, X):
        '''Return predicted class labels for input X with soft probabilities.'''
        activations, _ = self._forward(X)
        return np.argmax(activations[-1], axis=1)

    def score(self, X, y):
        '''Compute accuracy score of the model on X with true labels y.'''
        preds = self.predict(X)
        return np.mean(preds == y) # Returns the fraction of correct predictions
