import numpy as np

class NN_NoDecay:
    '''
    Fully-connected feed-forward neural network with no learning rate decay.

    Args:
        layers (list of int): Number of units per layer, including input and output.
        learning_rate (float): Learning rate for gradient descent.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each mini-batch for gradient descent.
        patience (int): Number of epochs with no improvement after which training will be stopped.
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

    def fit(self, X, y, X_val=None, y_val=None):
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



class NN_FlatDecay:
    '''
    Fully-connected feed-forward neural network with a flat learning rate decay.
    This decay reduces the learning rate by a flat factor at specified milestones.

    Args:
        layers (list of int): Number of units per layer, including input and output.
        learning_rate (float): Initial learning rate for gradient descent.
        epochs (int): Number of training epochs.
        decay (float): Flat decay factor for learning rate.
        milestones (list of int): Epochs at which to apply decay.
        batch_size (int): Size of each mini-batch for gradient descent.
    '''
    def __init__(self, layers, learning_rate=0.4, epochs=30, decay=0.6, milestones=[6,10], batch_size=64):
        self.layers = layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.decay = decay
        self.milestones = milestones  # List of epochs at which to apply decay
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

    def fit(self, X, y, X_val=None, y_val=None):
        '''Train the network on data X with labels y.'''
        if X_val is not None and y_val is not None: # Validate initial accuracy on validation set
            acc = self.score(X_val, y_val)
            best_acc = acc  # Store best accuracy for early stopping

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

            if epoch in self.milestones:
                self.lr *= self.decay  # Apply flat decay to learning rate

            if X_val is not None and y_val is not None:
                acc = self.score(X_val, y_val)
                if acc > best_acc + 0.0001:  # Check for improvement
                    best_acc = acc
                else:
                    break

    def predict(self, X):
        '''Return predicted class labels for input X with soft probabilities.'''
        activations, _ = self._forward(X)
        return np.argmax(activations[-1], axis=1)

    def score(self, X, y):
        '''Compute accuracy score of the model on X with true labels y.'''
        preds = self.predict(X)
        return np.mean(preds == y) # Returns the fraction of correct predictions
    

class NN_ExpDecay:
    '''
    Fully-connected feed-forward neural network with an exponential learning rate decay.

    Args:
        layers (list of int): Number of units per layer, including input and output.
        learning_rate (float): Initial learning rate for gradient descent.
        epochs (int): Number of training epochs.
        decay (float): Exponential decay factor for learning rate.
        batch_size (int): Size of each mini-batch for gradient descent.
    '''
    def __init__(self, layers, learning_rate=0.4, epochs=30, decay=0.95, batch_size=64):
        self.layers = layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.decay = decay
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

    def fit(self, X, y, X_val=None, y_val=None):
        '''Train the network on data X with labels y.'''
        if X_val is not None and y_val is not None: # Validate initial accuracy on validation set
            acc = self.score(X_val, y_val)
            best_acc = acc  # Store best accuracy for early stopping

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

            # Apply exponential decay to learning rate
            self.lr = self.lr * (self.decay ** epoch / self.epochs)

            if X_val is not None and y_val is not None:
                acc = self.score(X_val, y_val)
                if acc > best_acc + 0.0001:  # Check for improvement
                    best_acc = acc
                else:
                    break

    def predict(self, X):
        '''Return predicted class labels for input X with soft probabilities.'''
        activations, _ = self._forward(X)
        return np.argmax(activations[-1], axis=1)

    def score(self, X, y):
        '''Compute accuracy score of the model on X with true labels y.'''
        preds = self.predict(X)
        return np.mean(preds == y) # Returns the fraction of correct predictions
    

class NN_CosineDecay:
    '''
    Fully-connected feed-forward neural network with cosine learning rate decay.
    This decay reduces the learning rate following a cosine function, 
    which can help in fine-tuning the model towards the end of training.
    
    Args:
        layers (list of int): Number of units per layer, including input and output.
        learning_rate (float): Initial learning rate for gradient descent.
        min_lr (float): Minimum learning rate for cosine decay.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each mini-batch for gradient descent.
    '''
    def __init__(self, layers, learning_rate=0.4, min_lr=0.15, epochs=30, batch_size=64):
        self.layers = layers
        self.lr = learning_rate
        self.min_lr = min_lr  # Minimum learning rate for cosine decay
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

    def fit(self, X, y, X_val=None, y_val=None):
        '''Train the network on data X with labels y.'''
        if X_val is not None and y_val is not None: # Validate initial accuracy on validation set
            acc = self.score(X_val, y_val)
            best_acc = acc  # Store best accuracy for early stopping

        # Cosine decay parameters
        initial_lr = self.lr

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

            # Apply cosine decay to learning rate every epoch
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / self.epochs))
            self.lr = self.min_lr + (initial_lr - self.min_lr) * cosine_decay

            if X_val is not None and y_val is not None:
                acc = self.score(X_val, y_val)
                if acc > best_acc + 0.0001:  # Check for improvement
                    best_acc = acc
                else:
                    break

    def predict(self, X):
        '''Return predicted class labels for input X with soft probabilities.'''
        activations, _ = self._forward(X)
        return np.argmax(activations[-1], axis=1)

    def score(self, X, y):
        '''Compute accuracy score of the model on X with true labels y.'''
        preds = self.predict(X)
        return np.mean(preds == y) # Returns the fraction of correct predictions
    

class NN_CosineDecayMomentum:
    '''
    Fully-connected feed-forward neural network with cosine learning rate decay and momentum.
    This decay reduces the learning rate following a cosine function, and momentum is used to accelerate gradients.
    The momentum can lead to faster convergence.

    Args:
        layers (list of int): Number of units per layer, including input and output.
        learning_rate (float): Initial learning rate for gradient descent.
        min_lr (float): Minimum learning rate for cosine decay.
        momentum (float): Momentum factor for gradient updates.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each mini-batch for gradient descent.
    '''
    def __init__(self, layers, learning_rate=0.4, min_lr=0.15, momentum=0.9, epochs=30, batch_size=64):
        self.layers = layers
        self.lr = learning_rate
        self.min_lr = min_lr  # Minimum learning rate for cosine decay
        self.momentum = momentum
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
        self.vel_w = [np.zeros_like(w) for w in self.weights]
        self.vel_b = [np.zeros_like(b) for b in self.biases]


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
            # Compute gradients for weights and biases
            grads_w[i] = (activations[i].T @ delta) / m             # Gradient for weights
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / m   # Gradient for biases
            if i > 0:
                # Compute delta for previous layer if not the input layer
                delta = (delta @ self.weights[i].T) * self._relu_derivative(z_values[i - 1])

        # Update parameters
        for i in range(len(self.weights)):
            # Apply momentum to gradients
            self.vel_w[i] = self.momentum * self.vel_w[i] + self.learning_rate * grads_w[i]
            self.vel_b[i] = self.momentum * self.vel_b[i] + self.learning_rate * grads_b[i]
            # Update weights and biases using momentum
            self.weights[i] -= self.vel_w[i]
            self.biases[i]  -= self.vel_b[i]

    def fit(self, X, y, X_val=None, y_val=None):
        '''Train the network on data X with labels y.'''
        if X_val is not None and y_val is not None: # Validate initial accuracy on validation set
            acc = self.score(X_val, y_val)
            best_acc = acc  # Store best accuracy for early stopping

        # Cosine decay parameters
        initial_lr = self.lr

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

            # Apply cosine decay to learning rate every epoch
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / self.epochs))
            self.lr = self.min_lr + (initial_lr - self.min_lr) * cosine_decay

            if X_val is not None and y_val is not None:
                acc = self.score(X_val, y_val)
                if acc > best_acc + 0.0001:  # Check for improvement
                    best_acc = acc
                else:
                    break

    def predict(self, X):
        '''Return predicted class labels for input X with soft probabilities.'''
        activations, _ = self._forward(X)
        return np.argmax(activations[-1], axis=1)

    def score(self, X, y):
        '''Compute accuracy score of the model on X with true labels y.'''
        preds = self.predict(X)
        return np.mean(preds == y) # Returns the fraction of correct predictions
