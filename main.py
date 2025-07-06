from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from neural_network import NeuralNetwork


if __name__ == '__main__':
    '''Load MNIST from data folder, train the network, and evaluate accuracy.'''
    # Read and preprocess the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    # stack data + labels into one array, convert targets to ints
    data = np.column_stack((mnist.data, mnist.target.astype(np.int64)))
    # split off 10 000 examples for testing
    data_train, data_test = train_test_split(data, test_size=10000, random_state=42)
    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test = data_test[:, :-1], data_test[:, -1]

    # Normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Initialize and train the neural network
    nn = NeuralNetwork(
        layers=[784, 128, 64, 10], 
        learning_rate=0.25, 
        epochs=15, 
        batch_size=64)
    nn.fit(X_train, y_train)

    # Evaluate performance
    print('Test accuracy:', nn.score(X_test, y_test))