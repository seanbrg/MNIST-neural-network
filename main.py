import pandas as pd
from neural_network import NeuralNetwork


if __name__ == '__main__':
    '''Load MNIST from data folder, train the network, and evaluate accuracy.'''
    # Read and preprocess the MNIST dataset
    data_train = pd.read_csv("data\MNIST-train.csv").to_numpy()
    data_test = pd.read_csv("data\MNIST-test.csv").to_numpy()
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