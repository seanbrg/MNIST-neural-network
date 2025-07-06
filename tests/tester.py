from sklearn.model_selection import train_test_split
import pandas as pd
from plot import plot_labels_test, plot_params_test
from models import NN_NoDecay, NN_FlatDecay, NN_ExpDecay, NN_CosineDecay, NN_CosineDecayMomentum

def layers_acc(layer_configs, X_train, y_train, X_test, y_test):
    """
    Train & evaluate a NeuralNetwork for each configuration in each depth group.

    Args:
        layer_configs (dict[str, tuple[list[int], ...]]):
           {
              "3 layers": (...),
              "4 layers": (...),
              "5 layers": (...)
            }
        X_train, y_train   : training set
        X_test, y_test     : test set

    Returns:
        accuracies (dict[str, list[float]]):
            mapping each depth label to a list of validation accuracies
    """
    # initialize empty lists for each depth
    accuracies = {depth: [] for depth in layer_configs}

    # Iterate over each depth label and its configurations
    for depth_label, configs in layer_configs.items():
        for cfg in configs: # each cfg is a list of layer sizes for the network
            nn = NN_NoDecay(layers=cfg)
            nn.fit(X_train, y_train)

            acc = nn.score(X_test, y_test)
            print(f"{depth_label} -> {cfg}: Acc = {acc:.4f}")
            accuracies[depth_label].append(acc)
    return accuracies


def test_layers(X_train, y_train, X_test, y_test):
    """
    Test different layer configurations and return accuracies.

    Args:
        X_train, y_train   : training set
        X_test, y_test     : test set
    """
    # Define layer‐depth groups and test them
    layer_configs = {
        "3 layers": ([784, 64, 10],         [784, 128, 10],         [784, 256, 10]),
        "4 layers": ([784, 64, 32, 10],     [784, 128, 64, 10],     [784, 256, 128, 10]),
        "5 layers": ([784, 64, 32, 16, 10], [784, 128, 64, 32, 10], [784, 256, 128, 64, 10]),
    }
    accuracies = layers_acc(layer_configs, X_train, y_train, X_test, y_test)

    # Prepare data for plotting:
    x = [1, 2, 3]  # x-axis: first hidden layer size (1, 2, 3)
    x_labels = ['64', '128', '256']  # x-axis labels for first hidden layer sizes
    y_data = [accuracies['3 layers'], accuracies['4 layers'], accuracies['5 layers']]
    y_labels = ['3 layers', '4 layers', '5 layers']

    # Plot accuracy curves
    plot_labels_test(
        x=x,
        y_data=y_data,
        y_labels=y_labels, x_labels=x_labels,
        x_label="First Hidden Layer Size",
        y_label_main="Test Accuracy",
        title="Accuracy By Depth (no Decay)",
        filename="network_depths_accuracy.png",
        figsize=(5, 4)
    )

def test_lr(X_train, y_train, X_test, y_test):
    """
    Test different learning rates and return accuracies.

    Args:
        X_train, y_train   : training set
        X_test, y_test     : test set
    """
    # Define learning rates to test
    learning_rates = [i * 0.05 for i in range(1, 20)]  # from 0.05 to 0.95 with step of 0.05
    accuracies = []

    for lr in learning_rates:
        nn = NN_NoDecay(layers=[784, 128, 64, 10], learning_rate=lr)
        nn.fit(X_train, y_train)
        acc = nn.score(X_test, y_test)
        accuracies.append(acc)

    # Plot the results
    plot_labels_test(
        x=learning_rates,
        y_data=[accuracies],
        y_labels=['Accuracy'],
        x_label="Learning Rate",
        y_label_main="Test Accuracy",
        title="Accuracy By Learning Rate (no Decay)",
        filename="learning_rate_accuracy.png",
        figsize=(10, 4)
    )

def test_params(params, X_train, y_train, X_test, y_test):
    """
    Test different hyperparameters and return accuracies.

    Args:
        X_train, y_train   : training set
        X_test, y_test     : test set
    """
    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize results dictionary and layer configurations
    results = {}
    layers1, layers2 = [784, 128, 64, 10], [784, 256, 128, 64, 10]

    # Iterate over each hyperparameter configuration
    for label, (nn_class, param) in params.items():
        if nn_class == NN_NoDecay:          # No decay case
            nn1 = NN_NoDecay(layers=layers1, learning_rate=param)
            nn2 = NN_NoDecay(layers=layers2, learning_rate=param)
        elif nn_class == NN_FlatDecay:      # Flat decay case
            nn1 = NN_FlatDecay(layers=layers1, milestones=param)
            nn2 = NN_FlatDecay(layers=layers2, milestones=param)
        elif nn_class == NN_ExpDecay:       # Exponential decay case
            nn1 = NN_ExpDecay(layers=layers1, decay=param)
            nn2 = NN_ExpDecay(layers=layers2, decay=param)
        elif nn_class == NN_CosineDecay:    # Cosine decay case
            nn1 = NN_CosineDecay(layers=layers1)
            nn2 = NN_CosineDecay(layers=layers2)

        # Fit the networks and evaluate accuracy
        nn1.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        nn2.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        acc1 = nn1.score(X_test, y_test)
        acc2 = nn2.score(X_test, y_test)
        print(f"{label} - Layers1: {layers1}, Acc: {acc1:.4f} | Layers2: {layers2}, Acc: {acc2:.4f}")
        results[label] = {  # Store accuracies for each configuration
            '1': acc1,
            '2': acc2
        }

    # Plot the results
    plot_params_test(results, filename="params_accuracy_comparison.png")


if __name__ == "__main__":
    # Load & split MNIST
    train = pd.read_csv("data/MNIST-train.csv").to_numpy()
    test  = pd.read_csv("data/MNIST-test.csv").to_numpy()
    X_train, y_train = train[:, :-1], train[:, -1].astype(int)
    X_test,  y_test  = test[:,  :-1], test[:,  -1].astype(int)

    # Normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Define hyperparameter configurations
    params = {
        "No Decay, lr=0.25": (NN_NoDecay, 0.25),
        "No Decay, lr=0.35": (NN_NoDecay, 0.35),
        "Flat Decay, milestones=[25]": (NN_FlatDecay, [25]),
        "Flat Decay, milestones=[24, 28]": (NN_FlatDecay, [24, 28]),
        "Exp Decay, decay=0.97": (NN_ExpDecay, 0.97),
        "Exp Decay, decay=0.99": (NN_ExpDecay, 0.99),
        "Cosine Decay": (NN_CosineDecay, None),
        "Cosine Decay with Momentum": (NN_CosineDecayMomentum, None)
    }

    #test_layers(X_train, y_train, X_test, y_test)
    #test_lr(X_train, y_train, X_test, y_test)
    test_params(params, X_train, y_train, X_test, y_test)