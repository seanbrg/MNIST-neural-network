import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure the output directory exists
os.makedirs('tests\graphs', exist_ok=True)

def plot_labels_test(
    x, y_data, y_labels,
    x_label, y_label_main,
    title, filename,
    x_labels=None,
    y2_data=None, y2_label=None,
    y2_color='red',
    x_ticks_rotation=0,
    x_scale='linear',
    figsize=(6, 4)
):
    """
    Plot and save a graph with optional custom X-tick labels and secondary Y-axis.

    Parameters:
        x (list of any): Values or category labels for the X-axis.
        y_data (list of np.ndarray): Primary Y-axis datasets (same length as x).
        y_labels (list of str): Legend labels for each primary dataset.
        x_label (str): Label for the X-axis.
        y_label_main (str): Label for the primary Y-axis.
        title (str): Graph title.
        filename (str): Filename to save the plot in 'graphs/' dir.
        x_labels (list of str, optional): Custom labels for X-ticks. If None, uses str(x).
        y2_data (np.ndarray, optional): Secondary Y-axis dataset (same length as x).
        y2_label (str, optional): Label for the secondary dataset legend.
        y2_color (str): Color for the secondary line. Defaults to 'red'.
        x_ticks_rotation (int): Rotation angle for X tick labels. Defaults to 0.
        x_scale (str): Scale for the X-axis ('linear' or 'log'). Defaults to 'linear'.
        figsize (tuple): Figure size as (width, height) in inches. Defaults to (6, 4).
    """
    # Determine plotting positions
    positions = np.arange(len(x))

    # Use provided custom labels or default to stringified x
    if x_labels is not None:
        if len(x_labels) != len(x):
            raise ValueError("Length of x_labels must match length of x")
        tick_labels = x_labels
    else:
        tick_labels = [f"{val:.2f}" for val in x]

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot primary datasets at categorical positions
    for y, label in zip(y_data, y_labels):
        ax1.plot(positions, y, marker='o', label=label)

    # Axis labels and title
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label_main, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xscale(x_scale)

    # Set X-ticks to positions and labels
    ax1.set_xticks(positions)
    ax1.set_xticklabels(tick_labels, rotation=x_ticks_rotation, ha='right')
    ax1.set_xlim(positions[0] - 0.1, positions[-1] + 0.1)

    # Automatic Y-tick spacing for accuracy plots
    if "accuracy" in y_label_main.lower():
        all_vals = np.concatenate(y_data)
        lo = np.floor(np.min(all_vals) * 1000) / 1000
        hi = np.ceil(np.max(all_vals) * 1000) / 1000
        ax1.set_yticks(np.arange(lo, hi + 0.002, 0.002))

    # Plot secondary dataset if provided
    if y2_data is not None:
        ax2 = ax1.twinx()
        ax2.plot(positions, y2_data, marker='s', color=y2_color, label=y2_label)
        ax2.set_ylabel(y2_label, color=y2_color)
        ax2.tick_params(axis='y', labelcolor=y2_color)
        ax2.legend(loc='best')

    # Add legend for primary datasets
    ax1.legend(loc='best')

    plt.title(title)
    plt.tight_layout()
    # Save and close
    plt.savefig(f"tests\graphs\{filename}")
    plt.close(fig)


def plot_params_test(results: dict, filename: str = "accuracy_bar_chart.png"):
    """
    Plot and save a grouped bar chart of accuracies for two network layer configurations.

    Args:
        results (dict): Dictionary mapping configuration labels to accuracies:
                        {label: {'1': acc1, '2': acc2}, ...}
        filename (str): Output path for the saved plot image.
    """
    # Extract labels and accuracy pairs
    labels = list(results.keys())
    acc1 = [results[label]['1'] for label in labels]
    acc2 = [results[label]['2'] for label in labels]

    x = np.arange(len(labels))  # label locations
    width = 0.35               # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, acc1, width, label='Layers config 1')
    ax.bar(x + width/2, acc2, width, label='Layers config 2')

    # Labels and styling
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparison of Test Accuracies by Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0.94, 0.98)
    ax.set_yticks(np.arange(0.94, 1, 0.01))  # ticks every 0.01

    plt.tight_layout()
    plt.savefig(f"tests\graphs\{filename}")
    plt.close()