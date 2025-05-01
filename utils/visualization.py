import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(X, y_true, y_pred=None, title="Time Series Trajectories"):
    """
    Plot true and predicted trajectories for 3D systems (e.g., Lorenz attractor).
    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(y_true[0, :, 0], y_true[0, :, 1], y_true[0, :, 2], label="True")
    ax.set_title("True Trajectory")
    
    if y_pred is not None:
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(y_pred[0, :, 0], y_pred[0, :, 1], y_pred[0, :, 2], label="Predicted", color='r')
        ax2.set_title("Predicted Trajectory")

    plt.tight_layout()
    plt.show()

