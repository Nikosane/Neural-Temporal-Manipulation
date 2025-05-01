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
