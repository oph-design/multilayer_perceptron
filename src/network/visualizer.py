import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self, epochs: int, axes, title: str):
        self.axes = axes
        self.title = title
        self.primary = np.full(epochs, np.nan)
        self.secondary = np.full(epochs, np.nan)
        self.x = np.arange(epochs)
        self.count = 0

    def update_sets(self, primary_update: np.float_, secondary_update: np.float_):
        self.primary[self.count] = primary_update
        self.secondary[self.count] = secondary_update

    def plot_data(self, primary_update: np.float_, secondary_update: np.float_):
        self.update_sets(primary_update, secondary_update)
        self.axes.clear()
        self.axes.plot(
            self.x,
            self.secondary,
            color="orange",
            linestyle="dashed",
            label="validate",
        )
        self.axes.plot(self.x, self.primary, color="royalblue", label="train")
        self.axes.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.7)
        self.axes.set_title(self.title)
        self.axes.legend()
        self.count += 1

    def draw_plot(self):
        plt.show()
        plt.pause(0.005)
