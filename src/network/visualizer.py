import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self, epochs: int, axis, title: str):
        self.axis = axis
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
        self.axis.clear()
        self.axis.plot(
            self.x,
            self.secondary,
            color="orange",
            linestyle="dashed",
            label="validate",
        )
        self.axis.plot(self.x, self.primary, color="royalblue", label="train")
        self.axis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.7)
        self.axis.set_title(self.title)
        self.axis.legend()
        self.count += 1

    def draw_plot(self):
        plt.show()
        plt.pause(0.005)
