import matplotlib.pyplot as plt
import numpy as np


class Visualizer:

    def __init__(self, epochs: int, axis, title: str):
        self.axis = axis
        self.title = title
        self.primary = np.zeros(epochs)
        self.secondary = np.zeros(epochs)
        self.x = np.arange(epochs)
        self.count = 0

    def update_sets(self, primary_update: float, secondary_update: float):
        self.primary[self.count] = primary_update
        self.secondary[self.count] = secondary_update

    def plot_data(self, primary_update: float, secondary_update: float):
        self.update_sets(primary_update, secondary_update)
        self.axis.plot(self.x, self.primary, color="b")
        self.axis.plot(self.x, self.secondary, color="orange")
        self.axis.set_title(self.title)
        self.count += 1

    def draw_plot(self):
        plt.show()
        plt.pause(0.005)
        if self.count == len(self.x) - 1:
            plt.pause(3)
