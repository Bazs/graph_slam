import numpy as np


class State:
    def __init__(self):
        self.array = np.zeros((3, 1), dtype='float')

    @property
    def x(self):
        return self.array[0]

    @property
    def y(self):
        return self.array[1]

    @property
    def th(self):
        return self.array[2]

    @x.setter
    def x(self, x):
        self.array[0] = x

    @y.setter
    def y(self, y):
        self.array[1] = y

    @th.setter
    def th(self, th):
        self.array[2] = th
