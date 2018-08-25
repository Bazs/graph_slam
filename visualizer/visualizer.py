from graph_slam.graph_slam import GraphSlamState, generate_ground_truth_map, generate_ground_truth_path, \
    generate_measurements
from utils.plot_utils import plot_path, plot_measurements_for_state
from slam_parameters import *

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import random as rnd
import sys


class MainWindow(QDialog):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Button to trigger new path generation
        self.buttons = {
            "generate_path": QPushButton("Regenerate path"),
            "next_state": QPushButton("Next state"),
            "previous_state": QPushButton("Previous state")
        }

        self.buttons["generate_path"].clicked.connect(self.generate_path)
        self.buttons["next_state"].clicked.connect(self.next_state)
        self.buttons["previous_state"].clicked.connect(self.previous_state)

        # Attribute for holding the plot of the current path
        self.path_plot = None
        self.odometry_plot = None
        self.measurements_scatter = None
        self.current_state_idx = 0
        self.current_state_scatter = None

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        for button in self.buttons.values():
            layout.addWidget(button)
        self.setLayout(layout)

        self.slam_state = GraphSlamState()
        self._init_world()
        self._plot_world()

        # Set up truly random number generation for creating the ground truth path (if the system supports it)
        rnd.seed(self.slam_state.true_random_gen.random())
        self.generate_path()

    def _init_world(self):
        self.slam_state.ground_truth_map, self.slam_state.landmarks = \
            generate_ground_truth_map(MAP_HEIGHT, MAP_WIDTH, LANDMARK_COUNT)

    def _plot_world(self):
        plt.title("Ground truth map")
        plt.imshow(self.slam_state.ground_truth_map, origin='lower')

    def generate_path(self):
        self.remove_measurements_from_plot()
        self.current_state_idx = 0

        self.slam_state.ground_truth_states, self.slam_state.controls = generate_ground_truth_path(
            self.slam_state.ground_truth_map, max_velocity=MAX_VELOCITY, velocity_deviation=VELOCITY_DEVIATION,
            max_turn_rate=MAX_TURN_RATE, turn_rate_deviation=TURN_RATE_DEVIATION, step_count=STEP_COUNT,
            velocity_control_deviation=VELOCITY_CONTROL_DEVIATION,
            turn_rate_control_deviation=TURN_RATE_CONTROL_DEVIATION)

        self.slam_state.measurements = \
            generate_measurements(self.slam_state.ground_truth_states,
                                  self.slam_state.landmarks, max_sensing_range=MAX_SENSING_RANGE,
                                  sensing_range_deviation=SENSING_RANGE_DEVIATION,
                                  distance_deviation=DISTANCE_DEVIATION, heading_deviation=HEADING_DEVIATION)

        path_x = []
        path_y = []
        for state in self.slam_state.ground_truth_states:
            path_x.append(state[0])
            path_y.append(state[1])

        if self.path_plot is not None:
            self.path_plot.remove()
        if self.odometry_plot is not None:
            self.odometry_plot.remove()

        self.path_plot, = plot_path(self.slam_state.ground_truth_states, "C0")

        self.plot_measurements_for_current_state()
        # refresh canvas
        self.canvas.draw()

    def remove_measurements_from_plot(self):
        if self.measurements_scatter is not None:
            self.measurements_scatter.remove()
            self.measurements_scatter = None
        if self.current_state_scatter is not None:
            self.current_state_scatter.remove()
            self.current_state_scatter = None

    def plot_measurements_for_current_state(self):
        self.remove_measurements_from_plot()

        measurements_for_state = self.slam_state.measurements[self.current_state_idx]
        current_state = self.slam_state.ground_truth_states[self.current_state_idx]

        self.measurements_scatter, self.current_state_scatter = \
            plot_measurements_for_state(current_state, measurements_for_state)

    def previous_state(self):
        if self.current_state_idx == 0:
            return
        self.current_state_idx = self.current_state_idx - 1
        self.plot_measurements_for_current_state()
        self.canvas.draw()

    def next_state(self):
        if self.current_state_idx == STEP_COUNT - 1:
            return
        self.current_state_idx = self.current_state_idx + 1
        self.plot_measurements_for_current_state()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = MainWindow()
    main.show()

    sys.exit(app.exec_())
