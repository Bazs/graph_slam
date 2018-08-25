from utils.measurement_model import add_measurement_to_pose

import matplotlib.pyplot as plt


def plot_path(path_states, color):
    path_x = []
    path_y = []
    for state in path_states:
        path_x.append(state[0])
        path_y.append(state[1])

    return plt.plot(path_x, path_y, marker='o', c=color)


def plot_measurements_for_state(state, measurements):
    x_measurements = []
    y_measurements = []

    for measurement in measurements:
        x, y = add_measurement_to_pose(state, measurement)
        x_measurements.append(x)
        y_measurements.append(y)

    state_scatter = plt.scatter(state[0], state[1], s=100, c='green')
    measurements_scatter = plt.scatter(x_measurements, y_measurements, c="red")

    return state_scatter, measurements_scatter
