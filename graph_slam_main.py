from slam_parameters import *
from core.initialize import graph_slam_initialize
from core.linearize import graph_slam_linearize
from core.reduce import graph_slam_reduce
from core.solve import graph_slam_solve
from utils.map_generator import generate_ground_truth_map
from utils.measurement_model import generate_measurements
from utils.path_generator import generate_ground_truth_path
from utils.plot_utils import plot_path, plot_measurements_for_state

import matplotlib.pyplot as plt
import numpy as np

import random as rnd


class GraphSlamState(object):
    def __init__(self):
        self.ground_truth_map = np.empty((0, 0))
        self.landmarks = []

        self.ground_truth_states = []
        self.controls = []

        self.measurements = []

        self.initial_state_estimates = []

        self.true_random_gen = rnd.SystemRandom()


if __name__ == "__main__":
    ground_truth_map, landmarks = generate_ground_truth_map(MAP_HEIGHT, MAP_WIDTH, LANDMARK_COUNT)

    # Set up truly random number generation for creating the ground truth path (if the system supports it)
    true_random_gen = rnd.SystemRandom()
    rnd.seed(true_random_gen.random())

    ground_truth_states, controls = \
        generate_ground_truth_path(ground_truth_map, max_velocity=MAX_VELOCITY,
                                   velocity_deviation=VELOCITY_DEVIATION, max_turn_rate=MAX_TURN_RATE,
                                   turn_rate_deviation=TURN_RATE_DEVIATION, step_count=STEP_COUNT,
                                   velocity_control_deviation=VELOCITY_CONTROL_DEVIATION,
                                   turn_rate_control_deviation=TURN_RATE_CONTROL_DEVIATION)

    measurements, correspondences = generate_measurements(
        ground_truth_states, landmarks, max_sensing_range=MAX_SENSING_RANGE,
        sensing_range_deviation=SENSING_RANGE_DEVIATION, distance_deviation=DISTANCE_DEVIATION,
        heading_deviation=HEADING_DEVIATION)

    initial_state_estimates = graph_slam_initialize(controls, state_t0=ground_truth_states[0])

    landmark_estimates = dict()
    state_estimates = initial_state_estimates

    correspondence_index = 0
    correspondences = []

    for measurements_for_state in measurements:
        correspondences.append([index + correspondence_index for index, _ in enumerate(measurements_for_state)])
        correspondence_index = correspondence_index + len(measurements_for_state)

    R = np.identity(3) * 0.00001
    Q = np.identity(3) * 0.00001

    for iteration_index in range(5):
        xi, omega, landmark_estimates = \
            graph_slam_linearize(state_estimates=state_estimates, landmark_estimates=landmark_estimates,
                                 controls=controls, measurements=measurements, correspondences=correspondences,
                                 motion_error_covariance=R, measurement_noise_covariance=Q)

        xi_reduced, omega_reduced = graph_slam_reduce(xi, omega, landmark_estimates)
        state_estimates, sigma_states, landmark_estimates = graph_slam_solve(xi_reduced, omega_reduced, xi, omega)

    global_state_estimates = []

    for index, state_estimate in enumerate(state_estimates):
        global_state_estimates.append(np.concatenate((state_estimate[:2] + ground_truth_states[0][:2],
                                                     state_estimate[2].reshape((1, 1)))))

    plt.figure(figsize=[10, 5])
    plt.subplot(131)
    plt.title("Ground truth map")
    plt.imshow(ground_truth_map, origin='lower')

    plot_path(ground_truth_states, 'C0')
    plot_path(initial_state_estimates, 'C1')
    plot_path(global_state_estimates, 'C2')

    current_state = 1
    plot_measurements_for_state(ground_truth_states[current_state], measurements[current_state])

    plt.subplot(132)
    plt.title("Information matrix")
    omega_binary = omega != 0
    plt.imshow(omega_binary)

    plt.subplot(133)
    plt.title("Reduced information matrix")
    omega_reduced_binary = omega_reduced != 0
    plt.imshow(omega_reduced_binary)

    plt.show()
