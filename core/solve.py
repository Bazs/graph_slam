import numpy as np


def graph_slam_solve(xi_reduced, omega_reduced, xi, omega):
    sigma = np.linalg.inv(omega)
    mu_states_map = np.dot(sigma, xi)

    state_information_size = xi_reduced.shape[0]
    num_landmarks = int((xi.shape[0] - state_information_size) / 3)

    mu_states = mu_states_map[:state_information_size]

    landmark_estimates = dict()

    for landmark_index in range(num_landmarks):
        landmark_start_index = xi.shape[0] + (landmark_index - num_landmarks) * 3
        landmark_end_index = landmark_start_index + 3

        landmark_estimate = mu_states_map[landmark_start_index: landmark_end_index]

        landmark_estimates[landmark_index] = landmark_estimate

    return mu_states, sigma[:state_information_size, :state_information_size], landmark_estimates

    # sigma_state = np.linalg.inv(omega_reduced)
    # mu_state = np.dot(sigma_state, xi_reduced)
    #
    # state_information_size = xi_reduced.shape[0]
    # num_landmarks = (xi.shape[0] - state_information_size) / 3
    #
    # landmark_estimates = dict()
    #
    # for landmark_index in range(num_landmarks):
    #     landmark_start_index = state_information_size + (landmark_index - num_landmarks) * 3
    #     landmark_end_index = landmark_start_index + 3
    #
    #     omega_j_j_inv = np.linalg.inv(omega[landmark_start_index:landmark_end_index, landmark_start_index,
    #                                   landmark_end_index])
    #
    #     xi_j = xi[landmark_start_index:landmark_end_index]
    #
    #     omega_j_t = omega[landmark_start_index:landmark_end_index, :state_information_size]
    #     mu_t = xi_reduced[]
    #
    #     landmark_estimate =