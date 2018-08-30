import numpy as np


def graph_slam_solve(xi_reduced, omega_reduced, xi, omega):
    """
    Recovers the mean and covariance of the path, and the mean of the landmarks from the information representation.
    :param xi_reduced: The reduced information vector, result of GraphSLAM reduce.
    :param omega_reduced: The reduced information matrix, result of GraphSLAM reduce.
    :param xi: The full information vector.
    :param omega: The full information matrix
    :return: Tuple of (mu_states, sigma_states, landmark_estimates). The first two elements are the mean and covariance
    of the path, the last element is a dictionary mapping landmark index to landmark estimate.
    """
    sigma_state = np.linalg.inv(omega_reduced)
    mu_state = np.dot(sigma_state, xi_reduced)

    state_information_size = xi_reduced.shape[0]
    num_landmarks = int((xi.shape[0] - state_information_size) / 3)

    landmark_estimates = dict()

    for landmark_index in range(num_landmarks):
        landmark_start_index = xi_reduced.shape[0] + landmark_index * 3
        landmark_end_index = landmark_start_index + 3

        omega_j_j_inv = np.linalg.inv(omega[landmark_start_index:landmark_end_index,
                                      landmark_start_index:landmark_end_index])
        xi_j = xi[landmark_start_index:landmark_end_index]
        omega_j_t = omega[landmark_start_index:landmark_end_index, :state_information_size]

        # According to table 11.4, line 6 in the book, the right side of the matrix product below should be xi_j +
        # np.dot(omega_j_t, mu_state). However, it only seems to work with subtraction instead of addition.
        landmark_estimate = np.dot(omega_j_j_inv, xi_j - np.dot(omega_j_t, mu_state))

        landmark_estimates[landmark_index] = landmark_estimate

    return mu_state, sigma_state, landmark_estimates


def recover_state_estimates(mu_state):
    state_estimates = []

    for state_index in range(int(mu_state.shape[0] / 3)):
        state_estimates.append(mu_state[state_index * 3:state_index * 3 + 3])

    return state_estimates
