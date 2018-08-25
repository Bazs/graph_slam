import numpy as np


def graph_slam_reduce(xi, omega, landmark_estimates):
    """
    Incorporates information from measurements into the state information submatrix/subvector of omega/xi.
    :param xi: The information vector of [x m].T shape.
    :param omega: The information matrix of [[x, x_m], [m_x, m]] shape.
    :param landmark_estimates: A dictionary, from landmark index to landmark estimate.
    :return: The reduced information vector and information matrix as a tuple.
    """
    num_landmarks = len(landmark_estimates)
    state_information_size = xi.shape[0] - 3 * num_landmarks

    omega_reduced = np.copy(omega[:state_information_size, :state_information_size])
    xi_reduced = np.copy(xi[:state_information_size])

    landmark_indices = sorted(landmark_estimates.keys())

    for landmark_index in landmark_indices:
        landmark_start_index = omega.shape[0] + (landmark_index - num_landmarks) * 3
        landmark_end_index = landmark_start_index + 3

        omega_j_j_inv = np.linalg.inv(omega[landmark_start_index:landmark_end_index,
                                      landmark_start_index:landmark_end_index])

        omega_t_j = omega[:state_information_size, landmark_start_index:landmark_end_index]

        omega_t_j_dot_j_j_inv = np.dot(omega_t_j, omega_j_j_inv)

        omega_j_t = omega[landmark_start_index:landmark_end_index, :state_information_size]
        omega_t_t_reduction = np.dot(omega_t_j_dot_j_j_inv, omega_j_t)

        omega_reduced -= omega_t_t_reduction

        xi_t_reduction = np.dot(omega_t_j_dot_j_j_inv, xi[landmark_start_index:landmark_end_index])
        xi_reduced -= xi_t_reduction

    return xi_reduced, omega_reduced
