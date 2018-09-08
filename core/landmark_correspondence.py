import numpy as np

import math


def calculate_correspondence_probability(omega, sigma_states, landmark_estimates, j, k):
    state_information_size = sigma_states.shape[0]
    j_start_index = state_information_size + j * 3
    j_end_index = j_start_index + 3
    k_start_index = state_information_size + k * 3
    k_end_index = k_start_index + 3

    omega_j_k_0 = np.concatenate((omega[j_start_index:j_end_index, j_start_index:j_end_index], np.zeros((3, 3))),
                                 axis=1)
    omega_j_k_1 = np.concatenate((np.zeros((3, 3)), omega[k_start_index:k_end_index, k_start_index:k_end_index]),
                                 axis=1)
    omega_j_k = np.concatenate((omega_j_k_0, omega_j_k_1))

    omega_j_k_t = np.concatenate((omega[j_start_index:j_end_index, :state_information_size],
                                  omega[k_start_index:k_end_index, :state_information_size]))

    omega_t_j_k = np.concatenate((omega[:state_information_size, j_start_index:j_end_index],
                                  omega[:state_information_size, k_start_index:k_end_index]), axis=1)

    omega_j_k = omega_j_k - np.dot(np.dot(omega_j_k_t, sigma_states), omega_t_j_k)

    mu_j_k = np.concatenate((landmark_estimates[j], landmark_estimates[k]))
    xi_j_k = np.dot(omega_j_k, mu_j_k)

    delta_matrix = np.concatenate((np.identity(3), -np.identity(3)))

    omega_delta_j_k = np.dot(np.dot(delta_matrix.T, omega_j_k), delta_matrix)
    xi_delta_j_k = np.dot(delta_matrix.T, xi_j_k)

    sigma_delta_j_k = np.linalg.inv(omega_delta_j_k)
    mu_delta_j_k = np.dot(sigma_delta_j_k, xi_delta_j_k)

    # TODO fix calculation
    normalizer = 1 / math.sqrt(np.linalg.norm(2 * math.pi * sigma_delta_j_k))

    return math.exp(-1/2 * np.dot(np.dot(mu_delta_j_k.T, sigma_delta_j_k), mu_delta_j_k))

