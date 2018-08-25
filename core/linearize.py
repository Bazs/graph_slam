from utils.angle_utils import normalize_angle_pi_minus_pi
from utils.ctrv_motion_model import calculate_odometry_from_controls, calculate_jacobian_from_controls
from utils.measurement_model import add_measurement_to_pose

import numpy as np

import math


def initialize_xi_omega():
    # Initial state is extremely reliable, i.e. high values in the information matrix
    omega = np.identity(3, dtype="float") * 1000000
    xi = np.zeros((1, 1))

    return xi, omega


def ensure_square_matrix_size(matrix, size):
    if size > matrix.shape[0]:
        matrix = np.concatenate((matrix, np.zeros((matrix.shape[0], size - matrix.shape[1]))), axis=1)
        matrix = np.concatenate((matrix, np.zeros((size - matrix.shape[0], matrix.shape[1]))), axis=0)

    return matrix


def ensure_column_vector_length(vector, length):
    if length > vector.shape[0]:
        vector = np.concatenate((vector, np.zeros((length - vector.shape[0], 1))))

    return vector


def linearize_controls(xi, omega, R, state_estimates, controls):
    for index, control in enumerate(controls[1:]):
        v = control.item(0)
        om = control.item(1)
        previous_state = state_estimates[index]
        current_state_estimate = state_estimates[index] + calculate_odometry_from_controls(v, om, previous_state)

        # Calculate the 6-by-6 update for the information matrix omega
        Gt = np.identity(3)
        Gt[:, -1] = np.ravel(calculate_jacobian_from_controls(v, om, previous_state))
        Gt_exp_dot_R_inv = np.dot(np.concatenate((-Gt.T, np.identity(3))), np.linalg.inv(R))
        omega_xt_xt_1 = np.dot(Gt_exp_dot_R_inv, np.concatenate((-Gt, np.identity(3)), axis=1))
        xi_xt_xt_1 = np.dot(Gt_exp_dot_R_inv, current_state_estimate - np.dot(Gt, previous_state))

        start_index = index * 3
        end_index = index * 3 + 6

        omega = ensure_square_matrix_size(omega, end_index)
        omega[start_index:end_index, start_index:end_index] += omega_xt_xt_1

        xi = ensure_column_vector_length(xi, end_index)
        xi[start_index:end_index] += xi_xt_xt_1

    return xi, omega


def get_delta_q(landmark_estimate, current_state):
    delta = landmark_estimate[:2] - current_state[:2]
    q = float(np.dot(delta.T, delta))

    return delta, q


def linearize_measurements(xi, omega, Q, state_estimates, landmark_estimates, measurements, correspondences):
    for step_index, measurements_for_state in enumerate(measurements):

        current_state = state_estimates[step_index]

        # Precalculate indices into omega corresponding the state
        state_start_index = step_index * 3
        state_end_index = state_start_index + 3

        for landmark_index, landmark_measurement in enumerate(measurements_for_state):
            correspondence = correspondences[step_index][landmark_index]

            # The number of landmarks, for which there already exists an estimate
            num_landmarks = len(landmark_estimates)

            # Precalculate indices into omega corresponding to the measurement
            measurement_start_index = omega.shape[0] + (correspondence - num_landmarks) * 3
            measurement_end_index = measurement_start_index + 3

            # If this observation is not associated to any previously observed landmark
            if num_landmarks <= correspondence:
                x, y = add_measurement_to_pose(current_state, landmark_measurement)
                landmark_estimate = np.array([[x, y, landmark_measurement.item(2)]]).T

                landmark_estimates[correspondence] = landmark_estimate

                delta, q = get_delta_q(landmark_estimate, current_state)
                expected_measurement = landmark_measurement
            else:
                landmark_estimate = landmark_estimates[correspondence]
                delta, q = get_delta_q(landmark_estimate, current_state)
                expected_landmark_heading = math.atan2(delta[1], delta[0]) - current_state[2]
                expected_landmark_heading = normalize_angle_pi_minus_pi(expected_landmark_heading)
                expected_measurement = np.array([[
                    math.sqrt(q), expected_landmark_heading, landmark_estimate.item(2)]]).T

            q_root = math.sqrt(q)
            delta_x = float(delta[0])
            delta_y = float(delta[1])

            H = np.array([
                [-q_root * delta_x, -q_root * delta_y, 0, q_root * delta_x, q_root * delta_y, 0],
                [delta_y, -delta_x, -q, -delta_y, delta_x, 0],
                [0, 0, 0, 0, 0, q]
            ]) / q

            Ht_dot_Q_inv = np.dot(H.T, np.linalg.inv(Q))
            omega_xt_mj = np.dot(Ht_dot_Q_inv, H)

            state_and_landmark = np.concatenate((current_state, landmark_estimate))
            xi_xt_mj = np.dot(Ht_dot_Q_inv, (landmark_measurement - expected_measurement
                                             + np.dot(H, state_and_landmark)))

            omega = ensure_square_matrix_size(omega, measurement_end_index)

            # Add the top left submatrix to the state in omega
            omega[state_start_index:state_end_index, state_start_index:state_end_index] += omega_xt_mj[0:3, 0:3]
            # Add the bottom right submatrix to the measurement
            omega[measurement_start_index:measurement_end_index, measurement_start_index:measurement_end_index] += \
                omega_xt_mj[3:, 3:]
            # Add the off-diagonal submatrices to between the state and the measurements
            omega[measurement_start_index:measurement_end_index, state_start_index:state_end_index] += \
                omega_xt_mj[3:, 0:3]
            omega[state_start_index:state_end_index, measurement_start_index:measurement_end_index] += \
                omega_xt_mj[0:3, 3:]

            xi = ensure_column_vector_length(xi, measurement_end_index)

            # Add the upper half of the new information to the state in xi
            xi[state_start_index:state_end_index] += xi_xt_mj[:3]
            # Add the lower half to the landmark in xi
            xi[measurement_start_index:measurement_end_index] += xi_xt_mj[3:]

    return xi, omega, landmark_estimates


def graph_slam_linearize(state_estimates, landmark_estimates, controls, measurements, correspondences,
                         motion_error_covariance, measurement_noise_covariance):
    """
    Implements the GraphSLAM_linearize algorithm from Probabilistic Robotics Chapter 11.3.

    :param landmark_estimates: A dictionary mapping from correspondence index to a landmark estimate in the form of
        [x, y, signature].T.
    :param state_estimates: The current ego state estimates, list of [x, y, heading].T elements.
    :param controls: List of [v, omega].T controls, one for each element in state_estimates.
    :param measurements: A list of state_estimates length, each element being a list of [range, heading, classifier].T
        measurements.
    :param correspondences: A list of lists, same dimension as measurements. Each element is to the index of the real
        landmark, to which the element of measurements in the same position corresponds to.
    :param motion_error_covariance: The covariance of the motion error, diagonal elements are variances for
        [x, y, heading].
    :param measurement_noise_covariance: The covariance of the measurement noise, diagonal elements are variances for
        [x, y, signature].
    :return: The information vector xi, the information vector omega, and the updated number of landmarks being
        estimated tracked in them.
    """
    xi, omega = initialize_xi_omega()

    # The motion noise covariance
    R = motion_error_covariance
    Q = measurement_noise_covariance

    xi, omega = linearize_controls(xi, omega, R, state_estimates, controls)
    xi, omega, landmark_estimates = linearize_measurements(xi, omega, Q, state_estimates, landmark_estimates,
                                                           measurements, correspondences)

    return xi, omega, landmark_estimates
