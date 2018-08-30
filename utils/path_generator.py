from utils.angle_utils import normalize_angle_pi_minus_pi
from utils.ctrv_motion_model import calculate_odometry_from_controls
from utils.map_generator import generate_random_free_coordinate

import numpy as np

import math
import random as rnd


def add_noise_to_control(control, velocity_control_deviation, turn_rate_control_deviation):
    noisy_control = np.zeros(control.shape)
    noisy_control[0] = control[0] + rnd.normalvariate(0, velocity_control_deviation)
    noisy_control[1] = control[1] + rnd.normalvariate(0, turn_rate_control_deviation)
    return noisy_control


def clip(value, minimum, maximum):
    return min(max(value, minimum), maximum)


def generate_ground_truth_path(ground_truth_map, max_velocity, velocity_deviation, max_turn_rate, turn_rate_deviation,
                               step_count, velocity_control_deviation, turn_rate_control_deviation):
    """Generates a random path using a constant velocity and turn rate motion model. Also returns the velocity and turn
    rate for each state, after adding some zero mean Gaussian noise to them.

    Generates a sequence of step_count consecutive [x, y, theta].T states. The velocity and turn rate is piecewise
    constant between two states, and they change by a random value at each state, which is sampled from a zero mean
    gaussian distribution, whose standard deviation is given by the parameters velocity_deviation and
    turn_rate_deviation. The velocity and turn rate are also limited to the range of [0, max_velocity] and
    [-max_turn_rate, max_turn_rate], respectively.
    Additionally, each generated node is ensured to be collision free with respect to the specified map.
    """
    start_x, start_y = generate_random_free_coordinate(ground_truth_map)
    start_x = start_x + rnd.random() - 1
    start_y = start_y + rnd.random() - 1
    start_th = math.pi * (2 * rnd.random() - 1)

    current_state = np.array([[start_x, start_y, start_th]]).T
    ground_truth_states = [current_state]

    # velocity and turn rate
    v, omega = 0, 0
    controls = [add_noise_to_control(np.array([[v, omega]]).T, velocity_control_deviation=velocity_control_deviation,
                turn_rate_control_deviation=turn_rate_control_deviation)]

    for step in range(step_count - 1):
        proposal_state_is_valid = False

        while proposal_state_is_valid is False:
            v_proposal = v + rnd.normalvariate(0, velocity_deviation)
            v_proposal = clip(v_proposal, 0, max_velocity)

            omega_proposal = omega + rnd.normalvariate(0, turn_rate_deviation)
            omega_proposal = clip(omega_proposal, -max_turn_rate, max_turn_rate)

            # CTRV motion model
            delta_pos = calculate_odometry_from_controls(v_proposal, omega_proposal, current_state)
            proposal_state = current_state + delta_pos

            proposal_x = round(proposal_state.item(0))
            proposal_y = round(proposal_state.item(1))

            proposal_state_is_within_map = (0 <= proposal_y < ground_truth_map.shape[0] and
                                            0 <= proposal_x < ground_truth_map.shape[1])
            proposal_state_is_valid = (proposal_state_is_within_map and
                                       0 == ground_truth_map.item(proposal_y, proposal_x))

        v = v_proposal
        omega = omega_proposal

        controls.append(add_noise_to_control(np.array([[v, omega]]).T,
                                             velocity_control_deviation=velocity_control_deviation,
                                             turn_rate_control_deviation=turn_rate_control_deviation))

        proposal_state[2] = normalize_angle_pi_minus_pi(proposal_state[2])

        ground_truth_states.append(proposal_state)
        current_state = proposal_state

    return ground_truth_states, controls
