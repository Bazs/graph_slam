from math import sin, cos

import numpy as np

ZERO_HEADING_THRESHOLD = 1e-6


def calculate_odometry_from_controls(v, omega, previous_state):
    """Calculates a pose delta using a constant turn rate and velocity motion model.
    """
    if ZERO_HEADING_THRESHOLD >= abs(omega):
        dx = v * cos(previous_state[2])
        dy = v * sin(previous_state[2])
        dth = omega
    else:
        dx = v / omega * (sin(omega + previous_state[2]) - sin(previous_state[2]))
        dy = v / omega * (-cos(omega + previous_state[2]) + cos(previous_state[2]))
        dth = omega
    return np.array([[dx, dy, dth]]).T


def calculate_jacobian_from_controls(v, omega, previous_state):
    if ZERO_HEADING_THRESHOLD >= abs(omega):
        x_dot = -v * sin(previous_state[2])
        y_dot = v * cos(previous_state[2])
        th_dot = 1
    else:
        x_dot = v / omega * (cos(omega + previous_state[2]) - cos(previous_state[2]))
        y_dot = v / omega * (sin(omega + previous_state[2] - sin(previous_state[2])))
        th_dot = 1
    return np.array([[x_dot, y_dot, th_dot]]).T
