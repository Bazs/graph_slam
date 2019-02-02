import math


def normalize_angle_pi_minus_pi(angle: float) -> float:
    """
    Returns the representation of the input angle within the [-PI, PI) range.

    >>> '%.3f' % normalize_angle_pi_minus_pi(-math.pi)
    '-3.142'
    >>> '%.3f' % normalize_angle_pi_minus_pi(math.pi)
    '-3.142'
    >>> '%.3f' % normalize_angle_pi_minus_pi(2 * math.pi)
    '0.000'
    >>> '%.3f' % normalize_angle_pi_minus_pi(-2 * math.pi)
    '0.000'
    >>> '%.3f' % normalize_angle_pi_minus_pi(math.pi + 0.2)
    '-2.942'
    >>> '%.3f' % normalize_angle_pi_minus_pi(-math.pi - 0.2)
    '2.942'

    :param angle: [rad]
    :return: [rad]
    """
    while -math.pi > angle:
        angle = angle + math.pi * 2
    while math.pi <= angle:
        angle = angle - math.pi * 2

    return angle


def angle_difference(angle_a: float, angle_b: float) -> float:
    """
    Returns the smallest representation of the angle difference between the two inputs, in radians. The returned value
    will be the angle, which will produce angle_a, when added to angle_b, considering the [-PI, PI) range representation
    of all angles. I.e. angle_b + ret_val = angle_a

    >>> '%.3f' % angle_difference(0, math.pi / 4)
    '-0.785'
    >>> '%.3f' % angle_difference(0, -math.pi / 4)
    '0.785'
    >>> '%.1f' % angle_difference(math.pi - 0.1, -math.pi + 0.1)
    '-0.2'
    >>> '%.1f' % angle_difference(-math.pi + 0.1, math.pi - 0.1)
    '0.2'
    >>> '%.3f' % angle_difference(0, math.pi)
    '-3.142'
    >>> '%.3f' % angle_difference(math.pi, 0) # The output is also in the [-PI, PI) range, hence the negative result
    '-3.142'

    :param angle_a: [rad]
    :param angle_b: [rad]
    :return: the difference, in the [-PI, PI) representation, in [rad]
    """
    angle_a_norm = normalize_angle_pi_minus_pi(angle_a)
    angle_b_norm = normalize_angle_pi_minus_pi(angle_b)
    diff = angle_a_norm - angle_b_norm

    return normalize_angle_pi_minus_pi(diff)
