import math


def normalize_angle_pi_minus_pi(angle):
    while -math.pi > angle:
        angle = angle + math.pi * 2
    while math.pi < angle:
        angle = angle - math.pi * 2

    return angle
