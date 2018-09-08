from utils.angle_utils import normalize_angle_pi_minus_pi

import numpy as np

from collections import namedtuple, OrderedDict
import math
import random as rnd
from typing import List


def add_measurement_to_pose(pose, measurement):
    """
    Returns the position acquired by applying a [range, heading] measurement to a [x, y, heading] pose.
    :param pose: A pose in the form of [x, y, heading(rad)].
    :param measurement: A measurements in the form of [range, heading(rad)]
    :return: The resulting position as a (x, y) tuple.
    """
    x = pose[0] + math.cos(measurement[1] + pose[2]) * measurement[0]
    y = pose[1] + math.sin(measurement[1] + pose[2]) * measurement[0]

    return float(x), float(y)


def calculate_landmark_distance(pose, landmark):
    return np.linalg.norm(landmark[:2] - pose[:2])


def calculate_landmark_heading(pose, landmark):
    x_landmark = landmark[0]
    y_landmark = landmark[1]
    x_state = pose[0]
    y_state = pose[1]

    phi = math.atan2(y_landmark - y_state, x_landmark - x_state)
    phi = phi - pose[2]
    phi = normalize_angle_pi_minus_pi(phi)

    return phi


LandmarkDistanceIndex = namedtuple("LandmarkDistanceIndex", "landmark distance index")


def get_landmarks_in_range(ground_truth_state, landmarks, max_sensing_range):
    landmark_distances = [np.linalg.norm(landmark[:2] - ground_truth_state[:2]) for landmark in landmarks]
    return [LandmarkDistanceIndex(landmark, landmark_distances[index], index) for index, landmark in
            enumerate(landmarks) if landmark_distances[index] <= max_sensing_range]


def calculate_measurement_vector_for_detection(ground_truth_state, landmark_distance_index):
    return np.array([[landmark_distance_index.distance,
                     calculate_landmark_heading(ground_truth_state, landmark_distance_index.landmark),
                     landmark_distance_index.landmark[2]]]).T


def add_noise_to_measurements_for_state(measurements_for_state, distance_deviation, heading_deviation):
    for measurement in measurements_for_state:
        measurement[0] = measurement[0] + rnd.normalvariate(0, distance_deviation)
        measurement[1] = measurement[1] + rnd.normalvariate(0, heading_deviation)
        measurement[1] = normalize_angle_pi_minus_pi(measurement[1])

    return measurements_for_state


def compress_correspondences(correspondences: List[List[int]]) -> List[List[int]]:
    """
    Maps the given correspondence indices to a new set of indices, which start from zero (in order of appearance in the
    original correspondence lists), and are tightly increasing. NOTE: also modifies the input list!

    >>> compress_correspondences([[5, 8], [5, 9], [8, 9], [9, 5, 8]])
    [[0, 1], [0, 2], [1, 2], [2, 0, 1]]

    :param correspondences: list of correspondence lists for each state
    :return compressed correspondences
    """
    flat_correspondences = [correspondence for correspondences_for_state in correspondences for correspondence in
                            correspondences_for_state]

    original_correspondence_indices = list(OrderedDict.fromkeys(flat_correspondences))

    original_to_target = {original_correspondence_index: index for index, original_correspondence_index in
                          enumerate(original_correspondence_indices)}

    for index, original_correspondences_for_state in enumerate(correspondences):
        correspondences[index] = [original_to_target[original_correspondence] for original_correspondence in
                                  original_correspondences_for_state]

    return correspondences


def generate_measurements(ground_truth_states, landmarks, max_sensing_range, sensing_range_deviation,
                          distance_deviation, heading_deviation):
    """
    Generates a list of measurements for every state in ground_truth_states. Measurements are numpy arrays of
    [r, phi, s].T, r being the distance to landmark, phi is heading (positive X direction is 0 rad, grows
    counter-clockwise), s is the landmark descriptor - an integer. Measurements contain additional zero-mean gaussian
    noise for the distance and heading, as specified by distance_deviation and heading_deviation.

    For each state, only landmarks within max_sensing_range are considered. From this group of landmarks, detections
    are sampled using the absolute values of random samples from a zero mean, sensing_range_deviation normal
    distribution as distance threshold.
    """

    measurements = []
    correspondences = []

    for ground_truth_state in ground_truth_states:
        landmark_distance_index_list = get_landmarks_in_range(ground_truth_state, landmarks,
                                                              max_sensing_range)

        # Sample obstacles from the in-range ones, using the absolute values of samples from a zero-mean normal
        # distribution as distance thresholds
        landmark_distance_index_list = [landmark_distance_index for landmark_distance_index in
                                        landmark_distance_index_list if
                                        abs(rnd.normalvariate(0, sensing_range_deviation))
                                        >= landmark_distance_index.distance]

        measurements_for_state = [
            calculate_measurement_vector_for_detection(ground_truth_state, landmark_distance_index) for
            landmark_distance_index in landmark_distance_index_list]
        correspondences_for_state = [landmark_distance_index.index for landmark_distance_index in
                                     landmark_distance_index_list]

        add_noise_to_measurements_for_state(measurements_for_state, distance_deviation, heading_deviation)

        assert len(measurements_for_state) == len(correspondences_for_state)

        measurements.append(measurements_for_state)
        correspondences.append(correspondences_for_state)

    compress_correspondences(correspondences)

    assert len(measurements) == len(correspondences)
    for index, correspondence_for_state in enumerate(correspondences):
        assert len(measurements[index]) == len(correspondence_for_state)

    return measurements, correspondences
