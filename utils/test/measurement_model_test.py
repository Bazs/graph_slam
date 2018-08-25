from utils.measurement_model import get_landmarks_and_distances_in_range, \
    calculate_measurement_vector_for_detection

import numpy as np

import math

import unittest


class TestMeasurementModel(unittest.TestCase):

    def test_get_landmarks_and_distances_in_range(self):
        ground_truth_state = np.array([[10, 10, 0]]).T

        landmark_sensing_range = 5
        diagonal_element = math.sqrt(landmark_sensing_range ** 2 / 2)

        landmarks = [
            # should be in range
            np.array([[ground_truth_state[0] + landmark_sensing_range, ground_truth_state[0], 1]]).T,
            np.array([[ground_truth_state[0], ground_truth_state[0] + landmark_sensing_range, 2]]).T,
            np.array([[ground_truth_state[0] + diagonal_element, ground_truth_state[0] + diagonal_element, 3]]).T,
            # should be out of range
            np.array([[ground_truth_state[0] + landmark_sensing_range + 0.1, ground_truth_state[0], 4]]).T,
            np.array([[ground_truth_state[0] + diagonal_element + 0.1, ground_truth_state[0] + diagonal_element, 5]]).T,
            # should be in range again
            np.array([[ground_truth_state[0], ground_truth_state[0], 6]]).T,
        ]

        expected = [
            (landmarks[0], landmark_sensing_range),
            (landmarks[1], landmark_sensing_range),
            (landmarks[2], landmark_sensing_range),
            (landmarks[5], 0),
        ]

        landmarks_and_distances_in_range = get_landmarks_and_distances_in_range(ground_truth_state, landmarks,
                                                                                landmark_sensing_range)

        self.assertListEqual(expected, landmarks_and_distances_in_range)

    def test_calculate_measurement_vector_for_detection(self):
        ground_truth_state = np.array([[20, 20, 0]]).T

        test_landmarks_and_distances = [
            (np.array([[25, 20, 1]]).T, 5),
            (np.array([[20, 23, 1]]).T, 3),
            (np.array([[21, 21, 1]]).T, math.sqrt(2)),
            (np.array([[15, 20, 1]]).T, 5),
            (np.array([[19, 19, 1]]).T, math.sqrt(2)),
            ]

        expected = [
            np.array([[5, 0, 1]]).T,
            np.array([[3, math.pi / 2, 1]]).T,
            np.array([[math.sqrt(2), math.pi / 4, 1]]).T,
            np.array([[5, math.pi, 1]]).T,
            np.array([[math.sqrt(2), -3 * math.pi / 4, 1]]).T,
            ]

        for index, landmark_and_distance in enumerate(test_landmarks_and_distances):
            measurement_vector = calculate_measurement_vector_for_detection(ground_truth_state, landmark_and_distance)
            self.assertTrue(np.allclose(expected[index], measurement_vector))


if __name__ == "__main__":
    unittest.main()
