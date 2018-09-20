from utils.measurement_model import get_landmarks_in_range, \
    calculate_measurement_vector_for_detection, LandmarkDistanceIndex

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
            LandmarkDistanceIndex(landmarks[0], landmark_sensing_range, 0),
            LandmarkDistanceIndex(landmarks[1], landmark_sensing_range, 1),
            LandmarkDistanceIndex(landmarks[2], landmark_sensing_range, 2),
            LandmarkDistanceIndex(landmarks[5], 0, 5),
        ]

        landmark_distance_index_list = get_landmarks_in_range(ground_truth_state, landmarks,
                                                              landmark_sensing_range)

        self.assertListEqual(expected, landmark_distance_index_list)

    def test_calculate_measurement_vector_for_detection(self):
        ground_truth_state = np.array([[20, 20, 0]]).T

        default_landmark_index = 0

        test_landmark_distance_index_list = [
            LandmarkDistanceIndex(np.array([[25, 20, 1]]).T, 5, default_landmark_index),
            LandmarkDistanceIndex(np.array([[20, 23, 1]]).T, 3, default_landmark_index),
            LandmarkDistanceIndex(np.array([[21, 21, 1]]).T, math.sqrt(2), default_landmark_index),
            LandmarkDistanceIndex(np.array([[15, 20, 1]]).T, 5, default_landmark_index),
            LandmarkDistanceIndex(np.array([[19, 19, 1]]).T, math.sqrt(2), default_landmark_index),
            ]

        expected = [
            np.array([[5, 0, 1]]).T,
            np.array([[3, math.pi / 2, 1]]).T,
            np.array([[math.sqrt(2), math.pi / 4, 1]]).T,
            np.array([[5, math.pi, 1]]).T,
            np.array([[math.sqrt(2), -3 * math.pi / 4, 1]]).T,
            ]

        for index, landmark_distance_index in enumerate(test_landmark_distance_index_list):
            measurement_vector = calculate_measurement_vector_for_detection(ground_truth_state, landmark_distance_index)
            self.assertTrue(np.allclose(expected[index], measurement_vector))


if __name__ == "__main__":
    unittest.main()
