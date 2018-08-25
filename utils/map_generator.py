import numpy as np

import random


def generate_random_coordinate(map_matrix):
    return random.randint(0, map_matrix.shape[1]-1), random.randint(0, map_matrix.shape[0]-1)


def generate_random_free_coordinate(map_matrix):
    location_is_free = False
    while location_is_free is False:
        x, y = generate_random_coordinate(map_matrix)
        location_is_free = 0 == map_matrix.item(y, x)

    return x, y


def generate_landmarks(map_matrix, landmark_count):
    random.seed(3)
    generated_count = 0

    while generated_count < landmark_count:
        x, y = generate_random_free_coordinate(map_matrix)
        map_matrix[y, x] = 1
        generated_count = generated_count + 1


def generate_ground_truth_map(map_height, map_width, landmark_count):
    """
    Returns an integer numpy matrix of shape (map_height, map_width). Zero values represent free space,
    one values represent landmarks. An obstacle_count amount of uniformly randomly sampled point-like landmarks are
    generated. Additionally returns a list of landmarks, which are represented with the [x, y, integer-signature].T
    shape.
    """
    # The ground truth map is discretized. Zero values mean free space, one values mean landmarks.
    ground_truth_map = np.zeros((map_height, map_width), dtype='int')

    generate_landmarks(ground_truth_map, landmark_count)

    landmarks = []

    landmark_coords = np.argwhere(ground_truth_map == 1)

    for landmark_coord in landmark_coords:
        landmarks.append(np.array([[landmark_coord[1], landmark_coord[0], 0]]).T)

    return ground_truth_map, landmarks
