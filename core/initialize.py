from utils.ctrv_motion_model import calculate_odometry_from_controls


def graph_slam_initialize(controls, state_t0):
    """Initializes the mean pose vector by forward-propagating the controls using the motion model. The initial state
    is specified by state_t0.
    """
    initialized_states = [state_t0]

    for index, control in enumerate(controls[1:]):
        initialized_states.append(initialized_states[index] +
                                  calculate_odometry_from_controls(control.item(0), control.item(1),
                                                                   initialized_states[index]))

    return initialized_states
