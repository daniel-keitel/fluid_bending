import numpy as np
from util import *

frame_count = 5


def force_at_position(frame, pos):
    nog = np.array([0, 0, 0, 0])
    simple = np.array([0, -9.81, 0, 0])
    split = np.array([0, -9.81 if pos[0] > 0.5 else 9.81, 0, 0])

    return [nog, simple, -simple, split, -split][frame]


if __name__ == "__main__":
    create_force_field_with_function(frame_count, force_at_position)
