import numpy as np


side_length = 129


def create_force_field_with_function(frame_count, function):
    array = np.zeros((frame_count, side_length, side_length, side_length, 4), dtype=np.single)

    for frame in range(frame_count):
        print(f"{frame+1}/{frame_count}")
        for x in range(side_length):
            for y in range(side_length):
                for z in range(side_length):
                    array[frame, x, y, z] = \
                        function(frame, np.array([x / (side_length - 1), y / (side_length - 1), z / (side_length - 1)]))

    array.tofile("../res/force_fields/field.bin")

