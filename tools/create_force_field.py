import numpy as np


side_length = 129
frame_count = 2


def force_at_position(frame, pos):
    o = np.array([0, -9.81 if pos[0] > 0.5 else 9.81, 0, 0])

    if frame == 0:
        return o
    else:
        return -o


def main():
    array = np.zeros((frame_count, side_length, side_length, side_length, 4), dtype=np.single)

    for frame in range(frame_count):
        for x in range(side_length):
            for y in range(side_length):
                for z in range(side_length):
                    array[frame, x, y, z] = \
                        force_at_position(frame, np.array([(x+1) / side_length, (y+1) / side_length, (z+1) / side_length]))

    array.tofile("../res/force_fields/test.bin")


if __name__ == "__main__":
    main()
