import numpy as np
from util import *
from math import sqrt

damp = 0.3

###########################################################################################################
# position dependent functions
###########################################################################################################

length = lambda a: np.linalg.norm(a)
dist = lambda a, b: length(a-b)
normalized = lambda a: a/(length(a)+0.000001)
zero = lambda _: np.array([0, 0, 0, 0])
zero_d = lambda _, d=damp: np.array([0, 0, 0, d])
gravity = lambda _: np.array([0, -9.81, 0, 0])
gravity_d = lambda _, d=damp: np.array([0, -9.81, 0, d])
P = lambda x, y, z: np.array([x, y, z])
F = lambda x, y, z, d=damp: np.array([x, y, z, d])
dir_to_f = lambda a, d=damp: np.array([a[0], a[1], a[2], d])


def sphere_mask(pos: np.ndarray, sphere_center: np.ndarray, radius: float = 0.25) -> np.ndarray:
    return 1.0 if dist(pos, sphere_center) < radius else 0.0


def choose_sphere(pos: np.ndarray, sphere_center: np.ndarray, in_f, out_f, radius: float = 0.25) -> np.ndarray:
    return in_f(pos) if dist(pos, sphere_center) < radius else out_f(pos)


def force_towards(pos: np.ndarray, center: np.ndarray, strength: float = 1, d: float = damp) -> np.ndarray:
    direction = normalized(center-pos)
    return dir_to_f(direction * strength, d)


def split(pos, fn=gravity, axis: int = 0, threshold: float = 0.5):
    return -fn(pos) if pos[axis] > threshold else -fn(pos)


def rotation_x(pos: np.ndarray, strength: float = 1, d: float = damp) -> np.ndarray:
    return np.array([0, (pos[2] - 0.5), -(pos[1] - 0.5), d])*2*strength


def rotation_y(pos: np.ndarray, strength: float = 1, d: float = damp) -> np.ndarray:
    return np.array([(pos[2] - 0.5), 0, -(pos[0] - 0.5), d])*2*strength


def rotation_z(pos: np.ndarray, strength: float = 1, d: float = damp) -> np.ndarray:
    # f(x, y) = (y, -x)
    return np.array([(pos[1] - 0.5), -(pos[0] - 0.5), 0, d])*2*strength


def rotation_inward(pos):
    # f(x, y) = (y-x, -x-y)
    # transform coordinates
    x = pos[0] - 0.5
    y = pos[1] - 0.5
    z = pos[2] - 0.5
    return np.array([y - x, -x - y, 0, 0])


def to_center(pos: np.ndarray) -> np.ndarray:
    return force_towards(pos, P(0.5, 0.5, 0.5), 10.0)


# frameFunctions = [zero, gravity, split, rotation_y, rotation_inward, to_center]

def collecting_sphere(pos: np.ndarray, height: float, inner_f=gravity) -> np.ndarray:
    center = P(0.5, height, 0.5)
    return choose_sphere(pos, center, inner_f,
                         lambda p: force_towards(p, center, 30, 4))


def spring(pos: np.ndarray, env_fn=gravity) -> np.ndarray:
    x = pos[0]
    y = pos[1]
    z = pos[2]

    if y > 0.5:
        return F(6,-12,0)

    center_dist_2d = sqrt((x-0.5)**2+(z-0.5)**2)

    if center_dist_2d > 0.05:
        if y > 0.2:
            return env_fn(pos)
        else:
            return env_fn(pos) + dir_to_f(-normalized(P(x, 0, z)-P(0.5, 0, 0.5)), 0) * 2

    if center_dist_2d > 0.03:
        return dir_to_f(-normalized(P(x, 0, z)-P(0.5, 0, 0.5)), damp*0.1) * 100

    return F(0,10,0,0)


frameFunctions = [
    zero,
    gravity,
    spring,
    gravity,
    gravity,
    lambda p: collecting_sphere(p, 0.25),
    lambda p: collecting_sphere(p, 0.25*1.25),
    lambda p: collecting_sphere(p, 0.25*1.5),
    lambda p: collecting_sphere(p, 0.25*1.75),
    lambda p: collecting_sphere(p, 0.5),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_y(p_, 10) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_y(p_, 30, 0) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_y(p_, 50, 0) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_y(p_, 50, 0) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_y(p_, 50, 0) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_y(p_, 50, 0) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_z(p_, 50, 0) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_z(p_, 50, 0) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_z(p_, 50, 0) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_z(p_, 50, 0) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_x(p_, 50, 0) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_x(p_, 50, 0) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_x(p_, 50, 0) + gravity(p_)),
    lambda p: collecting_sphere(p, 0.5, lambda p_: rotation_x(p_, 50, 0) + gravity(p_)),
    gravity

]

###########################################################################################################
# Output
###########################################################################################################


def force_at_position(frame, pos):
    return frameFunctions[frame](pos)


def task(frame):
    array = np.zeros((side_length, side_length, side_length, 4), dtype=np.single)
    print(f"start {frame}", flush=True)
    for x in range(side_length):
        for y in range(side_length):
            for z in range(side_length):
                array[x, y, z] = \
                    force_at_position(frame, np.array([x / (side_length - 1), y / (side_length - 1), z / (side_length - 1)]))

    print(f"calculated {frame}", flush=True)
    return frame, array


def main():
    from multiprocessing.pool import Pool
    frame_count = len(frameFunctions)

    array = np.zeros((frame_count, side_length, side_length, side_length, 4), dtype=np.single)

    with Pool() as pool:
        print("created pool")
        for i, result in pool.map(task, range(frame_count)):
            array[i] = result
            print(f"added {i}", flush=True)

    array.tofile("../res/force_fields/field.bin")


if __name__ == "__main__":
    main()

