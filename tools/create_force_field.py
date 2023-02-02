import numpy as np
from util import *
# unused
frame_count = 5

###########################################################################################################
# position dependent functions
###########################################################################################################
def noGravity(pos):
    return np.array([0, 0, 0, 0])

def gravity(pos):
    return np.array([0, -9.81, 0, 0])

def split(pos):
    return np.array([0, -9.81 if pos[0] > 0.5 else 9.81, 0, 0])

def rotation(pos):
    # f(x, y) = (y, -x)
    return np.array([(pos[1] - 0.5), -(pos[0] - 0.5), 0, 0])

def rotationInward(pos):
    # f(x, y) = (y-x, -x-y)
    # transform coordinates
    x = pos[0] - 0.5;
    y = pos[1] - 0.5;
    z = pos[2] - 0.5;
    return np.array([y-x, -x-y, 0, 0])

def center(pos):
    # forces directed towards the center
    return np.array([0.5 - pos[0], 0.5 - pos[1], 0.5 - pos[2], 0]) 

frameFunctions = [ noGravity, gravity, split, rotation, rotationInward, center ]

###########################################################################################################
# Output
###########################################################################################################
def force_at_position(frame, pos):
    return frameFunctions[frame](pos)


if __name__ == "__main__":
    create_force_field_with_function(len(frameFunctions), force_at_position)
