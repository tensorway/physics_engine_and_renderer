import numpy as np
import math
from utils import nonzero

def get_line_equation(points):
    div = points[1][1] - points[0][1]
    k = (points[1][0]-points[0][0])/nonzero(div)
    l = points[0][0] - k*points[0][1]
    return k, l

def get_plane_equation(points):
    a, b, c = points
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    d = a @ normal
    return normal, d

def rotate3d(points, alphax, alphay, alphaz):
    rx = np.array([
        [1, 0, 0],
        [0, math.cos(alphax), -math.sin(alphax)],
        [0, math.sin(alphax), math.cos(alphax)]
    ])
    ry = np.array([
        [math.cos(alphay), 0, math.sin(alphay)],
        [0, 1, 0],
        [-math.sin(alphay), 0, math.cos(alphay)]
    ])
    rz = np.array([
        [math.cos(alphaz), -math.sin(alphaz), 0],
        [math.sin(alphaz), math.cos(alphaz), 0],
        [0, 0, 1]
    ])
    rotmat = np.transpose(rx@ry@rz, axes=(1, 0))
    return points @ rotmat