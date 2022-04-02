import numpy as np
import math
from utils import nonzero

def get_line_equation(points):
    div = points[1][1] - points[0][1]
    k = (points[1][0]-points[0][0])/nonzero(div)
    l = points[0][0] - k*points[0][1]
    return k, l

def get_perpendicular_line_equation3d(point_on_resulting, point_not_on_resulting, normal):
    '''
    takes a plane and two points in it and
    creates a line that is normal to the line 
    connecting the two points and is in the plane
    '''
    ab = point_on_resulting-point_not_on_resulting
    line_direction_vector = np.cross(ab, normal)
    return line_direction_vector, point_on_resulting

def point2line_vector(point, line_direction_vector, point_on_line):
    ab = point_on_line-point
    return np.dot(ab, line_direction_vector)

def get_plane_equation(points):
    a, b, c = points
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    d = a @ normal
    return normal, d

def get_center_of_mass(points):
    return np.mean(points, axis=0)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def signed_plane_point_distance(normal, d, point):
    return (np.dot(normal, point)-d)/np.linalg.norm(normal)

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