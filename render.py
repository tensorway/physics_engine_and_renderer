from re import X
from turtle import pos
import numpy as np
import cv2 
import math
import matplotlib.pyplot as plt
import random

cam = np.array([250, 250, -300])
zplane = -200
points = np.array([
    [-100, -100, -100],
    [100, -100, -100],
    [100, 100, -100],
    [-100, 100, -100],

    [-100, -100, 100],
    [100, -100, 100],
    [100, 100, 100],
    [-100, 100, 100],
])

faces = np.array([
    [0, 1, 2],
    # [0, 2, 3],

    # [4, 5, 6],
    # [4, 6, 7],

    # [0, 3, 7],
    # [0, 7, 4],

    # [1, 2, 6],
    # [1, 6, 5],

    # [0, 1, 5],
    # [0, 5, 4],

    [3, 2, 6],
    # [3, 6, 7],
])


def get_projections(proj, cam, zplane):
    zs = proj[:, 2]
    _, _, zcam = cam
    alphas = (zplane-zs)/(zcam-zs)
    alphas = np.expand_dims(alphas, axis=1)
    camxy = np.expand_dims(cam[:2], axis=0)
    projection = camxy*alphas + proj[:, :2]*(1-alphas)
    return projection

def render_points(proj, points3d, zplane, cam, faces=[], img_size=(500, 500, 3), colors=[]):
    print(proj)
    print(points3d)
    img = np.zeros(img_size)
    zbuff = np.zeros((img.shape[0], img.shape[1])) + float('inf')
    center = np.array([img.shape[0]//2, img.shape[1]//2])
    for point in proj:
        # point = point + center
        img = cv2.circle(img, tuple(point.astype('int')), 4, (255, 255, 255))

    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for face_point_idxs, color2 in zip(faces, colors):
        face_proj = proj[face_point_idxs].astype('int')
        face3d    = points3d[face_point_idxs]
        # draw a triangle
        # face_proj += center
        zdraw_triangle(img, zbuff, face_proj, color, face3d, zplane, cam)
    return img, zbuff

def get_line_equation(points):
    div = points[1][1] - points[0][1]
    k = (points[1][0]-points[0][0])/zero_diff(div)
    l = points[0][0] - k*points[0][1]
    return k, l

def get_plane_equation(points):
    a, b, c = points
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    d = a @ normal
    return normal, d

def zero_diff(x, eps=1e-8):
    return eps if x==0 else x

def zdraw_triangle(img, zbuff, vertices, color, points3d, zplane, cam):
    isorted_vertices = np.flip(np.sort(vertices), axis=0)
    isorted_vertices = vertices[vertices[:,0].argsort()]
    normal, d = get_plane_equation(points3d)
    xpos = lambda k, l, y:int((y-l)/zero_diff(k))
    xposin=lambda k, l, y: max(0, min(img.shape[1]-1, xpos(k, l, y) ))

    k1, l1 = get_line_equation([vertices[0], vertices[2]])
    k2, l2 = get_line_equation([vertices[0], vertices[1]])
    if (xpos(k1, l1, vertices[0][0]+100) > xpos(k2, l2, vertices[0][0]+100)):
        k1, l1, k2, l2 = k2, l2, k1, l1

    for i in range(max(0, isorted_vertices[0][0]), isorted_vertices[1][0]):
        for j in range(xposin(k1, l1, i), xposin(k2, l2, i)+1):
            alpha = d / ((np.array([i, j, zplane]) - cam) @ normal)
            z = alpha * (zplane - cam[-1])
            if zbuff[i][j] > z:
                zbuff[i][j] = z
                img[i][j] = color

    k2, l2 = get_line_equation([vertices[1], vertices[2]])
    if (xpos(k1, l1, vertices[2][0]-100) < xpos(k2, l2, vertices[2][0]-100)):
        k1, l1, k2, l2 = k2, l2, k1, l1

    for i in range(max(0, isorted_vertices[1][0]), isorted_vertices[2][0]):
        for j in range(xposin(k1, l1, i), xposin(k2, l2, i)+1):
            alpha = d / ((np.array([i, j, zplane]) - cam) @ normal)
            z = alpha * (zplane - cam[-1])
            if zbuff[i][j] > z:
                zbuff[i][j] = z
                img[i][j] = color 


def rotate(points, alphax, alphay, alphaz):
    rx = np.array([
        [1, 0, 0],
        [0, math.cos(alphax), -math.sin(alphax)],
        [0, math.sin(alphax), math.cos(alphax)]
    ])
    ry = np.array([
        [math.cos(alphay), 0, math.sin(alphay)],
        [0, 1, 0],
        [-math.sin(alphay), 0, math.cos(alphax)]
    ])
    rz = np.array([
        [math.cos(alphaz), -math.sin(alphaz), 0],
        [math.sin(alphaz), math.cos(alphaz), 0],
        [0, 0, 1]
    ])
    rotmat = np.transpose(rx@ry@rz, axes=(1, 0))
    return points @ rotmat


face_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(faces))]
print(face_colors)
for i in range(1):
    # points = rotate(points, 0, 0.002, 0.002)
    proj = get_projections(points, cam, zplane)
    img, zbuff = render_points(proj, points, zplane, cam, faces, colors=face_colors)
    # print(zbuff)

    minv = np.min(zbuff)
    print(np.max(zbuff), np.min(zbuff))
    zbuff = np.nan_to_num(zbuff, posinf=minv)
    maxv = np.max(zbuff)
    zbuff = ((zbuff-minv)/(maxv-minv+1e-8)*255).astype('uint8')
    print(minv, maxv, np.max(zbuff), np.min(zbuff))
    print(zbuff)
    print(minv, maxv, np.max(zbuff), np.min(zbuff))
    cv2.imshow('main', img)
    cv2.imshow('zbuff', zbuff)
    if cv2.waitKey(10) == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()