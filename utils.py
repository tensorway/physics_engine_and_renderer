import numpy as np
import random

def nonzero(x, eps=1e-8):
    return eps if x==0 else x

def get_cube(iwidth=100, jwidth=100, zwidth=100):
    points = np.array([
        [-iwidth, -jwidth, -zwidth],
        [iwidth, -jwidth, -zwidth],
        [iwidth, jwidth, -zwidth],
        [-iwidth, jwidth, -zwidth],

        [-iwidth, -jwidth, zwidth],
        [iwidth, -jwidth, zwidth],
        [iwidth, jwidth, zwidth],
        [-iwidth, jwidth, zwidth],
    ])

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],

        [4, 5, 6],
        [4, 6, 7],

        [0, 3, 7],
        [0, 7, 4],

        [1, 2, 6],
        [1, 6, 5],

        [0, 1, 5],
        [0, 5, 4],

        [3, 2, 6],
        [3, 6, 7],
    ])

    return points, faces

def load_points_and_faces(fname, everyith = 1):
    '''
    loads points and faces from an obj file

    fname : obj file name
    everyith : load every ith point

    returns:
        points and faces (np arrays)
    '''
    points, faces = [], []
    with open(fname, 'r') as fin:
        for i, line in enumerate(fin):
            try: 
                if line.startswith('v ') and i%everyith==0:
                    line = line[2:].strip()
                    x, y, z = line.split()
                    points.append([float(x), float(y), float(z)])
                elif line.startswith('f '):
                    line = line[2:].strip()
                    a, b, c = line.split()
                    a, b, c = a.split('//')[0], b.split('//')[0], c.split('//')[0]
                    a, b, c = int(a)-1, int(b)-1, int(c)-1
                    faces.append([a, b, c])
            except:
                pass
    return points, faces

def gen_random_face_colors(faces, groups=2):
    face_colors = []
    for _ in range(len(faces)//groups):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = (150, 150, 150)
        for _ in range(groups):
            face_colors.append(color)
    return np.array(face_colors)

def colorize_faces(faces, color=(150, 150, 150)):
    face_colors = []
    for _ in range(len(faces)):
        face_colors.append(color)
    return np.array(face_colors)