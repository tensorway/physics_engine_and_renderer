import numpy as np
import random
from tqdm import tqdm

def nonzero(x, eps=1e-8):
    return eps if x==0 else x

def get_cube(xwidth=100, ywidth=100, zwidth=100, xcenter=0, ycenter=0, zcenter=0):
    xwidth, ywidth, zwidth = xwidth/2, ywidth/2, zwidth/2
    points = np.array([
        [-xwidth, -ywidth, -zwidth],
        [xwidth, -ywidth, -zwidth],
        [xwidth, ywidth, -zwidth],
        [-xwidth, ywidth, -zwidth],

        [-xwidth, -ywidth, zwidth],
        [xwidth, -ywidth, zwidth],
        [xwidth, ywidth, zwidth],
        [-xwidth, ywidth, zwidth],
    ])
    points = points + np.array([xcenter, ycenter, zcenter])

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

def get_springy_cube(xwidth=100, ywidth=100, zwidth=100, xcenter=0, ycenter=0, zcenter=0):
    springs = np.array([
        [1, 3, 4],
        [0, 2, 5],
        [1, 3, 6],
        [0, 2, 7],

        [0, 5, 7],
        [1, 4, 6],
        [5, 2, 7],
        [3, 4, 6]
    ])
    points, faces = get_cube(xwidth, ywidth, zwidth, xcenter, ycenter, zcenter)
    return points, faces, springs

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
    return np.array(points), np.array(faces)

def springify_triangle_faces(points, faces):
    springs = [ [] for i in range(len(points))]
    for face in faces:
        for i in face:
            for j in face:
                if i!=j:
                    springs[i].append(j)
    return np.array(springs)


def load_points_and_faces_and_springs(fname, everyith=1):
    points, faces = load_points_and_faces(fname, everyith)
    springs = springify_triangle_faces(points, faces)
    return points, faces, springs



def gen_random_face_colors(faces, groups=2):
    face_colors = []
    for _ in range(len(faces)//groups):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(groups):
            face_colors.append(color)
    return np.array(face_colors)

def colorize_faces(faces, color=(150, 150, 150)):
    face_colors = []
    for _ in range(len(faces)):
        face_colors.append(color)
    return np.array(face_colors)

def imglist2gif(frames, file_name):
    import imageio
    print("Saving GIF file")
    with imageio.get_writer(file_name, mode="I") as writer:
        for idx, frame in tqdm(enumerate(frames)):
            writer.append_data(frame)