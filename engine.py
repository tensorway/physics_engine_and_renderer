import numpy as np
import cv2
from sympy import PolynomialDivisionFailed
from linalg import get_line_equation, get_plane_equation, rotate3d, signed_plane_point_distance, normalize
from utils import nonzero, get_cube, load_points_and_faces, colorize_faces, gen_random_face_colors, get_springy_cube
from render import Renderer
import math



def plane_point_collider(face3d, point, collide_dist):
    normal_vec, d = get_plane_equation(face3d)
    signed_dist = signed_plane_point_distance(normal_vec, d, point)
    
    if abs(signed_dist) < collide_dist:
        '''
        if points of the triangle are a, b, c and point thet we are cheching is p:
            if A = [ab, ac] a matrix of vectors ab and ac and 
                B = [p] matrix out of a point it
            use least squres to find the solution to A@coeff = B
            if the coefficients that are received are both in [0, 1] and their
                sum is <= 1 then the point is in the triangle
        '''
        a1, a2, a3 = ab = face3d[1] - face3d[0]
        b1, b2, b3 = ac = face3d[2] - face3d[0]
        t1, t2, t3 = target = point - signed_dist*normalize(normal_vec) - face3d[0]

        alpha = (t1*b2-t2*b1) / nonzero(a1*b2 - a2*b1)
        beta = (t1 - alpha*a1) / nonzero(b1)

        # a = (face3d - np.expand_dims(face3d[0], axis=1))[1:].transpose()
        # b = np.expand_dims(point - face3d[0], axis=1)
        # coeffs = np.linalg.lstsq(a, b)[0]
        # alpha, beta = coeffs[0][0], coeffs[1][0]
        if 0<=alpha<= 1 and 0<=beta<=1 and alpha+beta<=1:
            return True, normalize(signed_dist*normal_vec)
    return False, normalize(signed_dist*normal_vec)
    




if '__main__' == __name__:

    # static_points, static_faces = get_cube(xwidth=1000, ywidth=10, zwidth=1000, xcenter=900, ycenter=-1000, zcenter=2000)
    # static_points = rotate3d(static_points, 0, 0, 0.15)
    static_points0, static_faces0 = get_cube(xwidth=1000, ywidth=50, zwidth=1000, xcenter=900, ycenter=-300, zcenter=2000)
    static_points0 = rotate3d(static_points0, 0, 0, 0.50)
    static_points1, static_faces1 = get_cube(xwidth=2000, ywidth=50, zwidth=1000, xcenter=100, ycenter=-100, zcenter=2000)
    static_points1 = rotate3d(static_points1, 0, 0, -0.50)
    static_faces1 += len(static_points0)
    static_points2, static_faces2 = get_cube(xwidth=1000, ywidth=50, zwidth=1000, xcenter=900, ycenter=-1500, zcenter=2000)
    static_points2 = rotate3d(static_points2, 0, 0, 0.50)
    static_faces2 += 2*len(static_points0)
    static_points3, static_faces3 = get_cube(xwidth=2000, ywidth=50, zwidth=1000, xcenter=255, ycenter=-1100, zcenter=2000)
    static_points3 = rotate3d(static_points3, 0, 0, -0.15)
    static_faces3 += 3*len(static_points0)
    static_points = np.concatenate((static_points0, static_points1, static_points2, static_points3), axis=0)
    static_faces  = np.concatenate((static_faces0, static_faces1, static_faces2, static_faces3), axis=0)

    # static_points = np.concatenate((static_points0, static_points1), axis=0)
    # static_faces  = np.concatenate((static_faces0, static_faces1), axis=0)

    # static_points, static_faces = load_points_and_faces('models/tough_easy.obj')
    # static_points = static_points*np.array([[6, 6, 10]])
    # static_points += np.array([[250, -500, 300]])
    
    # face_colors = colorize_faces(static_faces, color=(100, 100, 100))
    # static_points = np.array([[0, 1, 1]])
    face_colors = gen_random_face_colors(static_faces, groups=2*6)
    renderer = Renderer(
        light_vector = np.array([-25, -25, 30]),
        light_point = np.array([250, 0, -90]),
        cam=np.array([100, 250, -300]),
        image_size=(500, 500, 3)
        
    )

    img_static, zbuff_static = renderer.render_points(static_points, static_faces, face_colors)

    cube_dist = 100
    springk = 0.000001
    spring_damp = 2000
    point_mass = 1
    points, faces, springs = get_springy_cube(xwidth=cube_dist, ywidth=cube_dist, zwidth=cube_dist, xcenter=800, ycenter=190, zcenter=2000)
    points = points.astype('float32')
    face_colors = gen_random_face_colors(faces, groups=2*6)
    # points = np.array([
    #     [800, 0.0, 2000],
    # ])
    velocity = np.zeros_like(points)
    acceleration = np.zeros_like(points)
    dt = 0.03
    ball_radius = 2
    bouncyness = 0.8
    frames = []

    import tqdm
    for i in tqdm.tqdm(range(1000*20)):
        img = img_static + 0
        zbuff = zbuff_static + 0

        img, _ = renderer.render_points(
            pointsxyz=points, 
            faces=faces,
            colors=face_colors,
            render_points=False, 
            point_radius=ball_radius, 
            img=img, 
            zbuff=zbuff
            )

        cv2.imshow('main', img.astype('uint8'))
        frames.append(img.astype('uint8'))
        if cv2.waitKey(5) == ord('q'):
            break


        # velocity step and accel step
        # print(points)
        # print(velocity)
        # print(acceleration)
        points += velocity*dt
        velocity += acceleration*dt

        acceleration = np.zeros_like(points) + np.array([[0, -0.2, 0]])

        # collider step 
        ijz = renderer.xyz2ijz(points)
        for i in range(len(points)): 
            for face_point_idxs in static_faces:
                face3d    = static_points[face_point_idxs]
                collided, normalized_signed_dist = plane_point_collider(face3d, points[i], ball_radius*5)
                if collided:
                    # print("hit")
                    points[i] = points[i] + ball_radius*normalized_signed_dist
                    indir_velocity = np.dot(velocity[i], normalized_signed_dist)* normalized_signed_dist
                    # print(velocity[i], indir_velocity, normalized_signed_dist)
                    velocity[i] -= (1+bouncyness) * indir_velocity
                    # print(velocity[i], indir_velocity)

        # force step
        for i in range(len(points)):
            for j in springs[i]:
                dist = math.sqrt(((points[i] - points[j])**2).sum())
                dx = (dist-cube_dist)
                # print("dist=", dist, dx, "velocityji=", velocity[j], velocity[i], "spring=", (points[j]-points[i])*dx*springk, "damp=", (velocity[j]-velocity[i])*spring_damp)
                force = (points[j]-points[i])*dx**4*springk + (velocity[j]-velocity[i])*spring_damp
                force = np.clip(force, -10, 10)
                acceleration[i] += force / point_mass
                # print(force, acceleration[i], point_mass)
        
        
        # print(points)
        # accel calc

    # cv2.waitKey(0)
    cv2.destroyAllWindows()

import imageio
print("Saving GIF file")
with imageio.get_writer("smiling.gif", mode="I") as writer:
    for idx, frame in enumerate(frames):
        print("Adding frame to GIF file: ", idx + 1)
        writer.append_data(frame)