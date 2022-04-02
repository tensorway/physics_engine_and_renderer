import cv2
from matplotlib.pyplot import spring
import tqdm
import math
import numpy as np
from linalg import get_center_of_mass, get_line_equation, get_perpendicular_line_equation3d, get_plane_equation, point2line_vector, rotate3d, signed_plane_point_distance, normalize
from utils import load_points_and_faces_and_springs, imglist2gif, get_cube, load_points_and_faces, colorize_faces, gen_random_face_colors, get_springy_cube, springify_body
from render import Renderer




    

def create_simple_staged_world():
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
    face_colors = gen_random_face_colors(static_faces, groups=2*6)
    return static_points, static_faces, face_colors

def ball_falling_over_static_stairs():
    renderer = Renderer(
        light_vector = np.array([-25, -25, 30]),
        light_point = np.array([250, 0, -90]),
        cam=np.array([100, 250, -300]),
        image_size=(500, 500, 3)
    )

    static_points, static_faces, face_colors = create_simple_staged_world()
    img_static, zbuff_static = renderer.render_points(static_points, static_faces, face_colors)

    points, faces, springs = load_points_and_faces_and_springs('models/ico.obj')
    points = points.astype('float32')
    points *= 150
    springs = springify_body(points, offset=0)
    points = points + np.array([[800, 500, 2000]])
    face_colors = colorize_faces(faces, color=(255, 0, 0))

    engine = Engine(
        static_points,
        static_faces,
        points,
        faces,
        springs,
        springk = 4,
        spring_damp=4000,
        constant_acceleration=np.array([[0, -3.9, 0]]),

    )
    frames = []
    for i in tqdm.tqdm(range(1000)):
        img = img_static + 0
        zbuff = zbuff_static + 0

        img, _ = renderer.render_points(
            pointsxyz=points, 
            faces=faces,
            colors=face_colors,
            img=img, 
            zbuff=zbuff
            )
        points = engine.forward()

        cv2.imshow('main', img.astype('uint8'))
        frames.append(img.astype('uint8'))
        if cv2.waitKey(5) == ord('q'):
            break

    cv2.destroyAllWindows()
    return frames

def two_cubes_colliding():
    renderer = Renderer(
        light_vector = np.array([-25, -25, 30]),
        light_point = np.array([250, 0, -90]),
        cam=np.array([100, 250, -300]),
        image_size=(500, 500, 3)
    )

    cube_dist = 100
    points0, faces0 = get_cube(xwidth=cube_dist, ywidth=cube_dist, zwidth=cube_dist, xcenter=400, ycenter=-300, zcenter=300)
    points1, faces1 = get_cube(xwidth=cube_dist, ywidth=cube_dist, zwidth=cube_dist, xcenter=0, ycenter=-250, zcenter=300)
    faces1 += len(points0)
    springs0 = springify_body(points0, offset=0)
    springs1 = springify_body(points1, offset=len(points0))
    face_colors0 = colorize_faces(faces0, color=(255, 0, 0))
    face_colors1 = colorize_faces(faces1, color=(0, 0, 255))
    points = np.concatenate((points0, points1), axis=0)
    points = points.astype('float32')
    faces = np.concatenate((faces0, faces1), axis=0)
    springs = np.concatenate((springs0, springs1), axis=0)
    face_colors = np.concatenate((face_colors0, face_colors1), axis=0)
    a0 = np.zeros_like(points0) + np.array([[-0.3, 0, 0]])
    a1 = np.zeros_like(points1) + np.array([[0.3, 0, 0]])
    const_accel = np.concatenate((a0, a1), axis=0)

    engine = Engine(
        static_points=[],
        static_faces=[],
        movable_points=points,
        movable_faces=faces,
        movable_springs=springs,
        referent_spring_length=cube_dist,
        constant_acceleration=const_accel,
    )

    frames = []

    for i in tqdm.tqdm(range(1000)):
        points = engine.forward()
        print(engine.acceleration)
        img, _ = renderer.render_points(
            pointsxyz=points, 
            faces=faces,
            colors=face_colors,
            )

        cv2.imshow('main', img.astype('uint8'))
        frames.append(img.astype('uint8'))
        if cv2.waitKey(5) == ord('q'):
            break

    cv2.destroyAllWindows()
class Engine:
    def __init__(
        self, 
        static_points, 
        static_faces,
        movable_points,
        movable_faces,
        movable_springs,
        initial_velocity=None,
        constant_acceleration=np.array([[0, -0.7, 0]]),
        
        bouncyness = 0.9,
        springk = 32,
        spring_damp = 4000,
        point_mass = 1,
        ) -> None:

        self.static_points = static_points
        self.static_faces = static_faces

        self.movable_points = movable_points
        self.movable_faces = movable_faces
        self.movable_springs = movable_springs
        if initial_velocity:
            self.velocity = initial_velocity
        else:
            self.velocity = np.zeros_like(movable_points)
        self.acceleration = np.zeros_like(movable_points)
        self.constant_acceleration = constant_acceleration

        self.bouncyness = bouncyness
        self.springk = springk
        self.spring_damp = spring_damp
        self.point_mass = np.zeros((len(movable_points), 0)) + point_mass

    def forward(self, dt=1):

        # velocity step and accel step
        self.movable_points += self.velocity*dt
        self.velocity += self.acceleration*dt
        self.acceleration = np.zeros_like(self.movable_points) + self.constant_acceleration

        # force step
        for i in range(len(self.movable_points)):
            for j, referent_spring_length in self.movable_springs[i]:
                j = int(j)
                dist = math.sqrt(((self.movable_points[i] - self.movable_points[j])**2).sum())
                dx = (dist-referent_spring_length)
                dv_future = (self.velocity[j]-self.velocity[i]) # - 2*(self.movable_points[j]-self.movable_points[i])*dx**1*springk )
                force = (self.movable_points[j]-self.movable_points[i])*dx**1*self.springk + dv_future*self.spring_damp
                force = np.clip(force, -1, 1)
                self.acceleration[i] += force / self.point_mass[i]

        # collider static_face/movable_point step 
        for i in range(len(self.movable_points)): 
            for face_point_idxs in self.static_faces:
                face3d    = self.static_points[face_point_idxs]
                collided, normalized_signed_dist, point_on_plane = self.plane_point_collider_predictive(face3d, self.movable_points[i], self.velocity[i], np.zeros_like(face3d),dt)
                if collided:
                    indir_velocity = np.dot(self.velocity[i], normalized_signed_dist)* normalized_signed_dist
                    self.velocity[i] -= (1+self.bouncyness) * indir_velocity
                    self.movable_points[i] = point_on_plane

        # collider movable_point/movable_face step 
        for i in range(len(self.movable_points)): 
            for face_point_idxs in self.movable_faces:
                face3d    = self.movable_points[face_point_idxs]
                velocities_face3d = self.velocity[face_point_idxs]
                collided, normalized_signed_dist, point_on_plane = self.plane_point_collider_predictive(face3d, self.movable_points[i], self.velocity[i], velocities_face3d, dt)
                if collided:
                    indir_velocity = np.dot(self.velocity[i], normalized_signed_dist)* normalized_signed_dist

                    # calculating rotation
                    center_of_mass = get_center_of_mass(face3d)
                    line_direction_vector, _ = get_perpendicular_line_equation3d(center_of_mass, point_on_plane, normalized_signed_dist)
                    d0 = point2line_vector(face3d[0], line_direction_vector, center_of_mass)
                    d1 = point2line_vector(face3d[1], line_direction_vector, center_of_mass)
                    d2 = point2line_vector(face3d[2], line_direction_vector, center_of_mass)
                    ds = [d0, d1, d2]

                    # law of conservation of angular momentum
                    # i have excluded vectors for simpliticy :
                    # angular_velocity*(|d0| + |d1| + |d2|) + velocity_point_after*|dpoint| = velocity_point_before*|dpoint|
                    angular_velocity_vector = normalize(np.cross(ds[0], line_direction_vector))
                    mul = 1 if np.dot(angular_velocity_vector, indir_velocity)>0 else -1
                    angular_momentum = 0
                    masses = self.point_mass[face_point_idxs]
                    for distance, mass in zip(ds, masses):
                        angular_momentum += distance*mass
                    angular_velocity_norm = mul * (2*self.bouncyness)*np.linalg.norm(indir_velocity)/angular_momentum
                    for (point_idx, distance_vec) in enumerate(zip(ds, face_point_idxs)):
                        self.velocity[point_idx] += angular_velocity_norm*np.norm(distance_vec)*angular_velocity_vector






                    self.velocity[i] -= (1+self.bouncyness) * indir_velocity
                    self.movable_points[i] = point_on_plane


        
        # accel calc
        return self.movable_points + 0 

    def plane_point_collider_predictive(self, face3d, point, velocity, face3d_velocities, dt):
        normal, d = get_plane_equation(face3d)
        signed_dist_before = signed_plane_point_distance(normal, d, point)

        pa = point_after = point + velocity*dt
        pb = point_before = point
        face3d_after = face3d + face3d_velocities*dt
        normal_after, d_after = get_plane_equation(face3d_after)
        signed_dist_after = signed_plane_point_distance(normal_after, d_after, point_after)

        if np.sign(signed_dist_after) != np.sign(signed_dist_before):
            alpha = (d-np.dot(pa, normal)) / np.dot(pb-pa, normal)
            pop = point_on_plane = alpha*point_before + (1-alpha)*point_after

            a = face3d[0] - point_on_plane
            b = face3d[1] - point_on_plane
            c = face3d[2] - point_on_plane

            def angle(vector_1, vector_2):
                unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                return np.arccos(dot_product)

            angle_sum = abs(angle(a, b) + angle(b, c)+ angle(c, a))
            if 2*math.pi*0.999 <= angle_sum <= 2*math.pi*1.001:
                return True, normalize(signed_dist_before*normal), point_on_plane

        return False, normalize(signed_dist_before*normal), None


if '__main__' == __name__:
    # frames = two_cubes_colliding()
    ball_falling_over_static_stairs()
    # imglist2gif(frames, "ico_.gif")