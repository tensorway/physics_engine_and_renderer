import numpy as np
import cv2
from linalg import get_line_equation, get_plane_equation, rotate3d, signed_plane_point_distance
from utils import nonzero, get_cube, load_points_and_faces, colorize_faces



class Renderer:
    '''
    simple class that renders points to screen
    '''

    def __init__(
        self,
        cam=np.array([250, 250, -300]),
        zplane=0,
        image_size=(500, 500, 3),
        light_vector = np.array([1, 1, 1]),
        light_point = np.array([0, 0, 0]),
        ):

        self.cam = cam
        self.zplane = zplane
        self.image_size = image_size
        self.light_vector = light_vector
        self.light_point  = light_point

    def get_projections(self, pointsijz):
        '''
        projests points in ijz coordinate system on zplane using self.cam
        '''
        zs = pointsijz[:, 2]
        _, _, zcam = self.cam
        alphas = (self.zplane-zs)/(zcam-zs)
        alphas = np.expand_dims(alphas, axis=1)
        camxy = np.expand_dims(self.cam[:2], axis=0)
        projection = camxy*alphas + pointsijz[:, :2]*(1-alphas)
        return projection

    def render_points(
        self,
        pointsxyz, 
        faces=[], 
        colors=[],
        background_color = 255,
        img=None,
        zbuff=None,
        render_points=False,
        point_radius = 10,
        ):

        if img is None:
            img = np.zeros(self.image_size) + background_color
        if zbuff is None:
            zbuff = np.zeros((img.shape[0], img.shape[1])) + float('inf')

        pointsijz = self.xyz2ijz(pointsxyz)
        proj = self.get_projections(pointsijz)
        if render_points:
            for point in proj:
                point = point.astype('int')
                img = cv2.circle(img, (point[1], point[0]), point_radius, (100, 100, 100), 2*point_radius)

        for face_point_idxs, color in zip(faces, colors):
            face_proj = proj[face_point_idxs].astype('int')
            face3d    = pointsijz[face_point_idxs]
            normal, d = get_plane_equation(face3d)
            brightness = np.dot(normal, self.light_vector)/np.linalg.norm(normal)/np.linalg.norm(self.light_vector)
            brightness = abs(brightness)
            camera_dist_sign = np.sign(signed_plane_point_distance(normal, d, self.cam))
            light_dist_sign  = np.sign(signed_plane_point_distance(normal, d, self.light_point))
            if camera_dist_sign!=light_dist_sign:
                brightness = 0
            color = np.clip((color*brightness), 0, 255).astype('uint8')
            self.zdraw_triangle(img, zbuff, face_proj, color, face3d)

        return img, zbuff

    def zdraw_triangle(self, img, zbuff, vertices, color, pointsijz):
        zplane, cam = self.zplane, self.cam
        vertices = vertices[vertices[:,0].argsort()]
        normal, d = get_plane_equation(pointsijz)
        xpos = lambda k, l, y:int((y-l)/nonzero(k))
        xposin=lambda k, l, y: max(0, min(img.shape[1]-1, xpos(k, l, y) ))

        k1, l1 = get_line_equation([vertices[0], vertices[2]])
        k2, l2 = get_line_equation([vertices[0], vertices[1]])
        if (xpos(k1, l1, vertices[0][0]+1000) > xpos(k2, l2, vertices[0][0]+1000)):
            k1, l1, k2, l2 = k2, l2, k1, l1

        for i in range(max(0, vertices[0][0]), min(vertices[1][0], img.shape[0])):
            for j in range(xposin(k1, l1, i), xposin(k2, l2, i)+1):
                alpha = (d - np.dot(cam, normal)) / nonzero((np.array([i, j, zplane]) - cam) @ normal)
                z = alpha * (zplane - cam[-1])
                if zbuff[i][j] > z:
                    zbuff[i][j] = z
                    img[i][j] = color

        k1, l1 = get_line_equation([vertices[0], vertices[2]])
        k2, l2 = get_line_equation([vertices[1], vertices[2]])
        if (xpos(k1, l1, vertices[2][0]-1000) > xpos(k2, l2, vertices[2][0]-1000)):
            k1, l1, k2, l2 = k2, l2, k1, l1

        for i in range(max(0, vertices[1][0]), min(vertices[2][0], img.shape[0])):
            for j in range(xposin(k1, l1, i), xposin(k2, l2, i)+1):
                alpha = (d - np.dot(cam, normal)) / nonzero((np.array([i, j, zplane]) - cam) @ normal)
                z = alpha * (zplane - cam[-1])
                if zbuff[i][j] > z:
                    zbuff[i][j] = z
                    img[i][j] = color 

    def change_zbuff_for_viewing(self, zbuff):
        minv = np.min(zbuff)
        zbuff = np.nan_to_num(zbuff, posinf=minv)
        maxv = np.max(zbuff)
        zbuff = ((zbuff-minv)/(maxv-minv+1e-8)*255).astype('uint8')
        return zbuff

    def xyz2ijz(self, pointsxyz):
        permutation = [1, 0, 2]
        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))
        pointsijz = pointsxyz[:, idx]  # return a rearranged copy
        pointsijz[:, 0] = -pointsijz[:, 0]
        return pointsijz



if '__main__' == __name__:

    points, faces = get_cube(iwidth=10, jwidth=1000, zwidth=1000, icenter=600, jcenter=250, zcenter=1200)
    face_colors = colorize_faces(faces)
    renderer = Renderer(
        light_vector = np.array([-25, -25, 30]),
        light_point = np.array([250, 0, -90]),
    )


    for i in range(1):
        img, zbuff = renderer.render_points(points, faces, face_colors)
        # points = rotate3d(points, 0.12, 0, 0)
        cv2.imshow('main', img.astype('uint8'))
        if cv2.waitKey(1) == ord('q'):
            break
        # points = points + 10

    cv2.waitKey(0)
    cv2.destroyAllWindows()