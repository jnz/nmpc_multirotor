"""
Visualization for Vehicle and Terrain
(OpenGL)

(c) Jan Zwiener (jan@zwiener.org)
"""

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from stl import mesh
from geodetic_toolbox import quat_to_matrix, extract_rpy_from_R_b_to_n
import os  # used to set global env. variable for SDL's VSYNC

# This class is used to render a STL model using OpenGL and Pygame.
# Additionally this class renders a random terrain with the
# help of perlin noise.


class RenderStlPygame:
    def __init__(self, stlpath, logopath):
        self.mesh_path = stlpath
        self.logo_path = logopath
        self.rNED_to_OpenGL = np.identity(4)
        self.rNED_to_OpenGL[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    def init(self, window_title):
        # disable vsync (not guaranteed to work, but it might)
        os.environ["SDL_VSYNC"] = "0"

        # Use double buffering and antialiasing
        pygame.display.gl_set_attribute(GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(GL_MULTISAMPLESAMPLES, 4)
        display_mode = DOUBLEBUF | OPENGL
        width = 1024
        height = 768
        pygame.display.set_mode((width, height), display_mode, vsync=0)
        pygame.display.set_caption(window_title)
        glClearColor(0.392, 0.584, 0.929, 1.0)  # cornflower blue
        glCullFace(GL_FRONT)
        glEnable(GL_CULL_FACE)

        # <draw logo>
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, width, 0, height)
        glMatrixMode(GL_MODELVIEW)
        loading_screen_image = pygame.image.load(self.logo_path)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glRasterPos2i(0, height)
        glPixelZoom(1, -1)
        glDrawPixels(
            loading_screen_image.get_width(),
            loading_screen_image.get_height(),
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            pygame.image.tostring(loading_screen_image, "RGBA"),
        )

        pygame.display.flip()
        pygame.time.delay(500)  # Add a brief delay to control the loading screen duration
        # </draw logo>

        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (width / height), 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        self.model = mesh.Mesh.from_file(self.mesh_path)
        self.model_display_list = -1
        self.terrain_display_list = -1

    def render_terrain(self):
        glPushMatrix()
        glTranslatef(0.0, -25.0, 0.0)

        # render into a display list on first call to speed up drawing
        if self.terrain_display_list == -1:
            self.terrain_display_list = glGenLists(1)
            glNewList(self.terrain_display_list, GL_COMPILE)
            # <fill_list>
            X, Y, Z, N = perlin_tile(-250.0, 250.0, -250.0, 250.0, 2.5, 0.6)
            glColor3f(1.0, 1.0, 1.0)
            for i in range(2, X.shape[0] - 1):
                for j in range(2, X.shape[1] - 1):
                    glBegin(GL_QUADS)
                    glNormal3fv(N[i, j - 1])
                    glVertex3fv(
                        (
                            X[i, j - 1],
                            Z[i, j - 1],
                            Y[i, j - 1],
                        )
                    )
                    glNormal3fv(N[i - 1, j - 1])
                    glVertex3fv(
                        (
                            X[i - 1, j - 1],
                            Z[i - 1, j - 1],
                            Y[i - 1, j - 1],
                        )
                    )
                    glNormal3fv(N[i - 1, j])
                    glVertex3fv(
                        (
                            X[i - 1, j],
                            Z[i - 1, j],
                            Y[i - 1, j],
                        )
                    )
                    glNormal3fv(N[i, j])
                    glVertex3fv(
                        (
                            X[i, j],
                            Z[i, j],
                            Y[i, j],
                        )
                    )
                    glEnd()

            # </fill_list>
            glEndList()

        glCallList(self.terrain_display_list)

        glPopMatrix()

    def render_triangles(self, model, vehicle_config, alpha):
        if alpha == 1.0:
            glColor4f(1.0, 1.0, 1.0, alpha)
        else:
            glColor4f(0.0, 0.0, 1.0, alpha)

        glPushMatrix()
        glTranslatef(
            vehicle_config.model_offset[0],
            vehicle_config.model_offset[1],
            vehicle_config.model_offset[2],
        )
        glRotatef(180.0, 1.0, 0.0, 0.0)

        # render into a display list on first call to speed up drawing
        if self.model_display_list == -1:
            self.model_display_list = glGenLists(1)
            glNewList(self.model_display_list, GL_COMPILE)
            # <fill_list>
            glBegin(GL_TRIANGLES)
            for i in range(len(model.vectors)):
                normal = model.get_unit_normals()[i]
                glNormal3f(-normal[0], -normal[1], -normal[2])
                for vertex in model.vectors[i]:
                    glVertex3f(*vertex)
            glEnd()
            # </fill_list>
            glEndList()

        glCallList(self.model_display_list)

        glPopMatrix()

    def render_sphere(self, position_n, radius):
        # Draw object
        # -----------
        glPushMatrix()
        # Rotation Pipeline is: R_NED_to_OpenGL
        # NED to OpenGL: x=N, y=-D, z=E
        glTranslatef(position_n[0], -position_n[2], position_n[1])
        # # Rotate from NED to OpenGL
        glMultMatrixf(self.rNED_to_OpenGL.T.flatten())

        # Draw as wireframe model
        glDisable(GL_CULL_FACE)
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        gluSphere(gluNewQuadric(), radius, 20, 20)

        glPopMatrix()

    def render_object(self, R_b_to_n, position_n, vehicle_config, alpha, u):
        # Draw object
        # -----------
        glPushMatrix()
        # Rotation Pipeline is: R_NED_to_OpenGL * R_Body_to_NED * R_Model_to_Body
        glTranslatef(position_n[0], -position_n[2], position_n[1])
        # # Rotate from NED to OpenGL
        glMultMatrixf(self.rNED_to_OpenGL.T.flatten())
        rBody_to_NED_4x4 = np.identity(4)
        rBody_to_NED_4x4[:3, :3] = R_b_to_n
        glMultMatrixf(rBody_to_NED_4x4.T.flatten())

        # Draw as wireframe model
        glDisable(GL_CULL_FACE)
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self.render_triangles(self.model, vehicle_config, alpha)
        glPopMatrix()

    def render(
        self, R_b_to_n, position_n, cam_pitch, cam_yaw, cam_pos_gl, vehicle_config, predictedX, u
    ):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        light_position = [
            -0.3,
            1.0,
            -0.3,
            0.0,
        ]  # directional light coming from a slight angle, w=0 is a directional light, 1 is a positional light
        ambient_light = [0.5, 0.5, 0.5, 1.0]  # some light even in shadowed areas
        diffuse_light = [0.9, 0.9, 0.8, 1.0]  # slightly yellowish white
        specular_light = [1.0, 1.0, 0.9, 1.0]  # to give a bit of shininess
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular_light)
        glShadeModel(GL_SMOOTH)

        glEnable(GL_BLEND)  # You gotta enable blending first, bro!
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glLoadIdentity()
        glRotatef(cam_pitch, 1, 0, 0)
        glRotatef(cam_yaw, 0, 1, 0)
        glTranslatef(-cam_pos_gl[0], -cam_pos_gl[1], -cam_pos_gl[2])

        # Draw terrain with shaded polygons
        # ---------------------------------
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.render_terrain()

        # Draw as wireframe model
        glDisable(GL_CULL_FACE)
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # draw prediction as 'ghosts' with alpha blending
        if predictedX is not None:
            N_horizon = predictedX.shape[0]
            for i in range(N_horizon - 1, 0, -10):
                state_i = predictedX[i]
                pred_position_n = state_i[7:10]
                if np.linalg.norm(pred_position_n - position_n) > 5.0:
                    pred_R_b_to_n = quat_to_matrix(state_i[0:4])
                    alpha = (1.0 - i / float(N_horizon)) * 0.3 + 0.05
                    self.render_object(
                        pred_R_b_to_n, pred_position_n, vehicle_config, alpha, u=None
                    )

        # render the actual current object state
        self.render_object(R_b_to_n, position_n, vehicle_config, alpha=1.0, u=u)

        # Render obstacles:
        # glColor3f(1.0, 0.0, 0.0)
        # self.render_sphere(np.array([40.0, -10.0, 0.0]), 20.0)
        # self.render_sphere(np.array([100.0, 10.0, 0.0]), 20.0)

        pygame.display.flip()


def perlin_tile(xstart, xend, ystart, yend, stepsize, scale):
    X, Y = np.meshgrid(np.arange(xstart, xend, stepsize), np.arange(ystart, yend, stepsize))
    Z = np.zeros_like(X)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = scale * perlin_internal(X[i, j], Y[i, j])

    # calculate normals. Note: this does not calculate normals at the outer ring
    N = np.ones((X.shape[0], X.shape[1], 3))
    for i in range(1, X.shape[0] - 1):
        for j in range(1, X.shape[1] - 1):
            normals = [
                perlin_normal_from_tile(X, Y, Z, i, j),
                perlin_normal_from_tile(X, Y, Z, i + 1, j),
                perlin_normal_from_tile(X, Y, Z, i, j + 1),
                perlin_normal_from_tile(X, Y, Z, i + 1, j + 1),
            ]
            N[i, j] = np.mean(normals, axis=0)

    return X, Y, Z, N


def calculate_normal(v1, v2, v3):
    a = np.subtract(v2, v1)
    b = np.subtract(v3, v1)
    cross_product = np.cross(a, b)
    normal = cross_product / np.linalg.norm(cross_product)
    return normal


def perlin_normal_from_tile(X, Y, Z, i, j):
    vertices = [
        np.array([X[i, j], Z[i, j], Y[i, j]]),
        np.array([X[i - 1, j], Z[i - 1, j], Y[i - 1, j]]),
        np.array([X[i - 1, j - 1], Z[i - 1, j - 1], Y[i - 1, j - 1]]),
    ]
    return calculate_normal(*vertices)


def perlin_internal(x, z):
    return (
        4 * perlinLayer(x / 8, z / 8)
        + 8 * perlinLayer(x / 16, z / 16)
        + 16 * perlinLayer(x / 32, z / 32)
        + 32 * perlinLayer(x / 64, z / 64)
    )


def smootherstep(edge0, edge1, x):
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def dotGridGradient(ix, iz, x, z):
    dx = x - ix
    dz = z - iz
    gradientAngle = getPseudorandomAngle(ix, iz)
    return dx * np.cos(gradientAngle) + dz * (-np.sin(gradientAngle))


def getPseudorandomAngle(ix, iz):
    x = (np.sin(ix) + np.cos(iz)) * 10000
    return 2 * np.pi * (x - np.floor(x))


def perlinLayer(x, z):
    x0 = np.floor(x)
    x1 = x0 + 1
    z0 = np.floor(z)
    z1 = z0 + 1
    sx = smootherstep(x0, x1, x)
    sz = smootherstep(z0, z1, z)

    n0 = dotGridGradient(x0, z0, x, z)
    n1 = dotGridGradient(x1, z0, x, z)
    ix0 = lerp(n0, n1, sx)
    n0 = dotGridGradient(x0, z1, x, z)
    n1 = dotGridGradient(x1, z1, x, z)
    ix1 = lerp(n0, n1, sx)

    return lerp(ix0, ix1, sz)


def lerp(a, b, w):
    return (1.0 - w) * a + w * b
