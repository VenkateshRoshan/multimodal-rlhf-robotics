import pybullet as p
import numpy as np


class Camera:
    def __init__(self,
                 width=128,
                 height=128,
                 fov=60,
                 near=0.01,
                 far=2.0,
                 camera_pos=[1.2, 0, 1.0],
                 target_pos=[0.5, 0, 0.6]):
        """
        Simple PyBullet camera wrapper.

        Args:
            width (int): image width
            height (int): image height
            fov (float): field of view
            near (float): near clipping plane
            far (float): far clipping plane
            camera_pos (list): [x,y,z] camera location
            target_pos (list): [x,y,z] look-at point
        """

        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.camera_pos = camera_pos
        self.target_pos = target_pos

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
        )

        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(self.width) / float(self.height),
            nearVal=self.near,
            farVal=self.far
        )

    def get_rgb_image(self):
        """
        Returns:
            rgb: (H, W, 3) uint8 array
        """
        img = p.getCameraImage(
            self.width,
            self.height,
            self.view_matrix,
            self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb = np.reshape(img[2], (self.height, self.width, 4))[:, :, :3]
        return rgb

    def get_rgbd(self):
        """
        Returns:
            rgb:  (H, W, 3) uint8
            depth: (H, W) float32
        """
        img = p.getCameraImage(
            self.width,
            self.height,
            self.view_matrix,
            self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb = np.reshape(img[2], (self.height, self.width, 3))

        depth_buffer = np.reshape(img[3], (self.height, self.width))
        
        # Convert depth buffer to real depth (meters)
        depth = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer)
        
        return rgb, depth

camera = Camera()
def get_camera_rgb():
    return camera.get_rgb_image()