import pybullet as p
import numpy as np
from environment.panda_controller import PandaController


class CameraOnGripper:
    def __init__(
        self,
        robot_id,
        ee_link_index=11,
        width=224,
        height=224,
        fov=60,
        near=0.01,
        far=2.0,
        cam_offset=[0.05, 0.0, 0.10],        # camera 5cm forward, 10cm above gripper
        look_offset=[0.15, 0.0, -0.10]       # camera looks slightly downward
    ):
        """
        Eye-in-hand camera (camera attached to end-effector).

        Args:
            robot_id: Robot unique ID
            ee_link_index: End-effector link index (11 for Panda)
            cam_offset: camera position offset from EE in EE frame
            look_offset: camera look-at point relative to EE frame
        """

        self.robot_id = robot_id
        self.ee_link_index = ee_link_index

        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far

        self.cam_offset = np.array(cam_offset)
        self.look_offset = np.array(look_offset)

        print("[INFO] Camera attached to end-effector initialized.")


    # -----------------------------------------------------------
    # Compute camera pose based on EE pose + offsets
    # -----------------------------------------------------------
    def _compute_camera_pose(self):
        state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = np.array(state[0])
        ee_orn = state[1]

        # Rotation matrix of EE
        rot_matrix = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)

        # Transform offset into world frame
        cam_pos = ee_pos + rot_matrix @ self.cam_offset
        look_at = ee_pos + rot_matrix @ self.look_offset

        # Fixed up vector (can also compute from EE orientation)
        up_vector = rot_matrix @ np.array([0, 0, 1])

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=look_at,
            cameraUpVector=up_vector
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(self.width) / float(self.height),
            nearVal=self.near,
            farVal=self.far
        )

        return view_matrix, proj_matrix


    # -----------------------------------------------------------
    # Public API — Get RGB
    # -----------------------------------------------------------
    def get_rgb(self):
        view_matrix, proj_matrix = self._compute_camera_pose()

        img = p.getCameraImage(
            self.width,
            self.height,
            view_matrix,
            proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb = np.reshape(img[2], (self.height, self.width, 4))[:, :, :3]
        return rgb


    # -----------------------------------------------------------
    # Public API — Get RGB + Depth
    # -----------------------------------------------------------
    def get_rgbd(self):
        view_matrix, proj_matrix = self._compute_camera_pose()

        img = p.getCameraImage(
            self.width,
            self.height,
            view_matrix,
            proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb = np.reshape(img[2], (self.height, self.width, 4))[:, :, :3]
        depth_buffer = np.reshape(img[3], (self.height, self.width))

        depth = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer)
        return rgb, depth


    def get_rgbd_look_at(self, target_pos):
        """
        Get RGB + depth with camera attached to EE,
        explicitly looking at target_pos (e.g., cube position).
        """
        state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = np.array(state[0])
        ee_orn = state[1]

        rot_matrix = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)

        cam_pos = ee_pos + rot_matrix @ self.cam_offset
        up_vector = rot_matrix @ np.array([0, 0, 1])

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vector
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(self.width) / float(self.height),
            nearVal=self.near,
            farVal=self.far
        )

        img = p.getCameraImage(
            self.width,
            self.height,
            view_matrix,
            proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb = np.reshape(img[2], (self.height, self.width, 4))[:, :, :3]
        depth_buffer = np.reshape(img[3], (self.height, self.width))
        depth = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer)

        return rgb, depth
