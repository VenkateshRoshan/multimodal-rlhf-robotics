import pybullet as p
import numpy as np
import pybullet_data


class PandaController:
    def __init__(self, robot_id):
        self.robot_id = robot_id

        # Joint indices for Panda arm
        self.arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]

        # Gripper joint indices
        self.gripper_left = 9
        self.gripper_right = 10

        # Default gripping values
        self.gripper_open_val = 0.04
        self.gripper_close_val = 0.0

        # IK parameters
        self.max_iters = 100
        self.threshold = 1e-3

        # End-effector link of Panda
        self.ee_link_index = 11

    # ----------------------------------------------------------------------

    def reset_arm(self):
        """Reset Panda arm to a standard home pose."""
        # home_positions = [0, -0.5, 0, -2.0, 0, 1.5, 0.8]
        home_positions = [
                0 + np.random.uniform(-0.25, 0.25),
            -0.5 + np.random.uniform(-0.25, 0.25),
                0 + np.random.uniform(-0.25, 0.25),
            -2.0 + np.random.uniform(-0.25, 0.25),
                0 + np.random.uniform(-0.25, 0.25),
                1.5 + np.random.uniform(-0.25, 0.25),
                0.8 + np.random.uniform(-0.25, 0.25),
            ]

        for j, pos in zip(self.arm_joint_indices, home_positions):
            p.resetJointState(self.robot_id, j, pos)

        self.open_gripper()

    # ----------------------------------------------------------------------

    def open_gripper(self):
        p.setJointMotorControl2(self.robot_id, self.gripper_left,
                                p.POSITION_CONTROL,
                                targetPosition=self.gripper_open_val)

        p.setJointMotorControl2(self.robot_id, self.gripper_right,
                                p.POSITION_CONTROL,
                                targetPosition=self.gripper_open_val)

    def close_gripper(self):
        p.setJointMotorControl2(self.robot_id, self.gripper_left,
                                p.POSITION_CONTROL,
                                targetPosition=self.gripper_close_val)

        p.setJointMotorControl2(self.robot_id, self.gripper_right,
                                p.POSITION_CONTROL,
                                targetPosition=self.gripper_close_val)

    # ----------------------------------------------------------------------

    def get_end_effector_pose(self):
        """Return (pos, orn) of Panda's end-effector."""
        state = p.getLinkState(self.robot_id, self.ee_link_index)
        pos = state[0]
        orn = state[1]
        return pos, orn

    # ----------------------------------------------------------------------

    def move_ee(self, target_pos, target_orn=None):
        """
        Move end-effector using IK.
        
        Args:
            target_pos: [x, y, z]
            target_orn: quaternion. If None → use current orientation.
        """
        if target_orn is None:
            _, current_orn = self.get_end_effector_pose()
            target_orn = current_orn

        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            target_pos,
            target_orn,
            maxNumIterations=self.max_iters,
            residualThreshold=self.threshold
        )

        # Apply arm joint positions only
        for i, j in enumerate(self.arm_joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                joint_positions[i],
                force=200
            )

    # ----------------------------------------------------------------------

    def step_sim(self, steps=1):
        """Step physics forward."""
        for _ in range(steps):
            p.stepSimulation()
