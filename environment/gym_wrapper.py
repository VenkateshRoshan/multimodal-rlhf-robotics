import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from environment.env import env_setup
from environment.load_env import load_cube


class RobotGraspEnv(gym.Env):
    """Gym wrapper for PyBullet robot grasping"""
    
    def __init__(self, render=False):
        super(RobotGraspEnv, self).__init__()
        
        # Initialize PyBullet environment
        self.env, self.camera, self.panda = env_setup(use_gui=render)
        self.cube_id = None
        
        # Define action space: [delta_x, delta_y, delta_z, grip]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        })
        
        self.max_steps = 50
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation"""
        super().reset(seed=seed)
        
        self.panda.reset_arm()
        
        # Remove old cube if exists
        if self.cube_id is not None:
            try:
                p.removeBody(self.cube_id)
            except:
                pass
        
        # Spawn new cube
        self.cube_id = load_cube(position=[
            0.5 + np.random.uniform(-0.15, 0.15),
            0.0 + np.random.uniform(-0.15, 0.15),
            0.65
        ])
        
        self.current_step = 0
        
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Execute action and return (obs, reward, done, truncated, info)"""
        self.current_step += 1
        
        # Parse action
        delta_xyz = action[:3]
        grip = action[3]
        
        # Get current EE position
        ee_pos, _ = self.panda.get_end_effector_pose()
        ee_pos = np.array(ee_pos)
        
        # Compute target position (12cm max movement)
        target = ee_pos + delta_xyz * 0.12
        
        # Clamp to workspace
        target[0] = np.clip(target[0], 0.2, 0.8)
        target[1] = np.clip(target[1], -0.4, 0.4)
        target[2] = np.clip(target[2], 0.3, 1.0)
        
        # Move robot
        self.panda.move_ee(target)
        
        # Control gripper
        if grip > 0.5:
            self.panda.close_gripper()
        else:
            self.panda.open_gripper()
        
        # Step simulation
        self.panda.step_sim(10)
        
        # Compute reward
        new_ee_pos, _ = self.panda.get_end_effector_pose()
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        
        new_ee_pos = np.array(new_ee_pos)
        cube_pos = np.array(cube_pos)
        dist = np.linalg.norm(new_ee_pos - cube_pos)
        
        # Dense reward
        reward = -dist * 3.0
        
        # Proximity bonuses
        if dist < 0.20:
            reward += 3.0
        if dist < 0.15:
            reward += 8.0
        if dist < 0.10:
            reward += 15.0
        if dist < 0.07:
            reward += 25.0
        if dist < 0.05:
            reward += 40.0
        
        # Success bonus
        terminated = False
        if grip > 0.5 and dist < 0.05:
            reward += 100.0
            terminated = True
        
        # Time penalty
        reward -= 0.1
        
        # Max steps reached
        truncated = self.current_step >= self.max_steps
        
        obs = self._get_obs()
        info = {'distance': dist, 'success': (grip > 0.5 and dist < 0.05)}
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Get current observation"""
        rgb = self.camera.get_rgb_image()
        ee_pos, _ = self.panda.get_end_effector_pose()
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        
        ee_pos = np.array(ee_pos, dtype=np.float32)
        cube_pos = np.array(cube_pos, dtype=np.float32)
        delta = cube_pos - ee_pos
        
        state = np.concatenate([ee_pos, cube_pos, delta])
        
        return {
            'image': rgb,
            'state': state
        }
    
    def close(self):
        """Clean up"""
        p.disconnect()