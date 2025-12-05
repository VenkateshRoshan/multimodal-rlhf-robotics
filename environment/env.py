from load_env import load_environment
from camera import Camera
from panda_controller import PandaController

import pybullet as p
import time


def env_setup(use_gui=False):
    """Initialize PyBullet environment, Panda controller, and camera."""
    
    # Load environment: plane + table + panda
    env = load_environment(use_gui=use_gui)

    # Create Panda controller
    panda = PandaController(env["robot_id"])
    panda.reset_arm()

    # Create camera AFTER env is loaded
    camera = Camera()

    return env, camera, panda


if __name__ == "__main__":
    # Initialize everything
    env, camera, panda = env_setup(use_gui=True)

    # --- Example Movements ---
    # Move arm forward
    # panda.move_ee([0.5, 0.0, 0.1])
    # panda.step_sim(120)

    # # Move arm upward
    # panda.move_ee([0.5, 0.0, 0.9])
    # panda.step_sim(120)

    # # Open gripper
    # panda.open_gripper()
    # panda.step_sim(80)

    # # Close gripper
    # panda.close_gripper()
    # panda.step_sim(80)

    for i in range(100):
        panda.move_ee([0.5, 0.0, 0.5 + 0.1 * (i % 10)])
        panda.step_sim(500)
        panda.open_gripper()
        panda.step_sim(500)
        panda.close_gripper()
        panda.step_sim(500)

    # --- Example camera capture ---
    rgb = camera.get_rgb_image()
    print("Captured image shape:", rgb.shape)

    # Keep GUI open
    # time.sleep(30)
