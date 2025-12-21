import os
import cv2
import json
import numpy as np
import pybullet as p

from tqdm import tqdm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.load_env import load_environment, load_cube
from environment.camera import Camera
from environment.panda_controller import PandaController
from environment.camera import Camera

def save_sample(output_dir, idx, rgb, depth, label_dict):
    # Create subdirectories
    rgb_dir    = os.path.join(output_dir, "rgb")
    depth_dir  = os.path.join(output_dir, "depth")
    label_dir  = os.path.join(output_dir, "labels")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # -------------------
    # Save RGB image
    # -------------------
    img_path = os.path.join(rgb_dir, f"rgb_{idx}.png")
    cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # -------------------
    # Save Depth array
    # -------------------
    depth_path = os.path.join(depth_dir, f"depth_{idx}.npy")
    np.save(depth_path, depth)

    # -------------------
    # Save Label JSON
    # -------------------
    label_path = os.path.join(label_dir, f"label_{idx}.json")
    with open(label_path, "w") as f:
        json.dump(label_dict, f, indent=2)

def collect_data(
    output_dir="dataset",
    n_samples=2000,
    use_gui=False
):
    # Load environment
    env = load_environment(use_gui=use_gui)
    robot_id = env["robot_id"]

    panda = PandaController(robot_id)
    
    # FIXED WORLD CAMERA (never moves)
    camera = Camera()

    print(f"[INFO] Starting data collection for {n_samples} samples")

    # for idx in range(n_samples):
    for idx in tqdm(range(n_samples), desc="Collecting Data"):

        # ----------------------------------------------------
        # STEP 1 — RESET ROBOT TO HOME POSE (no simulation)
        # ----------------------------------------------------
        panda.reset_arm()   # instantly teleports robot
        # Do NOT call stepSimulation before taking picture

        # Disable motors so robot CANNOT move at all
        for j in panda.arm_joint_indices:
            p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

        # Save initial robot state (perfectly frozen)
        init_ee_pos, init_ee_orn = panda.get_end_effector_pose()
        init_joint_states = [
            p.getJointState(robot_id, j)[0] for j in panda.arm_joint_indices
        ]

        # ----------------------------------------------------
        # STEP 2 — Spawn cube (robot still frozen)
        # ----------------------------------------------------
        cube_id = load_cube()

        # Let ONLY the cube settle (robot motors OFF)
        for _ in range(20):
            p.stepSimulation()

        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)

        # ----------------------------------------------------
        # STEP 3 — TAKE PICTURE WHILE ROBOT IS 100% STILL
        # ----------------------------------------------------
        rgb, depth = camera.get_rgbd()

        # ----------------------------------------------------
        # STEP 4 — NOW robot may move (optional)
        # ----------------------------------------------------
        obs_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.30]
        
        # Re-enable motors before move
        for j in panda.arm_joint_indices:
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, force=200)

        panda.move_ee(obs_pos)
        panda.step_sim(80)

        # Final state capture
        final_ee_pos, final_ee_orn = panda.get_end_effector_pose()
        final_joint_states = [
            p.getJointState(robot_id, j)[0] for j in panda.arm_joint_indices
        ]

        # ----------------------------------------------------
        # STEP 5 — LABEL
        # ----------------------------------------------------
        label = {
            "cube_position": cube_pos,

            "initial": {
                "ee_position": init_ee_pos,
                "ee_orientation": init_ee_orn,
                "joint_states": init_joint_states
            },

            "final": {
                "ee_position": final_ee_pos,
                "ee_orientation": final_ee_orn,
                "joint_states": final_joint_states
            },

            "success": 1
        }

        # ----------------------------------------------------
        # STEP 6 — SAVE SAMPLE
        # ----------------------------------------------------
        save_sample(output_dir, idx, rgb, depth, label)

        # Cleanup
        p.removeBody(cube_id)

        if idx % 100 == 0:
            print(f"[INFO] Collected {idx} samples")

if __name__ == "__main__":
    collect_data(
        output_dir="dataset",
        n_samples=5000,
        use_gui=True
    )
