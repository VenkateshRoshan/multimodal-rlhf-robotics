import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pybullet as p

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.env import env_setup
from environment.load_env import load_cube
from models.model import PPOModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy.")
    parser.add_argument("--ckpt", type=str, default="ppo_policy.pth", help="Path to saved model weights")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=60, help="Max steps per episode")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--delta-scale", type=float, default=0.05, help="Step size (meters) applied to xyz deltas")
    parser.add_argument("--success-thresh", type=float, default=0.03, help="Distance (m) threshold for success")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 Testing on {device} with checkpoint: {args.ckpt}")

    # Load env + model
    env, camera, panda = env_setup(use_gui=args.gui)
    model = PPOModel(image_size=(128, 128, 3)).to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    successes = 0
    episode_dists = []

    for ep in range(args.episodes):
        panda.reset_arm()
        cube_id = load_cube(position=[0.5, 0.0, np.random.uniform(0.25, 0.75)])

        final_dist = None
        for step in range(args.max_steps):
            rgb = camera.get_rgb_image()
            obs_t = torch.tensor(rgb, dtype=torch.float32, device=device).unsqueeze(0) / 255.0

            with torch.no_grad():
                xyz_dist, grip_dist, _ = model(obs_t)
                xyz_action = xyz_dist.mean  # deterministic action
                grip_prob = torch.sigmoid(grip_dist.logits)

            a_xyz = xyz_action[0].cpu().numpy()
            a_grip = float(grip_prob.item())

            ee_pos, _ = panda.get_end_effector_pose()
            target = [
                ee_pos[0] + np.clip(a_xyz[0], -1, 1) * args.delta_scale,
                ee_pos[1] + np.clip(a_xyz[1], -1, 1) * args.delta_scale,
                ee_pos[2] + np.clip(a_xyz[2], -1, 1) * args.delta_scale,
            ]
            panda.move_ee(target)

            if a_grip > 0.5:
                panda.close_gripper()
            else:
                panda.open_gripper()

            panda.step_sim(12)

            ee_pos, _ = panda.get_end_effector_pose()
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            dist = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
            final_dist = dist

            if dist < args.success_thresh:
                successes += 1
                print(f"[Episode {ep+1}] Success at step {step+1} (dist={dist:.3f} m)")
                break

        episode_dists.append(final_dist if final_dist is not None else float("inf"))
        p.removeBody(cube_id)
        if args.gui:
            print(f"[Episode {ep+1}] Final dist: {final_dist:.3f} m")

    avg_dist = float(np.mean(episode_dists))
    print(f"✅ Eval done. Successes: {successes}/{args.episodes}, avg final dist: {avg_dist:.3f} m")


if __name__ == "__main__":
    main()


# python training/test.py --ckpt ppo_policy.pth --episodes 5 --gui
