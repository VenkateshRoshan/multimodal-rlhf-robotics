import torch
import numpy as np
import pybullet as p
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path().parent.parent))

from environment.load_env import load_environment, load_cube
from environment.camera import Camera
from environment.panda_controller import PandaController

import torch.nn as nn
from torchvision.models import vit_b_16

class RGBDViTGraspModel(nn.Module):
    def __init__(self, out_dim=7):
        """
        out_dim example:
        3  : EE position (x, y, z)
        4  : EE orientation (quaternion)
        """
        super().__init__()

        # ------------------
        # RGB encoder (ViT)
        # ------------------
        self.rgb_vit = vit_b_16(weights="IMAGENET1K_V1")
        self.rgb_vit.heads = nn.Identity()
        rgb_feat_dim = 768

        # ------------------
        # Depth encoder (CNN)
        # ------------------
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        depth_feat_dim = 64

        # ------------------
        # Fusion + policy head
        # ------------------
        self.policy_head = nn.Sequential(
            nn.Linear(rgb_feat_dim + depth_feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, rgb, depth):
        """
        rgb   : (B, 3, 224, 224)
        depth : (B, 1, 224, 224)
        """

        rgb_feat = self.rgb_vit(rgb)                  # (B, 768)
        depth_feat = self.depth_encoder(depth)        # (B, 64, 1, 1)
        depth_feat = depth_feat.view(depth_feat.size(0), -1)

        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        action = self.policy_head(fused)

        return action


# -----------------------------
# Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load environment
env = load_environment(use_gui=True)
robot_id = env["robot_id"]

panda = PandaController(robot_id)
camera = Camera()

# Load trained model
model = RGBDViTGraspModel(out_dim=7).to(device)
model.load_state_dict(torch.load("notebooks/rgbd_vit_grasp_model.pth", map_location=device))
model.eval()

# -----------------------------
# Reset robot (slightly randomized already)
# -----------------------------
panda.reset_arm()
panda.step_sim(50)

# -----------------------------
# Spawn cube at random location
# -----------------------------
cube_id = load_cube()   # already randomized in your script
for _ in range(30):
    p.stepSimulation()

cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
print("Cube position:", cube_pos)

# -----------------------------
# Capture RGB + Depth
# -----------------------------
rgb, depth = camera.get_rgbd()

# Preprocess RGB
rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
rgb_tensor = torch.nn.functional.interpolate(
    rgb_tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
).to(device)

# Preprocess Depth
depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
depth_tensor = torch.nn.functional.interpolate(
    depth_tensor, size=(224, 224), mode="bilinear", align_corners=False
).to(device)

# -----------------------------
# Predict end-effector pose
# -----------------------------
with torch.no_grad():
    pred = model(rgb_tensor, depth_tensor)[0].cpu().numpy()

pred_pos = pred[:3]
pred_orn = pred[3:]

print("Predicted EE position:", pred_pos)
print("Predicted EE orientation:", pred_orn)

# -----------------------------
# Move robot to predicted pose
# -----------------------------
panda.move_ee(pred_pos, pred_orn)
panda.step_sim(150)

# -----------------------------
# Close gripper (attempt grasp)
# -----------------------------
panda.close_gripper()
panda.step_sim(120)

# -----------------------------
# Lift slightly (optional)
# -----------------------------
lift_pos = pred_pos.copy()
lift_pos[2] += 0.15
panda.move_ee(lift_pos)
panda.step_sim(150)

time.sleep(5)

print("[DONE] Pick attempt completed")

