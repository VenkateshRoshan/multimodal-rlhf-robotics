import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np


class PPOModelFixed(nn.Module):
    """
    Multimodal Policy/Value Network
    
    Takes BOTH:
    - Image observations (vision)
    - State observations (robot + object positions)
    
    This makes learning MUCH easier and faster.
    Later you can remove state and go vision-only.
    """

    def __init__(self, image_size=(128, 128, 3), state_dim=9):
        super(PPOModelFixed, self).__init__()
        
        # ===================================
        # Vision Encoder (CNN)
        # ===================================
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(image_size[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        vision_out_size = self._get_conv_out(image_size)
        
        self.vision_fc = nn.Sequential(
            nn.Linear(vision_out_size, 256),
            nn.ReLU(),
        )

        # ===================================
        # State Encoder (MLP)
        # ===================================
        # state_dim = 9: ee_pos(3) + obj_pos(3) + delta(3)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # ===================================
        # Fusion Layer
        # ===================================
        fusion_dim = 256 + 128  # vision + state
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
        )

        # ===================================
        # Actor Heads
        # ===================================
        self.actor_xyz_mean = nn.Linear(512, 3)
        self.actor_grip_logits = nn.Linear(512, 1)
        
        # Learnable log std for continuous actions
        self.xyz_log_std = nn.Parameter(torch.zeros(3))

        # ===================================
        # Critic Head
        # ===================================
        self.critic = nn.Linear(512, 1)

    def forward(self, image, state):
        """
        Args:
            image: (B, H, W, C) - RGB image
            state: (B, state_dim) - robot state
            
        Returns:
            xyz_dist: Normal distribution for XYZ actions
            grip_dist: Bernoulli distribution for gripper
            value: Value estimate
        """
        # Vision encoding
        x_img = image.permute(0, 3, 1, 2)  # (B, C, H, W)
        vision_features = self.vision_encoder(x_img)
        vision_features = self.vision_fc(vision_features)

        # State encoding
        state_features = self.state_encoder(state)

        # Fusion
        combined = torch.cat([vision_features, state_features], dim=-1)
        features = self.fusion(combined)

        # Actor outputs
        xyz_mean = torch.tanh(self.actor_xyz_mean(features))  # [-1, 1]
        xyz_std = torch.exp(self.xyz_log_std).clamp(min=0.01, max=1.0)
        grip_logits = self.actor_grip_logits(features)

        # Critic output
        value = self.critic(features).squeeze(-1)

        # Create distributions
        xyz_dist = D.Normal(xyz_mean, xyz_std)
        grip_dist = D.Bernoulli(logits=grip_logits.squeeze(-1))

        return xyz_dist, grip_dist, value

    def _get_conv_out(self, shape):
        o = self.vision_encoder(torch.zeros(1, shape[2], shape[0], shape[1]))
        return int(np.prod(o.size()))
    
