import torch
import numpy as np
import pybullet as p
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.env import env_setup
from environment.load_env import load_cube
from models.model import CombinedPolicy

# Training config
NUM_EPISODES = 1000
MAX_STEPS = 10
LR = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Starting training on {DEVICE}")
print(f"📊 Episodes: {NUM_EPISODES}, Steps per episode: {MAX_STEPS}")

# Initialize with GUI
env, camera, panda = env_setup(use_gui=True)
model = CombinedPolicy(device=DEVICE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop with tqdm
pbar = tqdm(range(NUM_EPISODES), desc="Training", unit="episode")

for episode in pbar:
    # Reset environment
    panda.reset_arm()
    cube_id = load_cube(position=[0.5, 0.0, 0.65])
    
    episode_rewards = []
    
    for step in tqdm(range(MAX_STEPS), desc="Episode Steps", leave=False):
        # Get observation
        rgb = camera.get_rgb_image()
        
        # Get positions
        ee_pos, _ = panda.get_end_effector_pose()
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        
        # Compute action
        with torch.no_grad():
            rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
            action = model(rgb, text="Pick the cube")
            action = action.cpu().numpy()[0]
        
        # Execute action
        target_pos = [
            ee_pos[0] + action[0] * 0.05,
            ee_pos[1] + action[1] * 0.05,
            ee_pos[2] + action[2] * 0.05
        ]
        panda.move_ee(target_pos)
        
        if action[3] > 0:
            panda.close_gripper()
        else:
            panda.open_gripper()
        
        panda.step_sim(1)  # Reduced for smoother GUI
        
        # Reward
        distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
        reward = -distance * 100
        episode_rewards.append(reward)
    
    # Compute returns
    returns = []
    G = 0
    for r in reversed(episode_rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns, device=DEVICE)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Update policy
    optimizer.zero_grad()
    panda.reset_arm()
    p.removeBody(cube_id)
    cube_id = load_cube(position=[0.5, 0.0, 0.65])
    
    for step in range(min(len(returns), MAX_STEPS)):
        rgb = camera.get_rgb_image()
        action = model(rgb, text="Pick the cube")
        loss = -(action.mean() * returns[step])
        loss.backward()
    
    optimizer.step()
    p.removeBody(cube_id)
    
    # Update progress bar
    avg_reward = np.mean(episode_rewards)
    pbar.set_postfix({
        'avg_reward': f'{avg_reward:.4f}',
        'final_dist': f'{-episode_rewards[-1]:.4f}'
    })

pbar.close()
torch.save(model.state_dict(), "ppo_policy.pth")
print("✅ Training complete! Model saved to ppo_policy.pth")


# import torch
# import numpy as np
# import pybullet as p
# from tqdm import tqdm
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))

# from environment.env import env_setup
# from environment.load_env import load_cube
# from models.model import CombinedPolicy

# # Training config
# NUM_EPISODES = 1000
# MAX_STEPS = 50
# BATCH_SIZE = 32  # Collect 32 episodes before update
# UPDATE_EPOCHS = 4  # Number of update epochs per batch
# LR = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# print(f"🚀 Starting training on {DEVICE}")
# print(f"📊 Episodes: {NUM_EPISODES}, Steps: {MAX_STEPS}, Batch: {BATCH_SIZE}")

# # Initialize
# env, camera, panda = env_setup(use_gui=False)
# model = CombinedPolicy(device=DEVICE).to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# # Experience buffer
# class Buffer:
#     def __init__(self):
#         self.images = []
#         self.actions = []
#         self.rewards = []
#         self.returns = []
    
#     def add(self, img, action, reward):
#         self.images.append(img)
#         self.actions.append(action)
#         self.rewards.append(reward)
    
#     def compute_returns(self, gamma=0.99):
#         self.returns = []
#         G = 0
#         for r in reversed(self.rewards):
#             G = r + gamma * G
#             self.returns.insert(0, G)
#         return self.returns
    
#     def get_batch(self):
#         return (
#             torch.stack(self.images).to(DEVICE),
#             torch.tensor(self.actions, dtype=torch.float32).to(DEVICE),
#             torch.tensor(self.returns, dtype=torch.float32).to(DEVICE)
#         )
    
#     def clear(self):
#         self.images.clear()
#         self.actions.clear()
#         self.rewards.clear()
#         self.returns.clear()

# buffer = Buffer()
# pbar = tqdm(range(NUM_EPISODES), desc="Training", unit="episode")

# for episode in pbar:
#     panda.reset_arm()
#     cube_id = load_cube(position=[0.5, 0.0, 0.65])
    
#     for step in range(MAX_STEPS):
#         rgb = camera.get_rgb_image()
#         rgb_tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        
#         ee_pos, _ = panda.get_end_effector_pose()
#         cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        
#         # Get action
#         with torch.no_grad():
#             action = model(rgb_tensor.to(DEVICE), text="Pick the cube")
#             action_np = action.cpu().numpy()[0]
        
#         # Execute
#         target_pos = [
#             ee_pos[0] + action_np[0] * 0.05,
#             ee_pos[1] + action_np[1] * 0.05,
#             ee_pos[2] + action_np[2] * 0.05
#         ]
#         panda.move_ee(target_pos)
#         panda.close_gripper() if action_np[3] > 0 else panda.open_gripper()
#         panda.step_sim(1)
        
#         # Reward
#         distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
#         reward = -distance * 100
        
#         # Store in buffer
#         buffer.add(rgb_tensor.squeeze(0), action_np, reward)
    
#     p.removeBody(cube_id)
    
#     # Update every BATCH_SIZE episodes
#     if (episode + 1) % BATCH_SIZE == 0:
#         buffer.compute_returns()
#         images, actions, returns = buffer.get_batch()
        
#         # Normalize returns
#         returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
#         # Multiple update epochs
#         for epoch in range(UPDATE_EPOCHS):
#             optimizer.zero_grad()
            
#             # Forward pass on entire batch
#             pred_actions = []
#             for i in range(len(images)):
#                 pred_action = model(images[i:i+1], text="Pick the cube")
#                 pred_actions.append(pred_action)
            
#             pred_actions = torch.cat(pred_actions, dim=0)
            
#             # Policy loss
#             loss = -(pred_actions.mean(dim=1) * returns).mean()
#             loss.backward()
#             optimizer.step()
        
#         buffer.clear()
        
#         avg_return = returns.mean().item()
#         pbar.set_postfix({'avg_return': f'{avg_return:.2f}', 'loss': f'{loss.item():.4f}'})

# pbar.close()
# torch.save(model.state_dict(), "ppo_policy.pth")
# print("✅ Training complete!")