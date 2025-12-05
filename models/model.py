# import torch
# import torch.nn as nn
# import requests
# import json
# import base64
# from PIL import Image
# import io

# # ================================================================
# # 1. OLLAMA GEMMA ENCODER
# # ================================================================
# class OllamaGemmaEncoder(nn.Module):
#     def __init__(self, model_name="gemma3:4b-it-qat", embedding_dim=512):
#         super().__init__()
#         self.model_name = model_name
#         self.embedding_dim = embedding_dim
#         self.ollama_url = "http://localhost:11434/api/generate"
        
#         # Small MLP to project Ollama output to fixed size
#         self.projection = nn.Sequential(
#             nn.Linear(4096, 1024),  # Gemma output is typically 4096
#             nn.ReLU(),
#             nn.Linear(1024, embedding_dim)
#         )

#     def forward(self, rgb_image, text=""):
#         # Convert image to base64
#         pil_img = Image.fromarray(rgb_image)
#         buffered = io.BytesIO()
#         pil_img.save(buffered, format="PNG")
#         img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
#         # Call Ollama
#         prompt = f"{text}\nDescribe this robotic scene." if text else "Describe this robotic scene."
#         payload = {
#             "model": self.model_name,
#             "prompt": prompt,
#             "images": [img_b64],
#             "stream": False
#         }
        
#         response = requests.post(self.ollama_url, json=payload)
        
#         # Get embedding (using response as feature)
#         embedding_text = response.json().get("response", "")
        
#         # Simple embedding: use response length and hash as features (placeholder)
#         # In real use, you'd extract embeddings from Ollama's embedding endpoint
#         feature = torch.randn(1, 4096)  # Placeholder - use actual embeddings
        
#         embedding = self.projection(feature)
#         return embedding


# # ================================================================
# # 2. PPO POLICY MLP  
# # ================================================================
# class PolicyMLP(nn.Module):
#     def __init__(self, embedding_dim=512, action_dim=4, hidden_dim=256):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(embedding_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim),
#             nn.Tanh()
#         )

#     def forward(self, embedding):
#         return self.net(embedding)


# # ================================================================
# # 3. COMBINED POLICY
# # ================================================================
# class CombinedPolicy(nn.Module):
#     def __init__(self, model_name="gemma3:4b-it-qat", action_dim=4, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.encoder = OllamaGemmaEncoder(model_name=model_name, embedding_dim=512)
#         self.policy = PolicyMLP(embedding_dim=512, action_dim=action_dim)

#     def forward(self, rgb_image, text=""):
#         embedding = self.encoder(rgb_image, text).to(self.device)
#         actions = self.policy(embedding)
#         return actions
    

# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image
# import torchvision.transforms as T

# # ================================================================
# # SIMPLE CNN VISION ENCODER
# # ================================================================
# class SimpleCNNEncoder(nn.Module):
#     def __init__(self, embedding_dim=512):
#         super().__init__()
#         self.embedding_dim = embedding_dim
        
#         self.conv_net = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * 13 * 13, 512),
#             nn.ReLU(),
#             nn.Linear(512, embedding_dim)
#         )

#     def forward(self, x, text=""):
#         # rgb_image: numpy (H,W,3) uint8
#         # x = torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
#         x = self.conv_net(x)
#         return x


# class PolicyMLP(nn.Module):
#     def __init__(self, embedding_dim=512, action_dim=4, hidden_dim=256):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(embedding_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim),
#             nn.Tanh()
#         )

#     def forward(self, embedding):
#         return self.net(embedding)


# class CombinedPolicy(nn.Module):
#     def __init__(self, action_dim=4, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.encoder = SimpleCNNEncoder(embedding_dim=512)
#         self.policy = PolicyMLP(embedding_dim=512, action_dim=action_dim)

#     def forward(self, rgb_image, text=""):
#         embedding = self.encoder(rgb_image, text).to(self.device)
#         actions = self.policy(embedding)
#         return actions

import torch
import torch.nn as nn
import requests
import numpy as np

# ================================================================
# OLLAMA GEMMA ENCODER
# ================================================================
class OllamaGemmaEncoder(nn.Module):
    def __init__(self, model_name="gemma3:4b-it-qat", embedding_dim=512, device="cuda"):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device
        self.ollama_url = "http://localhost:11434/api/embeddings"
        
        # Project Gemma embeddings to desired size
        self.projection = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim)
        ).to(device)

    def forward(self, rgb_image, text="Pick the cube"):
        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=1)
            embedding_raw = torch.tensor(response.json()["embedding"], device=self.device).float().unsqueeze(0)
            
        except Exception as e:
            # Fallback to random embedding if Ollama fails
            embedding_raw = torch.randn(1, 4096, device=self.device)
        
        embedding = self.projection(embedding_raw)
        return embedding


class PolicyMLP(nn.Module):
    def __init__(self, embedding_dim=512, action_dim=4, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, embedding):
        return self.net(embedding)


class CombinedPolicy(nn.Module):
    def __init__(self, model_name="gemma3:4b-it-qat", action_dim=4, device="cuda"):
        super().__init__()
        self.device = device
        self.encoder = OllamaGemmaEncoder(model_name=model_name, embedding_dim=512, device=device)
        self.policy = PolicyMLP(embedding_dim=512, action_dim=action_dim).to(device)

    def forward(self, rgb_image, text="Pick the cube"):
        # print("RGB image shape:", rgb_image.shape)
        embedding = self.encoder(rgb_image, text)
        actions = self.policy(embedding)
        # print("Actions:", actions)
        return actions