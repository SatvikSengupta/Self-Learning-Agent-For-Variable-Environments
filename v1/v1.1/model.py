import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VisionActorCritic(nn.Module):
    def __init__(self):
        super(VisionActorCritic, self).__init__()
        
        # =================================================================
        # 1. VISION ENCODER (CNN)
        # Input: (Batch, 3, 128, 128) - PyTorch expects Channels-First
        # =================================================================
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # After these convolutions, a 128x128 image becomes a 12x12 feature map
        # 64 channels * 12 height * 12 width = 9216 flat features
        self.fc_shared = nn.Linear(9216, 512)
        
        # =================================================================
        # 2. ACTOR HEAD (Policy)
        # Outputs target joint angles for (Shoulder, Elbow, Wrist)
        # =================================================================
        self.actor_mean = nn.Linear(512, 3)
        
        # We need a log standard deviation parameter so the agent can "explore" 
        # by adding random noise to its actions during early training.
        self.actor_logstd = nn.Parameter(torch.zeros(1, 3)) 
        
        # =================================================================
        # 3. CRITIC HEAD (Value Function)
        # Outputs a single number evaluating the current state
        # =================================================================
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        # Pass image through CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the 3D tensor into a 1D vector
        x = x.view(x.size(0), -1) 
        
        # Create the shared latent representation
        latent = F.relu(self.fc_shared(x))
        
        # --- ACTOR ---
        # The Tanh activation bounds the output between -1 and 1. 
        # We multiply by 1.57 to match our URDF joint limits (-1.57 to +1.57 radians).
        action_mean = torch.tanh(self.actor_mean(latent)) * 1.57
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # --- CRITIC ---
        state_value = self.critic(latent)
        
        return action_mean, action_std, state_value

# Quick test to ensure the dimensions compile correctly
if __name__ == "__main__":
    model = VisionActorCritic()
    
    # Create a dummy image tensor: Batch of 1, 3 Channels, 128x128 Resolution
    # (Notice we divide by 255.0 to normalize the pixel values between 0 and 1)
    dummy_image = torch.zeros(1, 3, 128, 128) 
    
    mean, std, value = model(dummy_image)
    print(f"Action Mean (Joint Target): {mean.detach().numpy()}")
    print(f"Action Standard Deviation:  {std.detach().numpy()}")
    print(f"Critic State Value:         {value.detach().numpy()}")