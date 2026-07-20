import torch
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from env import VisionArmEnv
from model import VisionActorCritic

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
EPISODE_LENGTH = 250   # How many frames to collect before pausing to learn (~1 second)
GAMMA = 0.99           # Discount factor for future rewards

def preprocess_image(obs):
    """
    The Data Bridge: Converts PyBullet's (128, 128, 3) NumPy array 
    into PyTorch's (1, 3, 128, 128) normalized float Tensor.
    """
    # Swap axes from (Height, Width, Channels) to (Channels, Height, Width)
    img = np.transpose(obs, (2, 0, 1))
    # Add a batch dimension at the front
    img = np.expand_dims(img, axis=0)
    # Convert to Tensor and normalize pixels to [0, 1]
    tensor = torch.FloatTensor(img) / 255.0
    return tensor

def train():
    # Initialize Environment and Model
    env = VisionArmEnv()
    model = VisionActorCritic()
    
    # The Optimizer updates the neural network weights based on the calculated loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n=======================================================")
    print("RLHF Training Started.")
    print("Click the PyBullet window.")
    print("Up = GOOD (+1) | Right = OKAY (0) | Down = NO (-1)")
    print("=======================================================\n")

    try:
        episode = 1
        while True:
            # Memory buffers for this specific episode
            log_probs = []
            values = []
            rewards = []
            
            obs, _ = env.reset()
            
            # --- Data Collection Phase ---
            for step in range(EPISODE_LENGTH):
                state_tensor = preprocess_image(obs)
                
                # 1. Look at the camera feed and decide what to do
                action_mean, action_std, state_value = model(state_tensor)
                
                # 2. Add exploration (randomness) to the action so it tries new things
                distribution = dist.Normal(action_mean, action_std)
                action = distribution.sample()
                
                # Calculate how likely it was to take this specific action
                log_prob = distribution.log_prob(action).sum(dim=-1)
                
                # 3. Execute the physical movement in PyBullet
                # .squeeze() removes the batch dimension to get a simple [3] array for the motors
                numpy_action = action.squeeze(0).detach().numpy()
                
                # 4. Step the simulation and capture your keyboard feedback
                next_obs, reward, done, truncated, _ = env.step(numpy_action)
                
                # Store the memory
                log_probs.append(log_prob)
                values.append(state_value)
                rewards.append(reward)
                
                obs = next_obs
            
            # --- Learning Phase (Backpropagation) ---
            # Calculate the cumulative discounted reward from your keyboard presses
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + GAMMA * G
                returns.insert(0, G)
            
            returns = torch.tensor(returns)
            # Normalize the returns to stabilize the math
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            actor_loss = 0
            critic_loss = 0
            
            # Compare what the Critic thought would happen vs. what you actually rewarded it for
            for log_prob, value, R in zip(log_probs, values, returns):
                advantage = R - value.item()
                
                # Actor loss: Encourage actions that resulted in positive advantage (your +1s)
                actor_loss -= log_prob * advantage
                
                # Critic loss: Train the Critic to better guess your future rewards
                critic_loss += torch.nn.functional.mse_loss(value, torch.tensor([[R]]))
            
            total_loss = actor_loss + critic_loss
            
            # Update the neural network weights
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            print(f"Episode {episode} Complete | Total Human Reward: {sum(rewards)} | Loss: {total_loss.item():.4f}")
            episode += 1

    except KeyboardInterrupt:
        print("\nTraining manually stopped. Saving model weights...")
        torch.save(model.state_dict(), "rlhf_arm_weights.pth")
        print("Weights saved to 'rlhf_arm_weights.pth'")
        env.physicsClient.disconnect()

if __name__ == "__main__":
    train()