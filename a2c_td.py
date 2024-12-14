import abc
from typing import Any
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from collections import deque

def init_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gym.utils.seeding.np_random(seed)

class ActorCriticModel(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(ActorCriticModel, self).__init__()
        # 네트워크 구조는 동일하게 유지
        self.shared_fc1 = nn.Linear(state_size, 512)
        self.shared_fc2 = nn.Linear(512, 256)
        self.shared_fc3 = nn.Linear(256, 128)
        
        self.actor_fc = nn.Linear(128, action_size)
        self.critic_fc = nn.Linear(128, 1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        x = F.relu(self.shared_fc3(x))
        
        action_probs = F.softmax(self.actor_fc(x), dim=-1)
        state_value = self.critic_fc(x)
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        return action_probs, state_value

class A2CTDAgent:
    def __init__(self, state_size: int, action_size: int, discount_factor: float = 0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.discount_factor = discount_factor
        self.learning_rate = 0.0007
        self.entropy_coef = 0.02
        self.value_loss_coef = 0.7
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        
        self.model = ActorCriticModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=self.learning_rate, 
                                  eps=1e-5)

    def get_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.model(state)
        
        action = np.random.choice(self.action_size, p=action_probs.cpu().numpy().flatten())
        return action

    def train_step(self, state, action, reward, next_state, done):
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        # Get current action probabilities and state value
        action_probs, state_value = self.model(state)
        _, next_state_value = self.model(next_state)

        # TD target (r + γV(s') if not done, else r)
        td_target = reward + self.discount_factor * next_state_value * (1 - done)
        
        # Advantage = TD target - V(s)
        advantage = (td_target - state_value).detach()

        # Actor loss
        log_prob = torch.log(action_probs.squeeze(0).gather(0, action) + 1e-8)
        actor_loss = -(log_prob * advantage).mean()
        
        # Entropy loss for exploration
        entropy_loss = -(action_probs * torch.log(action_probs + 1e-8)).mean()
        
        # Critic loss (MSE between TD target and current value estimate)
        critic_loss = F.mse_loss(state_value, td_target.detach())

        # Total loss
        loss = (actor_loss 
                + self.value_loss_coef * critic_loss 
                - self.entropy_coef * entropy_loss)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy_loss.item()

    def evaluate(self, env, num_episodes=10):
        eval_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                state = next_state

            eval_rewards.append(episode_reward)
            print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward}")

        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        env.close()  
        return mean_reward, std_reward

def main():
    # Set random seed
    seed = 7
    init_seed(seed)

    # Create environment
    env = gym.make('LunarLander-v3')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize agent
    agent = A2CTDAgent(state_size, action_size)

    # Wandb initialization
    wandb.init(
        project="a2c-td-lunar-lander",         
        config={
            "seed": seed,
            "environment": "LunarLander-v3",
            "algorithm": "A2C TD(0)",
            "state_size": state_size,
            "action_size": action_size,
            "discount_factor": agent.discount_factor,
            "learning_rate": agent.learning_rate,
            "entropy_coefficient": agent.entropy_coef,
            "value_loss_coefficient": agent.value_loss_coef,
        }
    )

    # Training loop
    scores, episodes = [], []
    EPOCHS = 10000
    TARGET_SCORE = 250

    from tqdm import tqdm
    for episode in tqdm(range(EPOCHS)):
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        state, _ = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # TD 학습 수행
            actor_loss, critic_loss, entropy_loss = agent.train_step(
                state, action, reward, next_state, done
            )
            
            state = next_state
            score += reward

        scores.append(score)
        episodes.append(episode)

        # Logging
        wandb.log({
            "episode": episode + 1, 
            "score": score, 
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy_loss": entropy_loss,
            "epsilon": agent.epsilon
        })

        if score >= TARGET_SCORE:
            print(f"Target score reached in episode {episode + 1}")
            break

    # Final evaluation
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, scores)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    mean_reward, std_reward = agent.evaluate(env)
    
    wandb.log({
        "final_mean_reward": mean_reward, 
        "final_std_reward": std_reward
    })

    print(f"Evaluation: Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
    
    wandb.finish()

if __name__ == "__main__":
    main()