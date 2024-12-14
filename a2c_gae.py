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
        # Same network architecture
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

class A2CGAEAgent:
    def __init__(self, state_size: int, action_size: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.discount_factor = 0.99  # gamma
        self.gae_lambda = 0.95  # GAE lambda parameter
        self.learning_rate = 0.01
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        
        self.model = ActorCriticModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=self.learning_rate, 
                                  eps=1e-5)

    def get_action(self, state: np.ndarray) -> int:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.model(state)
            action_probs = action_probs.cpu().numpy().flatten()
        
        if np.random.random() > self.epsilon:
            action = np.argmax(action_probs)
        else:
            action = np.random.randint(self.action_size)
        
        return action

    def compute_gae(self, rewards, values, dones):
        # Calculate GAE
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards)-1)):
            delta = (rewards[t] + 
                    self.discount_factor * values[t+1] * (1 - dones[t]) - 
                    values[t])
            gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        # Add last advantage
        advantages[-1] = rewards[-1] - values[-1]
        
        # Calculate returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def train(self, trajectory):
        states, actions, rewards, dones = zip(*trajectory)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get values and action probabilities
        action_probs, values = self.model(states)
        values = values.squeeze()

        # Compute GAE and returns
        advantages, returns = self.compute_gae(rewards, values.detach(), dones)

        # Compute log probabilities
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()

        # Actor loss with entropy bonus
        actor_loss = -(log_probs * advantages).mean()
        entropy_loss = -(action_probs * torch.log(action_probs + 1e-10)).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values, returns)

        # Total loss
        loss = (actor_loss + 
                self.value_loss_coef * critic_loss - 
                self.entropy_coef * entropy_loss)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
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
    seed = 7
    init_seed(seed)

    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2CGAEAgent(state_size, action_size)

    wandb.init(
        project="a2c-gae-lunar-lander",         
        config={
            "seed": seed,
            "environment": "LunarLander-v3",
            "algorithm": "A2C GAE",
            "state_size": state_size,
            "action_size": action_size,
            "discount_factor": agent.discount_factor,
            "gae_lambda": agent.gae_lambda,
            "learning_rate": agent.learning_rate,
            "entropy_coefficient": agent.entropy_coef,
            "value_loss_coefficient": agent.value_loss_coef,
        }
    )

    scores, episodes = [], []
    EPOCHS = 10000
    TARGET_SCORE = 250

    from tqdm import tqdm
    for episode in tqdm(range(EPOCHS)):
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        state, _ = env.reset()
        trajectory = []
        done = False
        score = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            trajectory.append((state, action, reward, done))
            state = next_state
            score += reward

        actor_loss, critic_loss, entropy_loss = agent.train(trajectory)
        
        scores.append(score)
        episodes.append(episode)

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

    mean_reward, std_reward = agent.evaluate(env)
    
    wandb.log({
        "final_mean_reward": mean_reward, 
        "final_std_reward": std_reward
    })

    print(f"Evaluation: Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
    wandb.finish()

if __name__ == "__main__":
    main() 