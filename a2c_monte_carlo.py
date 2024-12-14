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
        # Deeper network with more capacity
        self.shared_fc1 = nn.Linear(state_size, 512)
        self.shared_fc2 = nn.Linear(512, 256)
        self.shared_fc3 = nn.Linear(256, 128)
        
        self.actor_fc = nn.Linear(128, action_size)
        self.critic_fc = nn.Linear(128, 1)
        
        # Weight initialization
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
        # Ensure no NaN
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        return action_probs, state_value

class A2CMonteCarloAgent:
    def __init__(self, state_size: int, action_size: int, 
                 discount_factor: float = 0.99,
                 learning_rate: float = 0.0005,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.05,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters from arguments
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize model and optimizer
        self.model = ActorCriticModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.learning_rate, 
                                    eps=1e-5)

    def get_action(self, state: np.ndarray) -> int:
        # Convert state to tensor and move to device
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities
        with torch.no_grad():
            action_probs, _ = self.model(state)
            action_probs = action_probs.cpu().numpy().flatten()
        
        # Epsilon-greedy action selection
        if np.random.random() > self.epsilon:
            action = np.argmax(action_probs)  # Greedy action
        else:
            action = np.random.randint(self.action_size)  # Random action
        
        return action

    def train(self, trajectory):
        # Unpack trajectory
        states, actions, rewards, dones = zip(*trajectory)

        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute returns using Monte Carlo estimation
        returns = []
        G = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + self.discount_factor * G * (1 - done)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Forward pass
        action_probs, state_values = self.model(states)
        state_values = state_values.squeeze()

        # Compute log probabilities
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())

        # Compute advantages
        advantages = returns - state_values.detach()

        # Actor loss with entropy bonus
        actor_loss = -(log_probs * advantages).mean()
        entropy_loss = -(action_probs * torch.log(action_probs + 1e-10)).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(state_values, returns)

        # Total loss
        loss = (actor_loss 
                + self.value_loss_coef * critic_loss 
                - self.entropy_coef * entropy_loss)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
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

        return np.mean(eval_rewards), np.std(eval_rewards)

def plot_rewards(episodes, rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Set random seed for reproducibility
    seed = 7
    init_seed(seed)

    # Create environment
    env = gym.make('LunarLander-v3')
    
    # Get state and action sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Hyperparameters
    hyperparams = {
        "discount_factor": 0.99,
        "learning_rate": 0.0005,
        "entropy_coef": 0.01,
        "value_loss_coef": 0.05,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.01
    }

    # Initialize agent with hyperparameters
    agent = A2CMonteCarloAgent(
        state_size=state_size,
        action_size=action_size,
        **hyperparams
    )

    # Wandb initialization
    wandb.init(
        project="a2c-monte-carlo-lunar-lander",         
        config={
            "seed": seed,
            "environment": "LunarLander-v3",
            "algorithm": "A2C Monte Carlo",
            "state_size": state_size,
            "action_size": action_size,
            **hyperparams  # Include all hyperparameters in wandb config
        }
    )

    # Training loop
    scores, episodes = [], []
    EPOCHS = 5000
    TARGET_SCORE = 250  # Slightly adjusted target score

    from  tqdm import tqdm
    for episode in tqdm(range(EPOCHS)):
        # epsilon 감소
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

        # Train on complete trajectory
        actor_loss, critic_loss, entropy_loss = agent.train(trajectory)
        
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

        # print(f"Episode {episode + 1}: SCORE: {score:.2f}, Epsilon: {agent.epsilon:.4f}")

        # Early stopping condition
        if score >= TARGET_SCORE:
            print(f"Target score reached in episode {episode + 1}")
            break

    # Final evaluation
    # plot_rewards(episodes, scores)
    mean_reward, std_reward = agent.evaluate(env)
    
    wandb.log({
        "final_mean_reward": mean_reward, 
        "final_std_reward": std_reward
    })

    print(f"Evaluation: Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
    
    # Close environment
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()