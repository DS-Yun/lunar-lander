import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb
import matplotlib.pyplot as plt
from  tqdm import tqdm

def init_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gym.utils.seeding.np_random(seed)
    torch.set_num_threads(10)  # 코어 수를 일부로 제한

class ActorCriticModel(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(ActorCriticModel, self).__init__()
        self.shared_fc1 = nn.Linear(state_size, 512)
        self.shared_fc2 = nn.Linear(512, 256)
        self.shared_fc3 = nn.Linear(256, 128)
        self.actor_fc = nn.Linear(128, action_size)
        self.critic_fc = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        x = F.relu(self.shared_fc3(x))
        
        action_probs = F.softmax(self.actor_fc(x), dim=-1)
        state_value = self.critic_fc(x)
        return action_probs, state_value

class A2CGAEAgent:
    def __init__(self, state_size: int, action_size: int, 
                 discount_factor: float = 0.99,
                 learning_rate: float = 0.0005,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.05,
                 epsilon: float = 1.0,
                 gae_lambda: float = 0.95
                 ):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cpu")        
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters from arguments
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda

        self.model = ActorCriticModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=self.learning_rate, 
                                  eps=1e-5)

    def get_action(self, state: np.ndarray) -> int:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities
        with torch.no_grad():
            action_probs, _ = self.model(state)
            dist = Categorical(action_probs)
            action = dist.sample()
        
        return action.item() #, dist.log_prob(action)

    def compute_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards)-1)):
            delta = (rewards[t] + 
                    self.discount_factor * values[t+1] * (1 - dones[t]) - 
                    values[t])
            gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        # Add last advantage
        advantages[-1] = rewards[-1] + self.discount_factor * (1 - dones[-1]) * values[-1].detach() - values[-1].detach()
        
        # Calculate returns
        returns = advantages + values
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return advantages, returns

    def train(self, trajectory):
        # Unpack trajectory
        states, actions, rewards, dones = zip(*trajectory)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Forward pass  
        action_probs, state_values = self.model(states)
        state_values = state_values.squeeze()    

        # Compute GAE and returns
        advantages, returns = self.compute_gae(rewards, state_values.detach(), dones)

        # Compute log probabilities
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()

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

        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropy_loss.item()

    def evaluate(self, env, num_episodes=10, render=True):
        # env = gym.make('LunarLander-v3', render_mode='human' if render else None)
        eval_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0

            while not (done or truncated):
                action = self.get_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                state = next_state

            eval_rewards.append(episode_reward)
            print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward}")
       
        env.close()
        return np.mean(eval_rewards), np.std(eval_rewards)

def main():
    seed = 7
    init_seed(seed)

    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    wandb.init(
        project="a2c-td-lunar-lander", 
        config={
            "seed": seed,
            "environment": "LunarLander-v3",
            "algorithm": "A2C Monte Carlo",
            "state_size": state_size,
            "action_size": action_size,
            "epsilon": 1
        }
    )        
    
    # Hyperparameters
    hyperparams = {
        "discount_factor": wandb.config.discount_factor,
        "learning_rate": wandb.config.learning_rate,
        "entropy_coef": wandb.config.entropy_coef,
        "value_loss_coef": wandb.config.value_loss_coef,
        "epsilon": wandb.config.epsilon,
        "gae_lambda": wandb.config.gae_lambda
    }

    agent = A2CGAEAgent(state_size, action_size, **hyperparams)    


    # Training loop
    scores, episodes = [], []
    EPOCHS = 3000
    TARGET_SCORE = 260

    for episode in range(EPOCHS):
        state, _ = env.reset()
        trajectory = []
        done = False
        truncated = False
        score = 0

        while not (done or truncated):
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            trajectory.append((state, action, reward, done))
            state = next_state
            score += reward

        actor_loss, critic_loss, entropy_loss = agent.train(trajectory)
        
        scores.append(score)
        episodes.append(episode)
        avg_score = np.mean(scores[-min(30, len(scores)):])

        # Logging
        wandb.log({
            "episode": episode + 1, 
            "score": score, 
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy_loss": entropy_loss,
            "epsilon": agent.epsilon,
            "avg_score": avg_score,
            "gae_lambda": agent.gae_lambda
        })

        ## 
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Average Score: {avg_score:.2f}")

        if avg_score >= TARGET_SCORE:
            print(f"Target score reached in episode {episode + 1}")
            break
        
    print(
        f'episode:{episode}, score:{score:.3f}, avg_score:{avg_score:.3f}'
    ) 
    mean_reward, std_reward = agent.evaluate(env)
    
    wandb.log({
        "final_mean_reward": mean_reward, 
        "final_std_reward": std_reward
    })

    # print(f"Evaluation: Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
    wandb.finish()

if __name__ == "__main__":
    # Sweep 설정
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'final_mean_reward',
            'goal': 'maximize'
        },
        'parameters': {
            # 'learning_rate': {'min': 0.0004, 'max': 0.0006},
            'learning_rate': {'min': 0.0002, 'max': 0.004},
            'entropy_coef': {'min': 0.02, 'max': 0.03},# 0.02},
            'value_loss_coef': {'min': 0.01, 'max': 0.03},
            'discount_factor': {'values': [0.99]},
            'gae_lambda': {'values': [0.0]},        
        }
    }
    
    # Sweep 초기화 및 실행
    sweep_id = wandb.sweep(sweep_config, project="a2c-td-lunar-lander")
    wandb.agent(sweep_id, function=main, count=20)  # 20번의 실험 실행    
    main()