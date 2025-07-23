import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
from collections import deque
#import pandas as pd

# Hyperparameters
DELTA_TRAIN = True  
HORIZON = 200  # Increased horizon for better planning
ILQR_ITER = 20  # More iterations for convergence
RANDOM_EPISODES = 10  # More initial exploration
TOTAL_EPISODES = 50  # More training episodes
MAX_STEPS = 1000 
EPOCHS = 100  # More training epochs
BATCH_SIZE = 128  # Larger batch size
LEARNING_RATE = 1e-4  # Lower learning rate for stability

# Network architecture
HIDDEN_DIM = 512  # Larger network
NUM_LAYERS = 3  # Deeper network

# Visualization
PLOT_REALTIME = True
PLOT_INTERVAL = 10
PLOT_HISTORY = 200
def update_plots():
    # Remove the clear command as it can interfere
    ax1.clear()
    ax2.clear()
    
    ax1.plot(position_errors, 'b-')
    ax1.set_title('Position Error (x - target)')
    ax1.axhline(0, color='r', linestyle='--')  # Add target line
    ax1.set_ylabel('Error')
    ax1.grid(True)
    
    ax2.plot(loss_history, 'r-')
    ax2.set_title('Neural Network Training Loss')
    ax2.set_ylabel('Loss')
    #ax2.set_xlabel('Training Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)  # REQUIRED for live updates
    
class StateMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.timesteps = []
        self.current_episode = 0

    def record(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.timesteps.append(len(self.states))

    def get_recent_data(self, window=PLOT_HISTORY):
        if len(self.states) == 0:
            return None, None, None, None

        window = min(window, len(self.states))
        recent_states = np.array(self.states[-window:])
        recent_actions = np.array(self.actions[-window:])
        recent_rewards = np.array(self.rewards[-window:])
        recent_timesteps = np.array(self.timesteps[-window:])

        return recent_states, recent_actions, recent_rewards, recent_timesteps

class CartPoleSwingUpEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make('CartPole-v1', render_mode=render_mode)
        super().__init__(env)

        self.max_force = 10.0
        self.state_monitor = StateMonitor()
        self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        
        # Target state
        self.target_theta = 0.0  # upright position
        self.steps = 0
        
        # Physics constants
        self.gravity = self.env.unwrapped.gravity
        self.masscart = self.env.unwrapped.masscart
        self.masspole = self.env.unwrapped.masspole
        self.total_mass = self.masscart + self.masspole
        self.length = self.env.unwrapped.length
        self.polemass_length = self.masspole * self.length
        self.dt = self.env.unwrapped.tau

    def reset(self):
        self.steps = 0
        obs, info = self.env.reset()
        
        # Start with pendulum down
        self.unwrapped.state = np.array([
            0.0,  # Cart position
            0.0,  # Cart velocity
            np.pi,  # Pole angle (down)
            0.0   # Pole angular velocity
        ])
        
        obs = np.array(self.unwrapped.state, dtype=np.float32)
        self.state_monitor.reset()
        self.state_monitor.current_episode += 1
        return obs, info

    def step(self, action):
        self.steps += 1
        force = np.clip(action[0], -self.max_force, self.max_force)
        
        # Get current state
        x, x_dot, theta, theta_dot = self.unwrapped.state
        
        # Physics calculations
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Update state (Euler integration)
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc
        
        self.unwrapped.state = np.array([x, x_dot, theta, theta_dot])
        obs = np.array(self.unwrapped.state, dtype=np.float32)
        
        # Reward shaping
        upright = np.cos(theta)
        x_penalty = 0.1 * x**2  # More penalty for being off-center
        velocity_penalty = 0.01 * (x_dot**2 + theta_dot**2)
        control_penalty = 0.001 * force**2
        
        reward = 10 * upright - x_penalty - velocity_penalty - control_penalty
        
        # Termination conditions
        terminated = abs(x) > 2.4  # Slightly larger threshold
        truncated = self.steps >= MAX_STEPS
        
        self.state_monitor.record(self.unwrapped.state.copy(), action.copy(), reward)
        return obs, reward, terminated, truncated, {}

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, next_state, reward):
        self.buffer.append((state, action, next_state, reward))
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        states, actions, next_states, rewards = zip(*samples)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(rewards, dtype=np.float32)
        )
        
    def __len__(self):
        return len(self.buffer)

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        
        # More sophisticated network architecture
        layers = []
        input_dim = state_dim + control_dim
        
        # Create multiple hidden layers
        for _ in range(NUM_LAYERS):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.loss_history = []
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        if DELTA_TRAIN:
            return state + self.network(x)
        return self.network(x)
        
    def rollout(self, x0, us):
        """Simulate trajectory with given initial state and controls"""
        T = len(us)
        xs = np.zeros((T+1, len(x0)))
        xs[0] = x0
        
        state_tensor = torch.from_numpy(x0.astype(np.float32)).unsqueeze(0)
        
        for t in range(T):
            action_tensor = torch.from_numpy(us[t].astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                next_state_tensor = self(state_tensor, action_tensor)
            xs[t+1] = next_state_tensor.numpy()[0]
            state_tensor = next_state_tensor
            
        return xs
        
    def train_batch(self, states, actions, next_states):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)
        
        dataset = torch.utils.data.TensorDataset(states, actions, next_states)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        losses = []
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for state_batch, action_batch, next_state_batch in loader:
                pred_next_state = self(state_batch, action_batch)
                loss = self.loss_fn(pred_next_state, next_state_batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Gradient clipping
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            self.loss_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")
                
        return losses

class iLQR:
    def __init__(self, dynamics_model, env, horizon=HORIZON):
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.env = env
        
        # Cost function weights - tuned for swing-up
        self.Q = np.diag([5.0, 0.1, 10.0, 0.1])  # State cost
        self.R = np.diag([0.1])  # Control cost
        self.QN = np.diag([10.0, 0.5, 20.0, 0.5])  # Terminal cost
        
        self.goal_state = np.array([0.0, 0.0, 0.0, 0.0])  # Upright position
        
    def control_and_update(self):
        replay_buffer = ReplayBuffer()
        episode_rewards = []
        
        # Initial random exploration
        print("Initial exploration...")
        for episode in range(RANDOM_EPISODES):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            for _ in range(MAX_STEPS):
                action = self.env.action_space.sample()
                next_obs, reward, done, truncated, _ = self.env.step(action)
                replay_buffer.add(obs, action, next_obs, reward)
                obs = next_obs
                episode_reward += reward
                
                if done or truncated:
                    break
                    
            episode_rewards.append(episode_reward)
            print(f"Episode {episode+1}, Reward: {episode_reward:.1f}")
        
        # Main training loop
        for episode in range(RANDOM_EPISODES, TOTAL_EPISODES):
            # Train dynamics model
            if len(replay_buffer) > BATCH_SIZE:
                states, actions, next_states, _ = replay_buffer.sample(min(len(replay_buffer), 10000))
                self.dynamics_model.train_batch(states, actions, next_states)
            
            # Run episode with iLQR
            obs, _ = self.env.reset()
            episode_reward = 0
            
            # Initialize controls with small random values
            us = np.random.normal(0, 0.5, (self.horizon, 1))
            
            for step in range(MAX_STEPS):
                # Optimize trajectory
                xs, us, _ = self.optimize(obs, us, max_iterations=ILQR_ITER)
                
                # Execute first action
                action = us[0]
                next_obs, reward, done, truncated, _ = self.env.step(action)
                replay_buffer.add(obs, action, next_obs, reward)
                
                # Shift controls for next step
                us = np.roll(us, -1, axis=0)
                us[-1] = 0  # Zero the last action
                
                obs = next_obs
                episode_reward += reward
                
                if done or truncated:
                    break
                    
            episode_rewards.append(episode_reward)
            print(f"Episode {episode+1}, Reward: {episode_reward:.1f}")
            
        return episode_rewards
    
    def optimize(self, x0, us_init, max_iterations=ILQR_ITER):
        """iLQR optimization"""
        us = us_init.copy()
        xs = self.dynamics_model.rollout(x0, us)
        best_cost = self.cost_function(xs, us)
        
        for _ in range(max_iterations):
            # Backward pass
            k, K = self.backward_pass(xs, us)
            
            # Forward pass with line search
            alpha = 1.0
            for _ in range(10):
                new_xs, new_us = self.forward_pass(xs, us, k, K, alpha)
                new_cost = self.cost_function(new_xs, new_us)
                
                if new_cost < best_cost:
                    xs, us = new_xs, new_us
                    best_cost = new_cost
                    break
                    
                alpha *= 0.5
                
        return xs, us, best_cost
        
    def cost_function(self, xs, us):
        cost = 0
        for t in range(len(us)):
            state_diff = xs[t] - self.goal_state
            cost += state_diff.T @ self.Q @ state_diff + us[t].T @ self.R @ us[t]
        
        # Terminal cost
        state_diff = xs[-1] - self.goal_state
        cost += state_diff.T @ self.QN @ state_diff
        return cost
        
    def backward_pass(self, xs, us):
        T = len(us)
        
        # Compute derivatives
        fx, fu = self.compute_dynamics_derivatives(xs, us)
        
        # Initialize value function
        Vx = 2 * self.QN @ (xs[-1] - self.goal_state)
        Vxx = 2 * self.QN
        
        k = [None] * T
        K = [None] * T
        
        for t in range(T-1, -1, -1):
            # Cost derivatives
            lx = 2 * self.Q @ (xs[t] - self.goal_state)
            lu = 2 * self.R @ us[t]
            lxx = 2 * self.Q
            luu = 2 * self.R
            
            # Q-function terms
            Qx = lx + fx[t].T @ Vx
            Qu = lu + fu[t].T @ Vx
            Qxx = lxx + fx[t].T @ Vxx @ fx[t]
            Quu = luu + fu[t].T @ Vxx @ fu[t]
            Qux = fu[t].T @ Vxx @ fx[t]
            
            # Regularization
            Quu += 1e-3 * np.eye(Quu.shape[0])
            
            # Compute gains
            k[t] = -np.linalg.solve(Quu, Qu)
            K[t] = -np.linalg.solve(Quu, Qux)
            
            # Update value function
            Vx = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu + Qux.T @ k[t]
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]
            Vxx = 0.5 * (Vxx + Vxx.T)  # Ensure symmetry
            
        return k, K
        
    def compute_dynamics_derivatives(self, xs, us):
        T = len(us)
        fx = []
        fu = []
        
        for t in range(T):
            state = torch.FloatTensor(xs[t]).requires_grad_(True)
            action = torch.FloatTensor(us[t]).requires_grad_(True)
            
            # Compute Jacobians
            jacobian_x = []
            jacobian_u = []
            
            # Compute derivatives for each output dimension
            for i in range(state.shape[0]):
                # Forward pass
                next_state = self.dynamics_model(state.unsqueeze(0), action.unsqueeze(0))
                
                # Backward pass for dx/df
                grad_x = torch.autograd.grad(next_state[0, i], state, retain_graph=True)[0]
                grad_u = torch.autograd.grad(next_state[0, i], action, retain_graph=True)[0]
                
                jacobian_x.append(grad_x.detach().numpy())
                jacobian_u.append(grad_u.detach().numpy())
                
            fx_t = np.stack(jacobian_x)
            fu_t = np.stack(jacobian_u)
            
            if DELTA_TRAIN:
                fx_t += np.eye(fx_t.shape[0])  # Add identity for delta prediction
                
            fx.append(fx_t)
            fu.append(fu_t)
            
        return fx, fu
        
    def forward_pass(self, xs, us, k, K, alpha=1.0):
        T = len(us)
        new_xs = np.zeros_like(xs)
        new_us = np.zeros_like(us)
        new_xs[0] = xs[0]
        
        state_tensor = torch.FloatTensor(new_xs[0]).unsqueeze(0)
        
        for t in range(T):
            # Compute feedback control
            state_diff = new_xs[t] - xs[t]
            new_us[t] = us[t] + alpha * k[t] + K[t] @ state_diff
            new_us[t] = np.clip(new_us[t], -self.env.max_force, self.env.max_force)
            
            # Apply dynamics
            action_tensor = torch.FloatTensor(new_us[t]).unsqueeze(0)
            with torch.no_grad():
                next_state_tensor = self.dynamics_model(state_tensor, action_tensor)
            new_xs[t+1] = next_state_tensor.numpy()[0]
            state_tensor = next_state_tensor
            
        return new_xs, new_us

def main():
    env = CartPoleSwingUpEnv()
    
    state_dim = env.observation_space.shape[0]
    control_dim = env.action_space.shape[0]
    
    dynamics = DynamicsModel(state_dim, control_dim)
    controller = iLQR(dynamics, env)
    
    rewards = controller.control_and_update()
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.plot(rewards)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.show()
    
    # Save results
    #pd.DataFrame({'reward': rewards}).to_csv('training_results.csv', index=False)

if __name__ == "__main__":
    main()