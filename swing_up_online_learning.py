import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
import time

# Neural Network Dynamics Model
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim=128):
        super(DynamicsModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return state + self.network(x)  # Predict state difference (residual)

# Wrapper for CartPole environment to modify it for the swing-up task
class CartPoleSwingUpEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        # Create the standard CartPole environment
        env = gym.make('CartPole-v1', render_mode=render_mode)
        super(CartPoleSwingUpEnv, self).__init__(env)
        
        # Override action space for continuous control
        self.action_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(1,), dtype=np.float32
        )
        
        # Environment constants (from Gym's CartPole implementation)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5  # half the pole's length
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length
        self.max_force = 10.0
        self.dt = self.unwrapped.tau  # Use Gym's timestep
        
        # Task settings
        self.target_theta = 0.0  # upright position
        self.max_steps = 1000
        self.steps = 0
    
    def reset(self, **kwargs):
        # Reset to a random position with pendulum pointing down
        self.steps = 0
        obs, info = self.env.reset(**kwargs)
        
        # Access the internal state vector and modify it
        self.unwrapped.state = np.array([
            0.0,                    # Cart position: centered
            0.0,                    # Cart velocity: stationary
            np.pi,                  # Pole angle: pointing down (π radians from upright)
            0.0                     # Pole angular velocity: stationary
        ])
        
        # Get the observation from the modified state
        obs = np.array(self.unwrapped.state, dtype=np.float32)
        
        return obs, info
    
    def step(self, action):
        self.steps += 1
        
        # Extract continuous force value and clip it
        force = np.clip(action[0], -self.max_force, self.max_force)
        
        # Convert the continuous force to the binary action that Gym's CartPole expects
        # Positive force pushes right (action=1), negative force pushes left (action=0)
        discrete_action = 1 if force > 0 else 0
        
        # Scale the force to match what gym would apply
        # In Gym's CartPole, a fixed force of 10N is applied based on the discrete action
        # We'll modify the state directly after the step to account for our continuous force
        
        # Take a step in the environment with the discrete action
        obs, reward, terminated, truncated, info = self.env.step(discrete_action)
        
        # Now we need to adjust the state to account for our actual force magnitude
        # We'll modify the velocity components
        # The scaling factor is (our_force / gym_force)
        scaling = abs(force) / 10.0
        
        # Get the current state
        x, x_dot, theta, theta_dot = self.unwrapped.state
        
        # Adjust velocities based on our force scaling
        # We're simplifying here - a more accurate approach would be to recompute the physics
        # But this gives us a reasonable approximation
        x_dot = x_dot * scaling
        
        # Update the environment's state with our adjusted velocities
        self.unwrapped.state = np.array([x, x_dot, theta, theta_dot])
        
        # Get the updated observation
        obs = np.array(self.unwrapped.state, dtype=np.float32)
        
        # Calculate reward based on closeness to upright position
        # We want cos(theta) to be 1 (upright) instead of -1 (hanging down)
        upright = np.cos(theta)  # 1 when upright, -1 when hanging down
        x_penalty = 0.1 * (x**2)  # Small penalty for distance from center
        velocity_penalty = 0.1 * (x_dot**2 + theta_dot**2)  # Small penalty for high velocities
        control_penalty = 0.01 * (force**2)  # Small penalty for large forces
        
        # Reward: higher when upright
        reward = 3*upright - 0.1** theta_dot**2 # - x_penal velocity_penalty - control_penaltyty -
        
        # Only terminate if the cart goes too far
        terminated = abs(x) > self.unwrapped.x_threshold
        
        # Check for truncation (max steps)
        truncated = self.steps >= self.max_steps
        
        return obs, reward, terminated, truncated, info

# Replay Buffer for storing experiences
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.position = 0
        self.size = 0
    
    def add(self, state, action, next_state, reward):
        if self.size < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.next_states.append(next_state)
            self.rewards.append(reward)
            self.size += 1
        else:
            self.states[self.position] = state
            self.actions[self.position] = action
            self.next_states[self.position] = next_state
            self.rewards[self.position] = reward
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            np.array(self.states, dtype=np.float32)[indices],
            np.array(self.actions, dtype=np.float32)[indices],
            np.array(self.next_states, dtype=np.float32)[indices],
            np.array(self.rewards, dtype=np.float32)[indices]
        )
    
    def get_all(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            np.array(self.next_states, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32)
        )

# Train dynamics model
def train_dynamics_model(model, states, actions, next_states, epochs=100, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    states_tensor = torch.from_numpy(states)
    actions_tensor = torch.from_numpy(actions)
    next_states_tensor = torch.from_numpy(next_states)
    
    # Training loop
    losses = []
    dataset_size = len(states)
    
    for epoch in tqdm(range(epochs), desc="Training dynamics model"):
        epoch_loss = 0
        # Shuffle data
        indices = np.random.permutation(dataset_size)
        
        # Mini-batch training
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            state_batch = states_tensor[batch_indices]
            action_batch = actions_tensor[batch_indices]
            next_state_batch = next_states_tensor[batch_indices]
            
            # Forward pass
            predicted_next_states = model(state_batch, action_batch)
            loss = criterion(predicted_next_states, next_state_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * (end_idx - start_idx)
        
        avg_loss = epoch_loss / dataset_size
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses

# Cost function for swing-up task
def cost_function(states, actions, goal_state, Q, R, QN):
    """
    Compute cost for a trajectory
    Q: State cost matrix
    R: Control cost matrix
    QN: Terminal state cost matrix
    """
    T = len(actions)
    cost = 0
    
    # Running cost
    for t in range(T):
        state_diff = states[t] - goal_state
        cost += state_diff.dot(Q).dot(state_diff) + actions[t].dot(R).dot(actions[t])
    
    # Terminal cost
    state_diff = states[-1] - goal_state
    cost += state_diff.dot(QN).dot(state_diff)
    
    return cost

# iLQR algorithm
class iLQR:
    def __init__(self, dynamics_model, env, horizon=100):
        self.dynamics_model = dynamics_model
        self.env = env
        self.horizon = horizon
        
        # State and control dimensions
        self.state_dim = 4  # [x, x_dot, theta, theta_dot]
        self.control_dim = 1  # force
        
        # Cost function weights
        self.Q = np.diag([10, 0, 0 , 0.1])#([0.1, 0.1, 10.0, 0.1])  # State cost
        self.R = np.diag([0.01])                 # Control cost
        self.QN = np.diag([1.0, 1.0, 100.0, 1.0])  # Terminal cost
        
        # Goal state (upright pendulum, centered cart)
        self.goal_state = np.array([0.0, 0.0, 0.0, 0.0])
    
    def rollout(self, x0, us):
        """Simulate trajectory with given initial state and controls"""
        T = len(us)
        xs = np.zeros((T+1, self.state_dim))
        xs[0] = x0
        
        # Convert to tensor for NN dynamics
        states_tensor = torch.from_numpy(xs[0:1].astype(np.float32))
        
        for t in range(T):
            action_tensor = torch.from_numpy(us[t:t+1].astype(np.float32))
            next_state_tensor = self.dynamics_model(states_tensor, action_tensor)
            
            # Update state
            xs[t+1] = next_state_tensor.detach().numpy()[0]
            states_tensor = next_state_tensor.detach()
        
        return xs
    
    def compute_derivatives(self, xs, us):
        """Compute linearized dynamics and quadratized cost around trajectory"""
        T = len(us)
        fx = [None] * T  # df/dx
        fu = [None] * T  # df/du
        
        lx = [None] * (T+1)  # dl/dx
        lu = [None] * T      # dl/du
        lxx = [None] * (T+1) # d^2l/dx^2
        luu = [None] * T     # d^2l/du^2
        lux = [None] * T     # d^2l/dudx
        
        # Linearize dynamics using autograd
        for t in range(T):
            state = xs[t]
            action = us[t]
            
            # Create tensors
            state_tensor = torch.from_numpy(state.astype(np.float32))
            action_tensor = torch.from_numpy(action.astype(np.float32))
            
            # Enable autograd
            state_tensor.requires_grad_(True)
            action_tensor.requires_grad_(True)
            
            # Forward pass
            next_state_tensor = self.dynamics_model(state_tensor.unsqueeze(0), 
                                                    action_tensor.unsqueeze(0)).squeeze(0)
            
            # Compute Jacobians
            fx_t = np.zeros((self.state_dim, self.state_dim))
            fu_t = np.zeros((self.state_dim, self.control_dim))
            
            for i in range(self.state_dim):
                # Create a unit vector for backprop
                unit = torch.zeros_like(next_state_tensor)
                unit[i] = 1.0
                
                # Backprop
                next_state_tensor.backward(unit, retain_graph=True)
                
                # Extract gradients
                fx_t[i] = state_tensor.grad.numpy()
                fu_t[i] = action_tensor.grad.numpy()
                
                # Reset gradients
                state_tensor.grad.zero_()
                action_tensor.grad.zero_()
            
            fx[t] = fx_t
            fu[t] = fu_t
            
            # Compute cost derivatives
            state_diff = xs[t] - self.goal_state
            lx[t] = 2 * self.Q.dot(state_diff)
            lu[t] = 2 * self.R.dot(us[t])
            lxx[t] = 2 * self.Q
            luu[t] = 2 * self.R
            lux[t] = np.zeros((self.control_dim, self.state_dim))
        
        # Terminal cost derivatives
        state_diff = xs[T] - self.goal_state
        lx[T] = 2 * self.QN.dot(state_diff)
        lxx[T] = 2 * self.QN
        
        return fx, fu, lx, lu, lxx, luu, lux
    
    def backward_pass(self, fx, fu, lx, lu, lxx, luu, lux):
        """Backward pass to compute optimal control law"""
        T = len(fu)
        
        # Initialize value function
        Vx = lx[T]
        Vxx = lxx[T]
        
        # Gains
        k = [None] * T
        K = [None] * T
        
        for t in range(T-1, -1, -1):
            Qx = lx[t] + fx[t].T.dot(Vx)
            Qu = lu[t] + fu[t].T.dot(Vx)
            Qxx = lxx[t] + fx[t].T.dot(Vxx).dot(fx[t])
            Quu = luu[t] + fu[t].T.dot(Vxx).dot(fu[t])
            Qux = lux[t] + fu[t].T.dot(Vxx).dot(fx[t])
            
            # Ensure Quu is positive definite
            Quu_reg = Quu + 1e-3 * np.eye(self.control_dim)
            
            # Compute gains
            try:
                k[t] = -np.linalg.solve(Quu_reg, Qu)
                K[t] = -np.linalg.solve(Quu_reg, Qux)
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudoinverse
                k[t] = -np.linalg.pinv(Quu_reg).dot(Qu)
                K[t] = -np.linalg.pinv(Quu_reg).dot(Qux)
            
            # Update value function
            Vx = Qx + K[t].T.dot(Quu).dot(k[t]) + K[t].T.dot(Qu) + Qux.T.dot(k[t])
            Vxx = Qxx + K[t].T.dot(Quu).dot(K[t]) + K[t].T.dot(Qux) + Qux.T.dot(K[t])
            Vxx = 0.5 * (Vxx + Vxx.T)  # Ensure symmetry
        
        return k, K
    
    def forward_pass(self, xs, us, k, K, alpha=1.0):
        """Forward pass to compute new trajectory"""
        T = len(us)
        new_xs = np.zeros_like(xs)
        new_us = np.zeros_like(us)
        
        new_xs[0] = xs[0]
        
        # State tensor for dynamics model
        state_tensor = torch.from_numpy(new_xs[0:1].astype(np.float32))
        
        for t in range(T):
            # Compute feedback control
            state_diff = new_xs[t] - xs[t]
            new_us[t] = us[t] + alpha * k[t] + K[t].dot(state_diff)
            
            # Clip control
            new_us[t] = np.clip(new_us[t], -self.env.max_force, self.env.max_force)
            
            # Apply dynamics
            action_tensor = torch.from_numpy(new_us[t:t+1].astype(np.float32))
            next_state_tensor = self.dynamics_model(state_tensor, action_tensor)
            
            # Update state
            new_xs[t+1] = next_state_tensor.detach().numpy()[0]
            state_tensor = next_state_tensor.detach()
        
        return new_xs, new_us
    
    def optimize(self, x0, max_iterations=50):
        """Main iLQR optimization loop"""
        # Initialize with zero controls
        us = np.zeros((self.horizon, self.control_dim))
        
        for t in range(self.horizon):
        # Simple heuristic: push right if pole is on the right side, left otherwise
            us[t] = 1.0 if (t % 20 < 10) else -1.0  # Alternating every 10 steps
        
        # Initial rollout
        xs = self.rollout(x0, us)
        
        # Cost of initial trajectory
        cost = cost_function(xs, us, self.goal_state, self.Q, self.R, self.QN)
        
        # Optimization loop
        for iteration in range(max_iterations):
            print(f"Iteration {iteration+1}, Cost: {cost:.4f}")
            
            # Linearize dynamics and quadratize cost
            fx, fu, lx, lu, lxx, luu, lux = self.compute_derivatives(xs, us)
            
            # Backward pass
            k, K = self.backward_pass(fx, fu, lx, lu, lxx, luu, lux)
            
            # Line search
            alpha = 1.0
            max_line_search = 10
            ls_success = False
            
            for ls_iter in range(max_line_search):
                new_xs, new_us = self.forward_pass(xs, us, k, K, alpha)
                new_cost = cost_function(new_xs, new_us, self.goal_state, self.Q, self.R, self.QN)
                
                if new_cost < cost:
                    xs, us = new_xs, new_us
                    cost = new_cost
                    ls_success = True
                    break
                
                alpha *= 0.5
            
            # Check for convergence
            if not ls_success:
                print("Line search failed to improve cost")
                break
            
            if iteration > 0 and alpha < 1e-3:
                print("Converged (small step size)")
                break
        
        return xs, us, cost

# Create a policy from optimized trajectory
def create_ilqr_policy(ilqr, xs, us):
    # Get the feedback gains
    _, _, _, _, _, _, _ = ilqr.compute_derivatives(xs, us)
    k, K = ilqr.backward_pass(*ilqr.compute_derivatives(xs, us))
    
    def policy(state):
        # Find the closest state in the trajectory
        diffs = xs[:-1] - state
        distances = np.sum(diffs**2, axis=1)
        closest_idx = np.argmin(distances)
        
        # Apply feedback control
        state_diff = state - xs[closest_idx]
        action = us[closest_idx] + K[closest_idx].dot(state_diff)
        
        # Clip action
        action = np.clip(action, -ilqr.env.max_force, ilqr.env.max_force)
        return action.reshape(1)
    
    return policy

# Online iLQR with dynamics learning
def online_ilqr(env, initial_episodes=5, total_episodes=20, horizon=100, 
               dynamics_epochs=50, ilqr_iterations=50):
    # State and action dimensions
    state_dim = 4
    control_dim = 1
    
    # Create dynamics model
    dynamics_model = DynamicsModel(state_dim, control_dim)
    
    # Create replay buffer
    replay_buffer = ReplayBuffer()
    
    # Create iLQR controller
    ilqr_controller = iLQR(dynamics_model, env, horizon=horizon)
    
    # Variables to track progress
    episode_rewards = []
    model_losses = []
    
    # Initial exploration with random actions
    print("Initial exploration phase...")
    for episode in range(initial_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Store experience
            replay_buffer.add(obs, action, next_obs, reward)
            
            # Update state and reward
            obs = next_obs
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    # Main loop: alternating between learning dynamics and optimizing policy
    current_policy = None
    
    for episode in range(initial_episodes, total_episodes):
        # Train dynamics model on all data collected so far
        states, actions, next_states, _ = replay_buffer.get_all()
        print(f"Training dynamics model on {len(states)} samples...")
        losses = train_dynamics_model(dynamics_model, states, actions, next_states, epochs=dynamics_epochs)
        model_losses.extend(losses)
        
        # Optimize trajectory with iLQR using current dynamics model
        print(f"Optimizing trajectory for episode {episode+1}...")
        obs, _ = env.reset()
        xs, us, cost = ilqr_controller.optimize(obs, max_iterations=ilqr_iterations)
        
        # Create policy from optimized trajectory
        current_policy = create_ilqr_policy(ilqr_controller, xs, us)
        
        # Execute the optimized policy and collect more data
        print(f"Executing optimized policy for episode {episode+1}...")
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            # Use current policy
            action = current_policy(obs)
            
            # Step environment
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Store experience
            replay_buffer.add(obs, action, next_obs, reward)
            
            # Update state and reward
            obs = next_obs
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Cost = {cost:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(episode_rewards)+1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(model_losses)
    plt.yscale('log')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Dynamics Model Training Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return dynamics_model, current_policy, episode_rewards



# Evaluate a policy on the environment
def evaluate_policy(env, policy=None, num_episodes=3, render=True):
    total_rewards = []
    
    if render:
        env = RecordVideo(env, "videos/cartpole-swingup")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            if policy is None:
                # Random action if no policy provided
                action = env.action_space.sample()
            else:
                # Use the learned policy
                action = policy(obs)
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward}")
    
    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

# Main function
def main():
    # Create CartPole swing-up environment
    env = CartPoleSwingUpEnv(render_mode='human')   
    
    # Run online iLQR with dynamics learning
    print("Starting online iLQR with dynamics learning...")
    dynamics_model, final_policy, rewards = online_ilqr(
        env, 
        initial_episodes=5,   # Initial exploration episodes
        total_episodes=20,    # Total episodes to run
        horizon=100,#200,          # Planning horizon
        dynamics_epochs=50,   # Epochs for dynamics model training
        ilqr_iterations=15    # iLQR iterations per planning step
    )
    
    # Evaluate the final policy
    print("\nEvaluating final policy...")
    evaluate_policy(env, final_policy, num_episodes=3, render=True)

if __name__ == "__main__":
    main()