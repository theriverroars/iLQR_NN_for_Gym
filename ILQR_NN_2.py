import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register

# Switch between NN and true dynamics
NN_MODE = False  # Set to False to use true dynamics
EXPLORATION_NOISE = 0.2  # For exploration in the environment        

# Environment parameters (needed for true dynamics)
GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5  # Actually half the pole's length
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # seconds between state updates

# Register swing-up environment
register(
    id='CartPoleSwingUp-v0',
    entry_point='gymnasium_cartpole_swingup:CartPoleSwingUpEnv',
)

def true_cartpole_dynamics(x, u):
    """True dynamics of cartpole system with Jacobians"""
    x, x_dot, theta, theta_dot = x
    force = FORCE_MAG * u[0]
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    
    # Dynamics calculations
    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (4.0/3.0 - MASSPOLE * costheta**2 / TOTAL_MASS))
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS
    
    # Next state (Euler integration)
    x_next = np.array([
        x + TAU * x_dot,
        x_dot + TAU * xacc,
        theta + TAU * theta_dot,
        theta_dot + TAU * thetaacc
    ])
    
    # Jacobian calculations (A and B matrices)
    # ... [complex Jacobian calculations would go here] ...
    # For simplicity, we'll use finite differences for Jacobians
    eps = 1e-5
    A = np.zeros((4, 4))
    B = np.zeros((4, 1))
    
    # Finite differences for A matrix
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        x_perturbed = x + dx
        x_next_perturbed = true_cartpole_dynamics(x_perturbed, u)[0]
        A[:, i] = (x_next_perturbed - x_next) / eps
    
    # Finite differences for B matrix
    du = np.array([eps])
    x_next_perturbed = true_cartpole_dynamics(x, u + du)[0]
    B[:, 0] = (x_next_perturbed - x_next) / eps
    
    return x_next, A, B

class iLQRController:
    def __init__(self, dynamics_model, horizon=50, max_iter=10, exploration_noise = EXPLORATION_NOISE, Q_terminal=None, R=0.01):
        self.horizon = horizon
        self.max_iter = max_iter
        self.Q_terminal = Q_terminal if Q_terminal is not None else np.diag([100, 10, 10, 1])
        self.R = R * np.eye(1)
        self.dynamics_model = dynamics_model
        self.position_errors = []

    def compute_trajectory(self, x0, u_seq):
        x_seq = [x0]
        for u in u_seq:
            if NN_MODE:
                x_next = self.dynamics_model.predict(x_seq[-1], u)
            else:
                x_next, _, _ = true_cartpole_dynamics(x_seq[-1], u)
            x_seq.append(x_next)
        return np.array(x_seq)

    def backward_pass(self, x_seq, u_seq, A_list, B_list):
        Vx = self.Q_terminal @ x_seq[-1]
        Vxx = self.Q_terminal
        k = np.zeros((self.horizon, 1))
        K = np.zeros((self.horizon, 1, x_seq[0].shape[0]))

        for t in reversed(range(self.horizon)):
            A, B = A_list[t], B_list[t]
            Qx = A.T @ Vx
            Qu = B.T @ Vx + self.R @ u_seq[t]
            Qxx = A.T @ Vxx @ A
            Quu = B.T @ Vxx @ B + self.R
            Qux = B.T @ Vxx @ A

            Quu_inv = np.linalg.inv(Quu)
            k[t] = -Quu_inv @ Qu
            K[t] = -Quu_inv @ Qux

            Vx = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]

        return k, K
    
    def forward_pass(self, x0, u_seq, k, K):
        new_u = np.zeros_like(u_seq)
        x = x0.copy()
        new_x = [x0]
        total_cost = 0

        for t in range(self.horizon):
            new_u[t] = u_seq[t] + k[t] + K[t] @ (x - new_x[t])
            new_u[t] = np.clip(new_u[t], -1, 1)
            
            if NN_MODE:
                x_next = self.dynamics_model.predict(x, new_u[t])
            else:
                x_next, _, _ = true_cartpole_dynamics(x, new_u[t])
                
            x = x_next
            new_x.append(x)
            total_cost += 0.5 * new_u[t].T @ self.R @ new_u[t]
            self.position_errors.append(abs(x[0]))  # Track cart position error

        total_cost += 0.5 * x.T @ self.Q_terminal @ x
        return np.array(new_x), new_u, total_cost

    def optimize(self, x0, u_guess):
        u_seq = u_guess
        x_seq = self.compute_trajectory(x0, u_seq)

        for _ in range(self.max_iter):
            A_list, B_list = [], []
            for t in range(self.horizon):
                if NN_MODE:
                    A, B = self.dynamics_model.get_jacobians(x_seq[t], u_seq[t])
                else:
                    _, A, B = true_cartpole_dynamics(x_seq[t], u_seq[t])
                A_list.append(A)
                B_list.append(B)

            k, K = self.backward_pass(x_seq, u_seq, A_list, B_list)
            x_seq, u_seq, _ = self.forward_pass(x0, u_seq, k, K)
            x_seq = x_seq[:-1]

        return u_seq

class DynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.X_train, self.y_train = [], []
        self.losses = []
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def predict(self, x, u):
        x = np.asarray(x).flatten()
        u = np.asarray(u).flatten()
        inp = torch.tensor(np.hstack([x, u]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.net(inp).squeeze(0).numpy()

    def update(self, x, u, x_next_true):
        x, u, x_next_true = np.asarray(x).flatten(), np.asarray(u).flatten(), np.asarray(x_next_true).flatten()
        self.X_train.append(np.hstack([x, u]))
        self.y_train.append(x_next_true)
        if len(self.X_train) % 50 == 0:
            X = torch.tensor(np.array(self.X_train), dtype=torch.float32)
            y = torch.tensor(np.array(self.y_train), dtype=torch.float32)
            for _ in range(10):
                self.optimizer.zero_grad()
                loss = self.loss_fn(self(X), y)
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.item())

    def get_jacobians(self, x, u):
        xu = torch.tensor(np.hstack([x, u]), dtype=torch.float32, requires_grad=True)
        output = self.net(xu.unsqueeze(0)).squeeze(0)
        jacobian = torch.zeros(4, 5)
        for i in range(4):
            grad = torch.autograd.grad(output[i], xu, retain_graph=True)[0]
            jacobian[i] = grad
        A = jacobian[:, :4].detach().numpy()
        B = jacobian[:, 4:].detach().numpy()
        
        return A, B
    
def main():
    # Initialize with rendering
    env = gym.make('CartPoleSwingUp-v0', render_mode='human')  # Changed to render
    
    dynamics_model = DynamicsModel()
    controller = iLQRController(dynamics_model, horizon=30, max_iter=10, exploration_noise=0.2)
    
    # Setup plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    obs, _ = env.reset()
    u_guess = np.zeros((controller.horizon, 1))

    # Exploration phase
    for _ in range(100):
        action = np.random.uniform(-1, 1, size=(1,))
        next_obs, _, _, _, _ = env.step(action)
        dynamics_model.update(obs, action, next_obs)
        obs = next_obs

    # Training phase
    for step in range(1000):
        x0 = np.array(obs, dtype=np.float32).flatten()
        u_opt = controller.optimize(x0, u_guess)
        action = np.clip(u_opt[0], -1, 1)
        next_obs, _, terminated, truncated, _ = env.step([action])
        dynamics_model.update(obs, action, next_obs)
        obs = next_obs
        u_guess[:-1] = u_opt[1:]
        u_guess[-1] = action
        
        # Update plots
        if step % 10 == 0:
            ax1.clear()
            ax2.clear()
            
            # Plot training loss
            ax1.plot(dynamics_model.losses)
            ax1.set_title("Dynamics Model Training Loss")
            ax1.set_ylabel("Loss")
            
            # Plot position error
            ax2.plot(controller.position_errors)
            ax2.set_title("Cart Position Error (Distance from Center)")
            ax2.set_xlabel("Steps")
            ax2.set_ylabel("Position Error")
            
            plt.pause(0.01)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()