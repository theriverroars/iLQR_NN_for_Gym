# 🌀 iLQR with Neural Network Dynamics Model

This repository contains an implementation of **iterative Linear Quadratic Regulator (iLQR)** for model-based control, using a learned neural network dynamics model. The system is applied to the **MountainCar** task in a custom Gym environment.

> 🔬 This work was completed as part of course project for "Topics for AI" course at IISc, under the guidance of **Prof. Aditya Gopalan**.

---

## 🚀 Project Overview

The goal of this project is to explore **model-based reinforcement learning** by combining:
- A **neural network** to learn system dynamics from interaction data
- An **iLQR controller** to compute optimal control inputs based on the learned model

This framework allows control of systems with **unknown nonlinear dynamics**, without relying on analytical models.

---

## 🧠 Key Components

- ✅ **Neural Network Dynamics Model**  
  Trained using transition tuples \((s_t, a_t, s_{t+1})\), the NN predicts next state given current state and action. Implemented as a 2-layer MLP with ReLU activations.

- ✅ **iLQR Controller**  
  Uses a linearized model at each iteration to optimize a sequence of control inputs over a horizon. Backward and forward passes are used to compute control gains and update trajectories.

- ✅ **MountainCar Task**  
  The system successfully reaches te target destination using only learned dynamics.

---
## 🌱 Branches

- **`main`**:  
  Contains full working implementation of iLQR with NN for CartPole Swing-Up task in a singly .py file and the corresponding report.

- **`Swing_up`**:  
  Work in progress. Logs experimental variations, hyperparameter studies, and ablation results for a swing up cartpole environment.

---

## 👩‍💻 Contributors

- **Ganga Nair B** – M.Tech. Robotics, Indian Institute of Science  
- *Project under the guidance of Prof. Aditya Gopalan*

