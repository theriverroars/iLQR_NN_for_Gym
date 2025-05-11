# iLQR with Neural Network Dynamics Model

This repository contains an implementation of an iterative Linear Quadratic Regulator (iLQR) using a learned neural network dynamics model, applied to the classic reinforcement learning environment `MountainCarContinuous-v0` from OpenAI Gym.

## 🚗 Project Overview

The goal of this project is to explore model-based control using iLQR, where the system dynamics are learned using a neural network.
## 🧠 Key Features

- iLQR/MPC type controller
- Simple 2 layered MLP for dynamics model
- Evaluation on OpenAI Gym's `MountainCarContinuous-v0` environment.

## 📁 Repository Structure
- Main  branch has working code with iLQR with NN implemented on `MountainCarContinuous-v0` environment. The car is able to reach targetin about 120 steps on average.
- iLQR branch has implementation of iLQR with known dynamics. IT also has a simple LQR implementation.
- Swing_Up branch has code with attempt to implement iLQR with NN on Swing UP Cartpole problem. Changes mmade to implement this is listed in detail in seperate ReadMe.
