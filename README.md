# Lab 4 - Reinforcement Learning  

## Temporal-Difference Learning in Cliff Walking (CliffWalking-v1)

## Overview

This project implements **Temporal-Difference (TD) control algorithms** in the Gymnasium Cliff Walking environment to compare **on-policy and off-policy learning**.

### This project contains:

#### 1. TD Control Algorithms (CliffWalking-v1)
- Implements:
  - **SARSA (on-policy TD control)**
  - **Q-learning (off-policy TD control)**
- Learns action-value function \( Q(s,a) \) using **online updates**
- Uses **ε-greedy policy** for exploration
- Trains across **multiple random seeds (≥30)**
- Experiments with:
  - Different step-sizes \(α)
  - Fixed and decaying ε schedules

---

#### 2. Evaluation & Visualization
- **Learning curves** with 95% confidence intervals
- **Policy visualization** (grid arrows)
- **Value function heatmaps**
- **Sample trajectories** from learned policies
- Sensitivity analysis for:
  - Step-size (α)
  - Exploration rate \(ε)

---

## Environment

- **Environment:** CliffWalking-v1 (Gymnasium)
- **Grid Size:** 4 × 12

### State Space:
- Discrete (48 states)
- Each state represents agent position:
 $s = row \times 12 + col$
### Action Space:
- Discrete (4 actions):
  - 0 → Up  
  - 1 → Right  
  - 2 → Down  
  - 3 → Left  

### Rewards:
- -1 → Each step  
- -100 → Falling into cliff (agent resets to start)  
- 0 → Goal reached  

### Episode Termination:
- Episode ends when agent reaches the goal state

---

## Algorithms

### 1. SARSA (On-Policy TD Control)
- Updates using:
  $Q(s,a) \leftarrow Q(s,a) + α [r + \gamma Q(s',a') - Q(s,a)]$
- Learns value of the **behavior policy (ε-greedy)**
- Produces **safer policies** away from the cliff

---

### 2. Q-Learning (Off-Policy TD Control)
- Updates using:
  $Q(s,a) \leftarrow Q(s,a) + α [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
- Learns value of the **optimal greedy policy**
- Produces **optimal but riskier paths**

---

## Experimental Setup

- Episodes per run: **500**
- Number of seeds: **30**
- Discount factor: **γ = 0.99**
- Exploration:
  - Fixed ε = 0.1
  - Optional ε-decay
- Step-size values:
  - $α \in {0.1, 0.3, 0.5, 0.9}$

### Evaluation Metrics:
- Sum of rewards per episode
- Mean performance across seeds
- 95% confidence intervals

---

## Key Results

- **SARSA learns safer paths** away from the cliff due to on-policy updates
- **Q-learning converges faster** to the optimal path but is riskier during learning
- Step-size \(α) significantly affects:
  - Convergence speed
  - Stability of learning
- Medium α values (0.3–0.5) provide the best tradeoff

---

## Key Insights

- On-policy vs off-policy learning leads to **different behaviors in risky environments**
- Exploration strategy directly impacts learned policy
- TD learning is sensitive to hyperparameters (α, ε)
- Confirms theoretical results from Sutton & Barto:
  - SARSA → risk-aware learning  
  - Q-learning → optimal but optimistic learning  

---

## Setup

### Install dependencies:
```python
%pip install numpy matplotlib gymnasium
