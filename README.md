# Reinforcement Learning on MinAtar Breakout

Reinforcement Learning project implementing both classical RL and deep RL algorithms to learn how to play Breakout in the MinAtar environment.

This project was completed as part of a Reinforcement Learning programming assignment. The objective is to analyze the environment, implement reinforcement learning algorithms from scratch, and evaluate their performance.  ￼

# Overview

MinAtar is a simplified version of Atari environments designed for reinforcement learning research.
Observations consist of 10×10 grids with multiple channels, allowing experiments to run much faster than full Atari environments while keeping the core challenges of RL problems.  ￼

This project includes:
	•	Environment analysis
	•	Baseline policy evaluation
	•	Implementation of Tabular Q-Learning
	•	Implementation of Double Deep Q-Network (Double DQN)
	•	Policy interpretation
	•	Performance improvement techniques
	•	Comparison between classical and deep RL methods

# Environment

## Environment used:

MinAtar/Breakout-v1

## Observation space:

(10, 10, 4)

The four channels represent:
	1.	Paddle position
	2.	Ball position
	3.	Ball trail (used to estimate velocity)
	4.	Brick layout

## Action space:

0 → no-op
1 → move left
2 → move right

## Reward structure:

+1 for each brick destroyed

Episodes terminate when the ball falls below the paddle.

# Baseline Policies

Two baseline policies were implemented for comparison.

## Random Policy

Selects actions uniformly at random.

Performance:

Reward ≈ 0.37 ± 0.64
Episode length ≈ 9.76 steps

## Heuristic Policy

Moves the paddle toward the horizontal position of the ball.

Performance:

Reward ≈ 8.18 ± 9.26
Episode length ≈ 95.86 steps

The heuristic policy demonstrates that simple domain knowledge can significantly improve performance.

# Classical Reinforcement Learning

## Tabular Q-Learning

A tabular Q-learning algorithm was implemented.

Update rule:

Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]


## State Representation

The raw observation space is high-dimensional.
To reduce complexity, the state was represented as:

(x_rel, y_ball, dx, dy)

Where:
	•	x_rel: relative horizontal distance between paddle and ball
	•	y_ball: vertical ball position
	•	dx, dy: ball velocity estimated using the trail channel

This compact representation allows the use of a tabular learning algorithm.

## Hyperparameters

Selected using grid search:

α = 0.5
γ = 0.9
ε_decay = 0.995

Exploration strategy:

ε-greedy

## Performance

Learned Policy (Q-learning)

Reward ≈ 4.21 ± 2.30
Episode length ≈ 51.38 steps

The learned policy improves significantly over the random baseline but still underperforms the heuristic policy.

Limitations include:
	•	sparse reward signal
	•	reduced state representation
	•	state aliasing

# Deep Reinforcement Learning

## Double Deep Q-Network (Double DQN)

A Double DQN was implemented to learn directly from raw observations.

Techniques used:
	•	replay buffer
	•	target network
	•	ε-greedy exploration
	•	Huber loss

Target computation:

a* = argmax_a Q_online(s',a)

y = r + γ Q_target(s', a*)

Loss function:

Huber Loss

## Neural Network Architecture

Input:

4 × 10 × 10 tensor

Architecture:

Conv2D (4 → 16, kernel 3x3)
ReLU
Conv2D (16 → 32, kernel 3x3)
ReLU
Flatten
Fully Connected (256)
ReLU
Output layer (Q-values for actions)

Convolutional layers allow the network to learn spatial relationships between the paddle, ball, and bricks.

# Policy Interpretation

To better understand the learned behavior, we analyzed the trained policies through:
	•	gameplay visualizations
	•	reward and episode length statistics
	•	comparison with baseline policies

Observations:
	•	The learned policy successfully keeps the ball in play longer than the random policy.
	•	The agent learns to position the paddle relative to the ball trajectory.
	•	However, the agent does not fully learn strategic targeting of bricks due to the sparse reward signal and limited training time.

These analyses help interpret what the agent has actually learned rather than relying solely on performance metrics.


# Improving Performance

Several techniques were explored to improve training performance and stability for Double DQN.

## Exploration Strategies

Beyond the baseline ε-greedy schedule, we tested alternative exploration setups to better handle sparse rewards:
	•	Longer exploration phase (slower ε decay) to improve state-space coverage and avoid premature convergence.
	•	Different ε schedules (same endpoints, different decay horizons) to balance early exploration and late exploitation.

Observation: keeping ε high for longer generally increases robustness, but may slow down short-term learning curves.

Ablation Studies

To understand which components contribute most to performance, we ran one-factor-at-a-time ablations (starting from a baseline configuration and modifying a single element):
	•	Learning rate: baseline vs lower learning rate
	•	Target network update period: baseline vs slower target updates
	•	Exploration decay steps: baseline vs longer exploration

Example baseline configuration:

baseline
lr = 1e-3
target_update = 1000
eps_decay_steps = 200000

Variants tested:
	•	lower learning rate
	•	slower target network updates
	•	longer exploration period

Gradient Clipping

To mitigate occasional instability during optimization, gradient norm clipping was considered (and can be enabled in the code):
	•	Clip gradients to a maximum norm (e.g., 10) before the optimizer step.

Effect: reduces large, destabilizing updates and helps prevent rare training collapses, especially when Q-values or TD-errors spike.

Observations

Key factors affecting performance:

Learning rate
	•	too large → unstable learning / divergence
	•	too small → slow convergence

Exploration schedule
	•	longer exploration improves coverage of the state space
	•	too fast decay may lead to premature convergence to suboptimal behavior

Target network updates
	•	slower updates improve stability but may slow learning
	•	more frequent updates can speed up learning but can destabilize training

Gradient clipping
	•	improves stability by preventing oversized gradient steps
	•	most useful when training exhibits sudden loss spikes

Results

Main observations:
	•	Tabular Q-learning improves over random play.
	•	Double DQN achieves significantly higher performance.
	•	Neural networks enable learning directly from raw observations.
	•	Exploration strategy, target network update frequency, and replay buffer all strongly affect training stability and final performance.


# Project Structure

.
├── EILLES_CHAN_WAY_Stephane.ipynb
├── README.md

# References

Watkins & Dayan (1992)
Q-learning

Mnih et al. (2015)
Human-level control through deep reinforcement learning

van Hasselt et al. (2016)
Deep Reinforcement Learning with Double Q-learning

MinAtar Environment
https://arxiv.org/abs/1903.03176
