# minatar-breakout-rl

Reinforcement Learning course project (programming assignment) on MinAtar: implementing and analyzing tabular RL and deep RL methods in a single notebook: EILLES_CHAN_WAY_Stephane.ipynb.  ￼

The notebook focuses on Breakout first, then tests transfer/generalization to other MinAtar games.  ￼

# What’s inside
- Environment analysis (state/action/reward/episode dynamics, stochasticity) for MinAtar/Breakout-v1  ￼
- Baselines: random policy + heuristic policy  ￼
- Classical RL: tabular Q-learning with a compact hand-crafted state  ￼
- Deep RL: Double DQN (CNN + replay buffer + target network + Huber loss) trained from raw observations  ￼
- Policy interpretation: videos + state–action heatmaps / controlled state slices  ￼
- Improving performance: alternative exploration, ablation studies, gradient clipping  ￼
- Generalization: direct transfer + retraining on Asterix, Freeway, Seaquest, Space Invaders  ￼


# Environment
- Gymnasium id: MinAtar/Breakout-v1  ￼
- Observation: (10, 10, 4) binary channels (paddle / ball / trail / bricks)  ￼
- Actions: 0 no-op, 1 left, 2 right  ￼
- Reward: +1 per brick destroyed (sparse rewards)  ￼
- Episode ends when the ball falls below the paddle  ￼
	

# Key results (Breakout)

Evaluation is typically reported as mean ± std over 1000 episodes.  ￼
| Method | Mean return | Mean episode length (steps) | Notes |
|---|---:|---:|---|
| Random policy | 0.37 ± 0.64 | 9.76 ± 6.62 | Very short episodes |
| Heuristic policy | 8.18 ± 9.26 | 95.86 ± 107.12 | “Track the ball” baseline |
| Tabular Q-learning (best grid-search params) | 4.21 ± 2.30 | 51.38 ± 25.06 | Better than random, below heuristic |
| Deep RL (best config, Breakout) | 18.66 ± 14.56 | 190.10 ± 152.75 | Best-performing setup in the notebook |￼

# Methods

## Tabular Q-learning (classical RL)

Because raw observations are high-dimensional, the notebook uses a compact state:
- x_rel: paddle x − ball x
- y_ball: ball y
- (dx, dy): estimated ball velocity using the trail channel  ￼

Hyperparameters selected via grid search:
- alpha = 0.5, gamma = 0.9, epsilon_decay = 0.995  ￼

## Double DQN (deep RL)

Deep agent uses the raw observation (transposed to 4×10×10) and trains a CNN-based Double DQN:
- Replay buffer + target network for stability
- Double DQN target (online selects action, target evaluates it)
- Huber loss (smooth_l1_loss)  ￼

CNN architecture (small, because inputs are tiny):
- Conv(4→16, 3×3) + ReLU
- Conv(16→32, 3×3) + ReLU
- FC(32·10·10 → 256) + ReLU
- FC(256 → n_actions)  ￼

# Policy interpretation

The notebook goes beyond raw scores by interpreting the learned policy using:
- Gameplay videos
- State–action heatmaps over controlled state slices

Main qualitative finding: the learned policy’s decisions depend strongly on ball motion (e.g., more intercept-like behavior when the ball is descending), and it differs from the heuristic’s pure “track the ball” strategy.  ￼

# Improving performance

This section evaluates three levers: alternative exploration, ablation studies, and gradient clipping.  ￼

## Alternative exploration

Compared to standard ε-greedy, the notebook tests:
- Boltzmann (softmax) exploration with a temperature schedule
- Sticky ε-greedy (repeat previous action with some probability)

Boltzmann yields a small improvement over ε-greedy in this setup (evaluation 18.72 ± 14.65 vs 17.79 ± 13.67 in one experiment), while sticky ε-greedy is roughly comparable (17.19 ± 12.07).  ￼

## Ablation studies (what matters most?)

Removing core DQN stabilizers shows:
- No target network: performance drops (13.33 ± 9.05)  ￼
- No replay buffer: learning collapses (0.01 ± 0.10, ~6 steps)  ￼
- No Double DQN (standard DQN): slightly lower / similar (16.16 ± 13.39)  ￼

## Gradient clipping

Gradient clipping (norm clipping) slightly reduces final return in the reported run (16.56 ± 11.51 vs 17.79 ± 13.67) but can improve robustness by preventing occasional unstable updates.  ￼

# Extending to other MinAtar games (generalization)

The notebook evaluates generalization on:
- MinAtar/Asterix-v1
- MinAtar/Freeway-v1
- MinAtar/Seaquest-v1
- MinAtar/SpaceInvaders-v1  ￼

## 1) Direct transfer (no retraining)

The best Breakout agent is applied as-is to other games. Example: on Asterix, direct transfer does not improve reward over random (both mean return 0.49), although it survives longer (≈ 99 vs 65 steps).  ￼

Overall conclusion reported in the notebook: direct transfer helps on some games (e.g., Freeway) but not on others (e.g., Seaquest, Space Invaders).  ￼

## 2) Retraining on other games

Using the same deep RL pipeline (with minor adjustments when observation channels differ), retraining results include:

| Game | Trained agent return | Random return | Notes |
|---|---:|---:|---|
| Freeway | 18.90 ± 12.55 | 0.36 ± 0.56 | Large gap over random |
| Seaquest | 0.33 ± 0.60 | 0.11 ± 0.40 | Improves, but remains low |
| Space Invaders | 29.95 ± 20.38 | 4.02 ± 3.24 | Strong improvement |
| Asterix | 0.53 ± 0.88 | 0.49 ± 0.85 | Only marginal gain |

High-level takeaway: games with simpler control geometry (often movement mainly along one axis) and denser/more attributable rewards are easier to learn; richer dynamics and harder credit assignment make learning tougher (e.g., Asterix).  ￼

# Installation

pip install numpy gymnasium minatar torch matplotlib

# How to run

Open and run the notebook end-to-end:
- EILLES_CHAN_WAY_Stephane.ipynb  ￼


# Repository structure

```text
.
├── EILLES_CHAN_WAY_Stephane.ipynb
└── README.md
```

# References

The notebook cites the key references:
- Watkins & Dayan — Q-learning (1992)  ￼
- Mnih et al. — DQN (2015)  ￼
- van Hasselt et al. — Double DQN (2016)  ￼

MinAtar paper: https://arxiv.org/abs/1903.03176

DQN paper:     https://www.nature.com/articles/nature14236

Double DQN:    https://arxiv.org/abs/1509.06461

# Author

Stéphane EILLES-CHAN WAY  ￼
