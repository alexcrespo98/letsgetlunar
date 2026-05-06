# Project 3 — Reinforcement Learning for Lunar Orbital Insertion
**Alex Crespo | 2026**

## Assignment Overview

The objective of this project was to design an intelligent agent capable of controlling spacecraft attitude for orbital insertion around the moon. The target orbit was a circular orbit at 400 km altitude with near-zero radial velocity and a tangential velocity matching the circular orbit speed (~1,633 m/s). The agent was trained using reinforcement learning, replacing the fixed optimal control approach used in project 2.

## Carryover from Project 2

Project 2 produced a working beta (thrust angle) profile using direct transcription and nonlinear programming. That profile was not a clean smooth signal. It contained discontinuous jumps between +90° and −90°, which the professor identified as an NLP numerical stability artifact rather than physically meaningful behavior. That result became relevant again in project 3 and is discussed in the hail mary section below.

## Environment

The physics environment was implemented from scratch in Python using the Gymnasium interface. The spacecraft dynamics follow a polar coordinate equations of motion model with RK4 integration at 0.05 second timesteps. Each episode runs up to 160 seconds. State observations fed to the agent are altitude error, radial velocity, and tangential velocity, each normalized. The action is the thrust angle beta, rate-limited to prevent discontinuous jumps, which directly addresses the professor's feedback from project 2. An episode terminates on success (within tolerance of target altitude and velocities), surface impact, escape above 600 km, or timeout.

## Reward Functions Tried

Three reward functions were implemented and tested as separate experiments.

**Experiment A — Sparse reward (PPO):** The agent only received a signal at episode termination: +1000 for success, −500 for crash, −100 for timeout. This gave the agent almost no gradient signal to learn from. It never meaningfully improved.

**Experiment B — Shaped reward (PPO):** A continuous signal was given at every step based on quadratic penalties on altitude error, radial velocity error, and tangential velocity error, plus a small penalty on control rate of change. Bonus and penalty terms were applied at termination. This performed better than sparse but plateaued early.

**Experiment C — Multi-objective Gaussian reward (SAC):** Each error term was transformed through a Gaussian kernel so that reward scaled smoothly from 0 to 1 as the spacecraft approached the target. Weights of 0.4 for altitude, 0.3 for radial velocity, and 0.3 for tangential velocity were applied. A small time penalty of 0.001 per step encouraged efficiency. Terminal bonuses and penalties were preserved. This formulation gave the agent a much richer gradient signal across the full state space. The best results came from experiment C and all further development focused there.

[Insert Experiment A, B, and C reward curves here]

The graphs above show the mean evaluation reward over training timesteps for all three experiments. Experiment A remains flat near zero throughout training, reflecting the inability of the sparse signal to guide learning. Experiment B climbs initially but stalls well below the success threshold. Experiment C shows consistent improvement over time and reaches the highest reward of the three, which is why all subsequent fine-tuning and optimization work was focused on this configuration.

## Agent Architectures

For experiments A and B, PPO was used with default network sizes. PPO is an on-policy algorithm and requires collecting fresh experience at every update, which is compute-inefficient for a problem this slow to simulate.

For experiment C, SAC (Soft Actor-Critic) was used. SAC is off-policy and samples from a replay buffer, making far better use of every simulated transition. It also includes automatic entropy tuning, which balances exploration and exploitation without manual tuning. The base architecture used two hidden layers of 128 neurons each. A wider 256x256 architecture was tested in the hail mary run described below.

## Training and Optimization Attempts

**Solo training** ran experiments A, B, and C sequentially on the main Windows machine. After about 16 hours, experiment C reached a reward of approximately 810.

**Collab mode** was built to run two machines simultaneously. The main Windows PC and a salvaged MacBook Air running Ubuntu were networked together over a TCP socket on port 7777. Both machines maintained a shared pool of the best experiment C models, automatically transferring weights between machines and fine-tuning whichever model had the highest reward. This ran unattended overnight for about 20 hours and pushed the best reward to approximately 912. The reward then plateaued and stopped improving.

**Parallel multicore sweep** was attempted on the Windows machine, which has 12 physical cores. The idea was to spawn 8 workers simultaneously with different random seeds and hyperparameter configurations, letting each train independently and comparing results. In practice, spawning that many SAC instances simultaneously caused memory exhaustion and repeated crashes. Each SAC instance uses several gigabytes of RAM for the replay buffer, and 8 simultaneous instances exceeded available memory. Runtime was more than twice as slow as serial training, not faster. The sweep was scaled back to 2 to 3 configs at a time.

**Hyperparameter sweep** tested 8 configurations bracketing the experiment C baseline, varying learning rate, network width, replay buffer size, batch size, and entropy coefficient. The best-performing configuration used a slightly elevated learning rate of 5e-4 with the same 128x128 architecture.

[Insert hyperparameter sweep results table or bar chart here]

The graph above summarizes the best reward achieved by each sweep configuration. The baseline config and the slightly elevated learning rate config performed best, which informed the decision to keep the learning rate at or near 3e-4 to 5e-4 for subsequent runs.

**Hail mary warm start** was the final approach. After a week of training with the reward stuck near 850 to 912 despite many fine-tune runs, the project 2 beta profile was used as a behavioral cloning seed. The profile was first smoothed to remove the discontinuous jumps the professor identified, using a neighbor-to-neighbor constraint that replaces any waypoint differing from its predecessor by more than 45 degrees with the average of its neighbors, applied in two passes. Twenty-five demo episodes were simulated using this smoothed profile as the action source, and all resulting transitions were injected directly into the SAC replay buffer before training began. The best existing trained model at a reward of approximately 912 was also loaded to initialize the network weights. This gave the agent both a pre-trained policy and a replay buffer seeded with physically informed trajectory examples. The hail mary run reached a reward of 732 at the halfway point of its training budget, consistent with exceeding the previous plateau if training runs to completion.

[Insert hail mary training curve here]

The graph above shows the hail mary reward curve. The agent begins at a much higher reward than any cold-start run, which reflects both the pre-loaded network weights and the beta profile demo data in the replay buffer. The steep early climb confirms the warm start was effective at skipping the slow initial learning phase that consumed most of the compute in earlier experiments.

## Supporting Infrastructure

Several supporting tools were built to manage the long training runs. A live terminal monitor auto-launched during training and printed reward curves, step counts, and buffer fill level every 30 seconds. A GUI monitor plotted the evaluation reward curve in a persistent window. An attempt log saved metadata about every run including hours trained, best reward, abort reason, and notes. Model filenames encoded the best reward directly so performance was visible without opening logs. These tools were necessary because most runs lasted 8 to 24 hours unattended.

## What Could Be Tried with More Time

The agent never fully converged to consistent success. With more compute, the most promising next steps would be curriculum learning (starting with an easier version of the problem and gradually tightening tolerances), a prioritized replay buffer that oversamples the rare near-success transitions, and adding the current beta angle as an observation so the agent is aware of its own control history. Including a smoothness penalty on beta rate of change in the reward function more aggressively would also help, consistent with the professor's recommendation to constrain nearest-neighbor control point deviation.

## Summary of Key Observations

Reward shaping had the single biggest impact on learning speed. Moving from sparse to shaped to Gaussian multi-objective reward dramatically improved gradient signal quality. SAC outperformed PPO significantly for this problem due to replay buffer efficiency. The agent learned faster when given more informative observations and smoother reward surfaces. Collab mode was the most effective compute multiplier that actually worked reliably. Parallel multicore training crashed repeatedly due to memory pressure. The project 2 beta profile, once smoothed, provided genuinely useful warm start data that helped the agent skip the earliest and slowest phase of learning.
