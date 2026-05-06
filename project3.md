# Project 3: Reinforcement Learning for Lunar Orbital Insertion
**Alex Crespo | 2026**

---

## Assignment Overview

The goal of this project was to train an agent to control a spacecraft's thrust angle and achieve
orbital insertion around the moon. The target was a circular orbit at 400 km altitude with near-zero
radial velocity and a tangential velocity of approximately 1,512 m/s (computed from lunar surface
gravity of 1.62 m/s^2 and a lunar radius of 1,734.7 km). This replaced the fixed optimal control
approach from Project 2 with a reinforcement learning agent that figures out the control profile on
its own through trial and error.

---

## Carryover from Project 2

Project 2 solved for the optimal thrust angle profile using direct transcription with 60 nodes and
scipy's trust-constr optimizer. The resulting profile had discontinuous jumps between +90 and -90
degrees, which the professor identified as a numerical artifact rather than physically meaningful
behavior. That profile became useful again in Project 3 as a warm-start seed, described below.

---

## Environment

The simulation was written from scratch using the Gymnasium interface. The spacecraft dynamics use a
polar coordinate equations-of-motion model integrated with RK4 at 0.05 second timesteps. Each
episode runs up to 160 seconds (3,200 steps). The agent observes normalized altitude error, radial
velocity, and tangential velocity. It outputs a thrust angle (beta), which is rate-limited so it
cannot jump discontinuously between steps. An episode ends when the spacecraft hits the success
tolerances (altitude within 20 km of 400 km, radial velocity under 50 m/s, tangential velocity
within 100 m/s of 1,512 m/s), crashes, escapes above 600 km, or runs out of time.

---

## Reward Functions Tried

Three reward functions were tested as separate experiments.

**Experiment A: Sparse reward (PPO).** The agent only got feedback at the very end of an episode:
+1000 for success, -500 for crashing, -100 for timeout. With 3,200 steps and no signal in between,
the agent had almost nothing to learn from. It never improved. (Figure 1)

**Experiment B: Shaped reward (PPO).** A signal was given at every step based on how far the
spacecraft was from the target altitude and velocities, plus a small penalty for changing the thrust
angle quickly. This worked better than sparse but plateaued early. (Figure 2)

**Experiment C: Gaussian reward (SAC).** Each error term was passed through a Gaussian function so
the per-step reward scaled smoothly from 0 to 1 as the spacecraft approached the target. Weights
were 0.4 for altitude, 0.3 for radial velocity, and 0.3 for tangential velocity. A small time
penalty encouraged the agent to finish faster. A +1000 bonus was added on success and penalties
applied for crash or escape. This gave the agent a much richer learning signal and produced the best
results. All further training focused on Experiment C. (Figure 3)

---

## Agent Architectures

For Experiments A and B, PPO was used. PPO requires generating fresh experience before every
update, which is slow for a problem where each simulation step is expensive.

For Experiment C, SAC (Soft Actor-Critic) was used instead. SAC stores past experience in a replay
buffer and learns from it repeatedly, which makes much better use of every simulated step. It also
automatically balances exploration and exploitation. The network used two hidden layers of 128
neurons each.

---

## Training and Optimization Attempts

**Solo training** ran all three experiments sequentially on the main Windows machine. After about
16 hours, Experiment C reached a cumulative reward of approximately 810.

**Collab mode** networked the main Windows PC and a salvaged MacBook Air running Ubuntu over a TCP
socket on port 7777. Both machines shared a pool of the best models and each kept fine-tuning
whichever model had the highest reward, transferring weights back and forth automatically. This ran
overnight for about 20 hours and pushed the best reward to approximately 912, then it plateaued.

**Parallel multicore sweep** tried to run 8 training workers simultaneously on the 12-core Windows
machine. In practice each SAC instance uses several gigabytes of RAM for its replay buffer, so 8 at
once exceeded available memory and caused repeated crashes. The sweep was scaled back to 2 to 3
configs at a time.

**Hyperparameter sweep** tested 8 configurations varying learning rate, network size, replay buffer
size, batch size, and entropy coefficient. No configuration beat the baseline by a meaningful margin.
The main takeaway was that the original setup was fine and more training time was the bottleneck,
not the hyperparameters.

**Hail mary warm start** was the final attempt. After more than 100 hours of cumulative training
with the best reward stuck at 912, the Project 2 beta profile was used to seed the agent. The
profile was first smoothed to remove the discontinuous jumps: any waypoint differing from its
neighbor by more than 45 degrees was replaced with the average of its two neighbors, applied in two
passes. Twenty-five episodes were simulated using this smoothed profile and all the resulting
transitions were loaded directly into the SAC replay buffer before training began. The best existing
model (reward 912) was also used to initialize the network weights. At the halfway point of this
run the reward was 732, which is expected when a model starts integrating new experience from a
different source. This run is continuing through the submission deadline.

---

## Supporting Infrastructure

A live terminal monitor printed reward curves and step counts every 30 seconds. A GUI monitor kept
a persistent reward curve plot open. An attempt log tracked every run's hours, best reward, and
abort reason. Model filenames included the best reward number directly so there was no need to open
logs to compare models. These tools were necessary because most runs lasted 8 to 24 hours unattended.

---

## Results and What I Would Do Differently

After more than 100 hours of training, the agent never triggered the success condition. The success
criteria require altitude within 20 km of 400 km, radial velocity under 50 m/s, and tangential
velocity within 100 m/s of 1,512 m/s -- all three at the same time, within 160 seconds. The best
reward of 912 reflects accumulated per-step progress across a full 3,200-step episode, not a score
out of some maximum. The agent learned to steer toward the target and avoid crashing, but could not
close all three error terms simultaneously before time ran out.

This is a real limitation but not a fundamental one. The reward shaping approach worked -- each
experiment was a clear improvement over the last. The sweep confirmed the setup was not the problem.
The bottleneck was compute time and the difficulty of the final convergence step where all three
tolerances have to be met at once.

Given more time, the three changes most likely to help would be starting the agent on an easier
version of the problem and gradually tightening the tolerances, sampling near-success episodes more
often during training since they are rare and informative, and adding the current thrust angle as an
input so the agent can see its own control state directly.

---

## Summary

Reward shaping made the biggest difference. Going from sparse to shaped to Gaussian reward at each
step gave the agent progressively better feedback to learn from. SAC was a better fit than PPO for
this problem because it reuses past experience instead of discarding it. Collab mode across two
machines was the most effective way to get more training done. The parallel sweep crashed due to
memory limits and was not useful. The hyperparameter sweep confirmed the baseline design was sound.
The Project 2 beta profile, once smoothed, provided a useful starting point for the final training
run. After 100+ hours, the agent made clear progress but did not achieve consistent orbital
insertion. The hail mary run is still going.
