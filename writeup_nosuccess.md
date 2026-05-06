# Project 3 — Reinforcement Learning for Lunar Orbital Insertion

## Overview and Problem Statement

The goal of this project is to train a reinforcement learning agent to autonomously control a spacecraft's thrust angle β(t) and insert it into a 400 km circular lunar orbit starting from rest on the lunar surface. The agent has no prior knowledge of orbital mechanics — it learns purely through trial and error, receiving reward signals based on how close it gets to the target orbit. Success is defined by three simultaneous conditions: altitude within ±20 km of the 400 km target, radial velocity Vr below 50 m/s, and tangential velocity Vtan within 100 m/s of the circular orbit speed of 1511.17 m/s.

This is a hard continuous control problem. The thrust angle β(t) must be swept precisely over the course of a roughly 160-second burn, transitioning from nearly vertical (to gain altitude) to nearly horizontal (to build orbital speed), all without overshooting the target orbit. The challenge is designing a reward function and learning algorithm that can discover this maneuver purely from reward signal, without access to any physics equations or reference trajectory.

Project 2 established a baseline using a numerical optimizer (scipy trust-constr) to find the optimal bang-bang β(t) profile. That NLP solution is used here as a sanity check — if the RL agent is actually learning something meaningful, its β(t) profile should qualitatively match: starting near 90°, then sweeping toward 0° as the spacecraft approaches orbital altitude and speed.

## Environment and Dynamics

The simulation environment is a custom Gymnasium environment in `scripts/lunar_env.py`. The Moon is modeled as a sphere with radius 1734.7 km and surface gravity 1.62 m/s². The spacecraft has a thrust of 90,000 N and a mass of 300 kg, giving a thrust-to-mass ratio of 300 m/s². These numbers keep the problem tractable — a real lunar ascent vehicle would have a much lower T/M ratio and a much longer burn time, but the physics are representative.

The state is a four-vector: radial distance from the Moon's center, angular position, radial velocity, and tangential velocity. The agent observes a normalized three-vector derived from this state: altitude error scaled by 0.0022, radial velocity scaled by 4×10⁻⁵, and tangential velocity scaled by 4×10⁻⁵. The action is the commanded thrust angle β in radians, clamped to [-π/2, π/2] and rate-limited to 2000 degrees per second so the agent cannot make instantaneous discontinuous jumps.

Each episode runs for up to 160 simulated seconds at a timestep of 0.05 seconds, giving a maximum of 3200 steps. The starting condition for the main experiments is fixed at the lunar surface with zero initial velocity, though Experiment C adds exploring starts to help the agent generalize beyond a single launch condition. The episode terminates early if the spacecraft crashes into the surface, escapes to a radius more than 10,000 km above the target orbit, or satisfies all three success tolerances simultaneously.

## Experiments

### Experiment A — PPO with Sparse Reward

The first experiment used Proximal Policy Optimization with a completely sparse reward: zero during the episode, +1000 on success, -500 for crashing, and -100 for timeout. The network was a small 64×64 MLP. Training ran for 500,000 steps.

The result was completely negative learning, which was expected going in. Sparse reward gives the agent no gradient signal during the episode — it only learns whether the entire 160-second maneuver succeeded or failed, with no intermediate feedback about whether it was heading in the right direction. From a fixed launch condition, the probability of accidentally stumbling into a correct orbit insertion purely through random exploration in 500,000 steps is essentially zero. The reward bounced between -100 (timeout) and -500 (crash) for the entire run, never once triggering the +1000 bonus. There was no meaningful learning curve.

This experiment serves an important purpose in the writeup: it establishes a clear baseline showing that sparse reward does not work for hard continuous control problems without curriculum learning or a good initialization. The training curve (rl_a_training.png in `output/`) is a flat line at negative values the entire run.

### Experiment B — PPO with Shaped Reward

Experiment B kept PPO but replaced the sparse reward with a dense shaped reward: quadratic penalties on altitude error, radial velocity error, and tangential velocity error, plus a control effort penalty. All penalty terms are negative, so the reward is bounded above by zero. On success the agent receives a +500 terminal bonus, -200 for crashing, and -100 for escaping. The network was a larger 256×128×64 MLP, and training ran for 2,000,000 steps.

The reward climbed from roughly -80,000 at the start to nearly 0 by around 750,000 steps, then flatlined. This plateau at 0 is not a fluke — it is the mathematical ceiling of the shaped reward function. Once PPO found a policy that keeps altitude error, Vr, and Vtan all small throughout the episode, every penalty term was near zero and there was no gradient signal left to follow. The agent learned to hover in the neighborhood of the target, but never precisely tripped all three success tolerances simultaneously to earn the +500 terminal bonus.

A key design flaw contributed to this plateau: the quadratic denominators were set to the full problem scale (400 km for altitude, 1000 m/s for velocity), rather than to the success tolerances (20 km, 50 m/s, 100 m/s). This means that when the altitude error is, say, 10 km — well within success range — the altitude penalty is -(10/400)² × 0.5 ≈ 0.0003, essentially zero. The reward landscape goes completely flat near the target, giving PPO nothing to differentiate between "close but not quite" and "actually inside the tolerance box." The agent has no incentive to push through that last 5% of the problem. The training curve (rl_b_training.png in `output/`) confirms this — a steep climb followed by a flat ceiling with zero recorded successes.

### Experiment C — SAC with Multiobjective Gaussian Reward

Experiment C is the main attempt. It switches from PPO to Soft Actor-Critic and introduces a fundamentally different reward structure: a weighted sum of Gaussian "bells" centered on the target values for altitude, Vr, and Vtan. The weights are (0.4, 0.3, 0.3) and the Gaussian widths are (50 km, 200 m/s, 200 m/s). A small control effort penalty (-0.10 × (Δβ/90)²) and a time penalty (-0.001 per step) encourage efficient trajectories. On success the agent earns a +1000 bonus, -500 for crashing, -200 for escaping.

The Gaussian reward structure is a deliberate improvement over the quadratic penalties in Experiment B. Unlike a quadratic that goes flat near zero, a Gaussian is maximally steep at the target and provides useful gradient everywhere in state space — the further away the agent is, the more reward it stands to gain by moving toward the target. There is no ceiling effect and no flat region near the success box.

SAC is also a better fit for this problem than PPO. SAC is off-policy and maintains a replay buffer of 500,000 transitions, meaning it can learn from past experience multiple times rather than throwing data away after each update. It uses entropy regularization to encourage exploration automatically, balancing exploitation and exploration without manual tuning. For continuous action spaces, SAC consistently outperforms PPO in sample efficiency.

Exploring starts were enabled for Experiment C: the initial altitude is randomized between 0 and 50 km above the surface, and initial velocities are sampled near orbital values. This forces the agent to learn a general insertion policy rather than memorizing a single fixed trajectory, and it dramatically reduces the chance of overfitting to the specific fixed launch condition. Training ran for 2,000,000 steps with a network size of 128×128, learning rate 3×10⁻⁴, batch size 256, and automatic entropy coefficient tuning.

Beyond the base SAC run, additional infrastructure was built during this project: a hyperparameter sweep across eight configurations bracketing the baseline parameters (learning rate, network size, buffer size, batch size, entropy coefficient, and Gaussian altitude width), two-machine collaborative training over a TCP socket between a Windows PC and an Ubuntu machine with model weights transferred as base64-encoded zip files, a fine-tune option to warm-start from the best saved model, and a live GUI monitor showing reward curves updating in real time.

## Results

Experiment C made substantial progress but did not achieve consistent, confirmed orbit insertion by the end of the training budget. The best evaluation reward reached approximately 850, which is well above the Gaussian reward baseline (maximum of around 1.0 per step from the bell terms alone) and reflects an agent that has learned to climb toward the target orbit and circularize its trajectory. However, the reward never cleanly crossed the +1000 success threshold in a way that triggered the `_success` suffix on a saved model — the agent repeatedly gets close to all three tolerances but does not reliably satisfy all three simultaneously within a single episode.

Looking at the trajectory plots in `output/`, the partial progress is visible. The rl_c_polar.png figure shows the spacecraft successfully climbing off the lunar surface and approaching the 400 km altitude band, but the trajectory ends with some residual radial velocity rather than a clean circular insertion. The rl_c_states.png altitude and velocity histories show the agent learning the correct qualitative shape of the maneuver — altitude increases monotonically, tangential velocity builds up toward orbital speed — but the final few seconds of the episode show it still slightly outside at least one of the three tolerance windows. The rl_c_training.png training curve shows reward climbing from negative values in the early phase, through the 0–1 range as the Gaussian bells engage, and eventually plateauing around 850 after approximately 1.4 million steps with no further meaningful improvement.

The β(t) profile (rl_c_beta.png in `output/`) is nonetheless interesting. The learned thrust angle profile starts near 90° on the surface and sweeps progressively toward 0° through the burn, qualitatively matching the bang-bang solution from the Project 2 NLP optimizer stored in output/optimal_beta_profile.csv. This is a strong signal that the agent has learned the correct physical intuition — it knows it needs to go up first and then go sideways — even if it has not yet converged to the precise angle schedule that satisfies all three tolerances at cutoff. The rate-limited actuator in the environment means the RL profile is smoother than the NLP bang-bang solution; the agent compensates by spreading the sweep more gradually.

## Discussion and Takeaways

The three experiments tell a clean story about reward function design for continuous control.

Sparse reward (Experiment A) is simply not viable for this problem without better initialization. The agent needs to stumble into a correct insertion by pure chance before it can learn anything, which essentially never happens in 500,000 steps from a fixed launch point. The lesson is that hard continuous control problems need dense reward — the agent has to receive feedback at every step, not just at success or failure.

Shaped quadratic reward (Experiment B) is a significant improvement but has a subtle and important flaw: the gradient goes flat exactly where you need it most. Setting the denominator to the full problem scale (400 km) rather than the tolerance scale (20 km) means the reward function is completely insensitive when the agent is actually close to success. Fixing this would require matching the denominators to the success tolerances, which would make the reward gradient steep near the target and give PPO something to chase. This is why Experiment C uses Gaussian kernels instead — the gradient is correct everywhere by design.

SAC outperformed PPO for two reasons beyond just the reward function. First, off-policy learning from a replay buffer is dramatically more sample efficient for continuous action spaces — SAC can squeeze 500,000 steps of useful gradient out of a buffer of past experience, while PPO throws data away after each update. Second, automatic entropy tuning means the agent maintains useful exploration throughout training without manual schedule tuning.

The exploring starts in Experiment C were also important. A fixed launch condition makes it very easy for the agent to overfit — it memorizes a single trajectory rather than learning a general control law. Randomizing the initial altitude and velocity forces it to solve the actual problem: a feedback control policy that can handle a range of initial conditions. The plateau around 850 suggests the agent has a good general policy but is falling short of the precision needed to trip all three tolerances simultaneously.

The most likely explanation for the plateau is that the Gaussian widths (50 km, 200 m/s, 200 m/s) are significantly wider than the success tolerances (20 km, 50 m/s, 100 m/s). The reward gradient is still shallow in the final kilometers and final tens of m/s before success, not as flat as Experiment B's quadratic denominators, but not steep enough to reliably pull the agent into the exact box. Tightening the Gaussian widths to match the tolerances more closely — say, 20 km, 50 m/s, 100 m/s — would create a steeper reward gradient right at the boundary of the success condition and likely push the agent through.

## Conclusion

Experiment C did not achieve confirmed orbit insertion, but it came meaningfully close. The SAC agent learned the correct qualitative structure of the lunar insertion maneuver — a thrust sweep from vertical to horizontal, building tangential velocity while gaining altitude — and its β(t) profile qualitatively matches the Project 2 NLP solution. The best evaluation reward of approximately 850 reflects an agent that has learned something physically real, not a policy that happens to score well through reward hacking.

What would push it over the line? A few options, in rough order of expected impact: tighter Gaussian reward widths matched to the success tolerances rather than the full problem scale, more training steps (the reward was still not fully flat at 2 million steps), curriculum learning starting from near-orbit initial conditions and gradually pulling the start point back toward the surface, and seeding the SAC replay buffer with transitions from the NLP bang-bang trajectory to give the agent a warm start. Any one of these changes would likely be sufficient; all together would almost certainly close the gap.

For a project of this scope, the partial convergence result is honest and informative. The agent clearly learned from the reward signal, the Gaussian multiobjective reward is a better design than either of the previous two experiments, and SAC is demonstrably the right algorithm for this class of continuous control problem. The reward function design lesson — match your gradient to your tolerance box, not your problem scale — is the sharpest takeaway and applies directly to any future RL spacecraft guidance work.
