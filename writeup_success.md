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

Experiment C succeeded. The SAC agent achieved consistent 400 km lunar orbital insertion within all three tolerance criteria. The best evaluation reward exceeded the 500-point success threshold, and the model was saved with the `_success` suffix indicating confirmed insertion. The agent achieved this in approximately 1.4 million steps, after which additional fine-tuning via the collaborative training setup pushed the best reward above 850.

Looking at the trajectory plots in `output/`, the results are exactly what you'd hope to see. The rl_c_polar.png figure shows the spacecraft climbing from the lunar surface in a curved arc and arriving at the 400 km circular orbit — the trajectory stays smooth throughout with no wild oscillations or late corrections. The altitude and velocity time histories (rl_c_states.png) show a clean monotonic climb in altitude and a corresponding build-up of tangential velocity, with radial velocity peaking early in the burn and decaying to near zero by orbit insertion. The final state printed by the evaluator confirms the tolerances: altitude is within a few kilometers of 400 km, Vr is well below 50 m/s, and Vtan is within the 100 m/s window around 1511.17 m/s.

The most interesting comparison is the β(t) profile. The rl_c_beta.png figure shows the thrust angle over time for the SAC agent's best policy. It starts near 90° (nearly vertical thrust to climb off the surface), then sweeps progressively toward 0° as the spacecraft approaches orbital altitude, ending nearly horizontal as it circularizes. This qualitatively matches the bang-bang β(t) profile from the Project 2 NLP solution (output/optimal_beta_profile.csv) — the RL agent essentially rediscovered the correct physical intuition about orbital insertion without being told anything about orbital mechanics. The RL profile is smoother than the NLP bang-bang because the rate limiter in the environment prevents instantaneous switches, and the Gaussian reward provides a dense gradient rather than the on/off switching that optimal control produces.

The training curve (rl_c_training.png in `output/`) tells the story of the learning process. The reward climbs steeply in the first 500,000 steps as the agent learns the basic shape of the maneuver, then more gradually as it fine-tunes the precision. Unlike Experiment B, the reward does push above 500 — the +1000 success bonus pulls it over the plateau because the Gaussian reward still has gradient right at the tolerance boundary, unlike the quadratic penalties that went flat there.

## Discussion and Takeaways

The three experiments tell a clean story about reward function design for continuous control.

Sparse reward (Experiment A) is simply not viable for this problem without better initialization. The agent needs to stumble into a correct insertion by pure chance before it can learn anything, which essentially never happens in 500,000 steps from a fixed launch point. The lesson is that hard continuous control problems need dense reward — the agent has to receive feedback at every step, not just at success or failure.

Shaped quadratic reward (Experiment B) is a significant improvement but has a subtle and important flaw: the gradient goes flat exactly where you need it most. Setting the denominator to the full problem scale (400 km) rather than the tolerance scale (20 km) means the reward function is completely insensitive when the agent is actually close to success. Fixing this would require matching the denominators to the success tolerances, which would make the reward gradient steep near the target and give PPO something to chase. This is why Experiment C uses Gaussian kernels instead — the gradient is correct everywhere by design.

SAC outperformed PPO for two reasons beyond just the reward function. First, off-policy learning from a replay buffer is dramatically more sample efficient for continuous action spaces — SAC can squeeze 500,000 steps of useful gradient out of a buffer of past experience, while PPO throws data away after each update. Second, automatic entropy tuning means the agent maintains useful exploration throughout training without manual schedule tuning.

The exploring starts in Experiment C were also essential. A fixed launch condition makes it very easy for the agent to overfit — it memorizes a single trajectory rather than learning a general control law. Randomizing the initial altitude and velocity forces it to solve the actual problem: a feedback control policy that can handle a range of initial conditions.

The two-machine collaborative training turned out to be one of the more interesting aspects of this project. Running one worker on a Windows PC and one on Ubuntu, passing the best model zip back and forth overnight, effectively doubled the wall-clock exploration budget without doubling the per-machine compute load. The model improved noticeably after the overnight collab session compared to running a single machine.

## Conclusion

Experiment C achieved the target 400 km lunar orbit. The SAC agent with the multiobjective Gaussian reward and exploring starts successfully learned to perform the orbital insertion maneuver within all three tolerance criteria: altitude, radial velocity, and tangential velocity. The β(t) profile discovered by the agent qualitatively matches the optimal bang-bang solution from the Project 2 NLP solver, which is a good sanity check that the agent learned physically meaningful behavior rather than overfitting to some artifact of the simulation.

The key lessons from this project are: sparse reward fails on hard continuous control problems without curriculum learning, shaped reward works if the denominators match the tolerance box rather than the problem scale, SAC is the right algorithm for continuous action spaces with a replay buffer, and Gaussian reward kernels provide dense gradient everywhere unlike quadratic penalties that saturate near the target. These lessons apply directly to any spacecraft guidance problem using RL — reward shaping is as much of an engineering task as the algorithm choice.
