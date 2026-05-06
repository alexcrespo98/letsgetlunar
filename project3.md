# project 3 — lunar orbital insertion via reinforcement learning
alex crespo | 2026

## the assignment

design a reinforcement learning agent to control the thrust angle β(t) of a spacecraft performing lunar orbital insertion. the target is a 400 km circular orbit around the moon (altitude ±20 km, radial velocity < 50 m/s, tangential velocity within 100 m/s of ~1511 m/s). deliverable: working code + 2–4 page write-up.

## what carried over from project 2

the physics did. the equations of motion, the RK4 integrator, the lunar constants (R_moon, μ, circular orbit velocity), and the success tolerances are all taken directly from the project 2 optimal control formulation. the NLP optimal β(t) profile from project 2 informed the reward shaping — specifically, knowing that the optimal thrust angle starts near +90° and transitions to negative values during insertion helped calibrate the gaussian reward widths and gave a sanity check that the RL agent was learning something physically reasonable. the NLP result was not used as a controller or constraint — just as a reference.

## what is new

everything else. the gymnasium environment (`lunar_env.py`), the three reward functions (sparse, shaped, multiobjective), the RL training pipeline, all the infrastructure below.

## experiments

**experiment A — PPO, sparse reward, 64×64 network**
baseline. sparse reward only fires at terminal state (+1000 success, −500 crash, −100 timeout). PPO with a small network learns essentially nothing — the signal is too rare. confirmed as expected: no convergence, reward stays near −100.

**experiment B — PPO, shaped reward, 256×128×64 network**
dense shaping via quadratic penalties on altitude error, radial velocity, and tangential velocity error. larger network, still PPO. marginal improvement over A but no meaningful convergence. PPO struggles with the continuous action space and long horizon.

**experiment C — SAC, multi-objective gaussian reward, 128×128 network**
switched to Soft Actor-Critic (off-policy, entropy-regularized, handles continuous actions well). reward replaced quadratic penalties with gaussian proximity terms so the agent gets a smooth, informative signal everywhere in state space rather than just near the goal. added a control effort penalty and a small time penalty. this is where real learning happened.

## what i tried to push it further

**fine-tuning / warm-starting** — after exp C converged partially, saved checkpoints were used as starting points for continued training with a longer budget. tagged saves (exp_C_main_001, exp_C_backup_003, etc.) keep every checkpoint.

**hyperparameter sweep** — 8 configs bracketing the exp C baseline were run in parallel (one per physical core), each with a different random seed. configs vary: learning rate (2e-4, 3e-4, 5e-4), network width (128×64, 128×128, 256×128), replay buffer size (250k, 500k, 750k), batch size (128, 256), entropy coefficient (auto vs fixed 0.05), and gaussian altitude width (30 km vs 50 km). results ranked by best reward and logged to `logs/sweep_results.csv`.

**two-machine collab mode** — a peer-to-peer TCP socket session (port 7777) between the windows PC and a salvaged macbook air running ubuntu. both machines train the same base model simultaneously with different seeds, transfer the best weights back and forth automatically, and keep a shared "model pot" ranked by best eval reward. this ran overnight multiple times and is how the best model was found — the backup machine found a reward of 912, which i only caught the next morning. glad i salvaged that laptop.

**multi-core parallel sweep** — beyond the two-machine setup, the windows PC (12 physical cores) spawns multiple SAC workers simultaneously via `multiprocessing.spawn`, each with a unique seed and config. live status table refreshes every 30s showing per-worker step count and best reward.

**live GUI monitor** — a matplotlib window (`gui_monitor.py`) that stays open during fine-tuning and sweep runs, showing the eval reward curve updating in real time with wall-clock timestamps.

**exploring starts** — exp C training env randomizes the initial state (altitude, velocity) within a range around the nominal IC. this prevents the agent from memorizing a single trajectory and forces it to learn a general policy.

## compute

over 100 hours of total compute time across the windows PC and the ubuntu macbook. best reward achieved: **912** (found on the auxiliary machine during a collab session). success threshold for exp C is 500.

## what i could try next (out of scope / didn't have time)

- **curriculum learning** — start with a narrow gaussian reward and tighten the widths over training, forcing the agent to converge rather than hover near the goal
- **behaviour cloning pre-training** — use the NLP β(t) profile to generate (obs, action) pairs and do a supervised pre-training pass on the SAC actor before RL fine-tuning, to give the policy a physically meaningful starting point
- **recurrent policy** — replace the MLP with an LSTM so the agent has memory of thrust history, which might help it handle the two-phase structure of the insertion maneuver
- **model-based RL** — learn a dynamics model from collected transitions and use it for planning (Dyna-style), which would be much more sample-efficient on a problem where the true dynamics are known
- **reward annealing** — start with wide gaussians (easy signal) and anneal toward the true tolerances over training
