# scripts/train_agent.py
# runs all 3 rl experiments sequentially
# alex crespo | 2026

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from lunar_env import LunarOrbitEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np

MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
LOGS   = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(MODELS, exist_ok=True)
os.makedirs(LOGS,   exist_ok=True)

# ── EXPERIMENT A: PPO, SMALL NETWORK, SPARSE REWARD ─────────────────────────
# Hypothesis: will not converge — sparse reward with 3200 steps/episode
# is too hard. Demonstrates why reward shaping is necessary.
# Expected result: flat reward curve, no successful episodes.
print("\n" + "="*60)
print("EXPERIMENT A: PPO | 64-64 | SPARSE REWARD | FIXED STARTS")
print("="*60)

env_A  = Monitor(LunarOrbitEnv(reward_fn='sparse', exploring_starts=False))
eval_A = Monitor(LunarOrbitEnv(reward_fn='sparse', exploring_starts=False))

model_A = PPO(
    'MlpPolicy', env_A,
    policy_kwargs=dict(net_arch=[64, 64]),  # small — matches study1.m suggestion
    learning_rate=3e-4,
    n_steps=3200,           # one full episode horizon
    batch_size=64,
    n_epochs=10,
    gamma=0.99,             # matches agentOpts.DiscountFactor in study1.m
    ent_coef=0.01,          # EntropyLossWeight analog
    verbose=1,
    tensorboard_log=os.path.join(LOGS, 'exp_A')
)

cb_A = EvalCallback(eval_A, best_model_save_path=os.path.join(MODELS,'ppo_sparse_small'),
                    log_path=os.path.join(LOGS,'exp_A'), eval_freq=16000,
                    n_eval_episodes=5, deterministic=True, verbose=1)

model_A.learn(total_timesteps=500_000, callback=cb_A)
model_A.save(os.path.join(MODELS, 'ppo_sparse_small', 'final'))
print("EXP A DONE")

# ── EXPERIMENT B: PPO, LARGE NETWORK, SHAPED REWARD ─────────────────────────
# Hypothesis: will show some convergence. Larger net + dense reward
# should guide the agent meaningfully. Beta-continuity term built into reward.
# Expected result: partial convergence, smoother beta profile than Proj 2 NLP.
print("\n" + "="*60)
print("EXPERIMENT B: PPO | 256-128-64 | SHAPED REWARD | FIXED STARTS")
print("="*60)

env_B  = Monitor(LunarOrbitEnv(reward_fn='shaped', exploring_starts=False))
eval_B = Monitor(LunarOrbitEnv(reward_fn='shaped', exploring_starts=False))

model_B = PPO(
    'MlpPolicy', env_B,
    policy_kwargs=dict(net_arch=[256, 128, 64]),  # large
    learning_rate=3e-4,
    n_steps=3200,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.005,
    verbose=1,
    tensorboard_log=os.path.join(LOGS, 'exp_B')
)

cb_B = EvalCallback(eval_B, best_model_save_path=os.path.join(MODELS,'ppo_shaped_large'),
                    log_path=os.path.join(LOGS,'exp_B'), eval_freq=16000,
                    n_eval_episodes=5, deterministic=True, verbose=1)

model_B.learn(total_timesteps=2_000_000, callback=cb_B)
model_B.save(os.path.join(MODELS, 'ppo_shaped_large', 'final'))
print("EXP B DONE")

# ── EXPERIMENT C: SAC, MEDIUM NETWORK, MULTI-OBJECTIVE, EXPLORING STARTS ─────
# Hypothesis: best convergence. SAC handles continuous actions better,
# auto-tunes entropy, off-policy so more sample efficient.
# Exploring starts expands state coverage. Multi-objective reward is most
# physically meaningful (min-time + continuity + proximity).
# Expected result: smoothest beta profile, closest to NLP optimal solution.
print("\n" + "="*60)
print("EXPERIMENT C: SAC | 128-128 | MULTI-OBJECTIVE | EXPLORING STARTS")
print("="*60)

env_C  = Monitor(LunarOrbitEnv(reward_fn='multiobjective', exploring_starts=True))
eval_C = Monitor(LunarOrbitEnv(reward_fn='multiobjective', exploring_starts=True))

model_C = SAC(
    'MlpPolicy', env_C,
    policy_kwargs=dict(net_arch=[128, 128]),  # medium
    learning_rate=3e-4,
    buffer_size=500_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef='auto',        # SAC auto-tunes entropy coefficient
    verbose=1,
    tensorboard_log=os.path.join(LOGS, 'exp_C')
)

cb_C = EvalCallback(eval_C, best_model_save_path=os.path.join(MODELS,'sac_multiobjective_medium'),
                    log_path=os.path.join(LOGS,'exp_C'), eval_freq=10000,
                    n_eval_episodes=5, deterministic=True, verbose=1)

model_C.learn(total_timesteps=2_000_000, callback=cb_C)
model_C.save(os.path.join(MODELS, 'sac_multiobjective_medium', 'final'))
print("EXP C DONE")

print("\nALL EXPERIMENTS COMPLETE")
print(f"Models saved to: {MODELS}")
print(f"Logs saved to:   {LOGS}")
print("Run: tensorboard --logdir logs/  to view training curves")