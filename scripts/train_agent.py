# scripts/train_agent.py
# runs all 3 rl experiments sequentially
# alex crespo | 2026

import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

from lunar_env import LunarOrbitEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
import numpy as np

MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
LOGS   = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(MODELS, exist_ok=True)
os.makedirs(LOGS,   exist_ok=True)


def next_model_tag(exp):
    """scan models/ for existing exp_X_NNN.zip and return the next tag."""
    pattern  = os.path.join(MODELS, f'exp_{exp}_[0-9][0-9][0-9].zip')
    existing = glob.glob(pattern)
    if not existing:
        return f'exp_{exp}_001'
    nums = []
    for f in existing:
        base = os.path.basename(f).replace('.zip', '')
        try:
            nums.append(int(base.split('_')[-1]))
        except ValueError:
            pass
    return f'exp_{exp}_{max(nums) + 1:03d}'


def run_experiments(budgets=None, exploring_starts_C=True):
    """run all three RL experiments. budgets dict maps exp letter to step count."""
    if budgets is None:
        budgets = {'A': 500_000, 'B': 2_000_000, 'C': 2_000_000}

    # exp A: PPO, sparse reward, small net — baseline, not expected to converge
    tag_A = next_model_tag('A')
    print("\n" + "="*60)
    print("EXPERIMENT A: PPO | 64-64 | SPARSE REWARD | FIXED STARTS")
    print(f"model tag: {tag_A}  steps: {budgets['A']:,}")
    print("="*60)

    env_A  = Monitor(LunarOrbitEnv(reward_fn='sparse', exploring_starts=False))
    eval_A = Monitor(LunarOrbitEnv(reward_fn='sparse', exploring_starts=False))

    model_A = PPO(
        'MlpPolicy', env_A,
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_rate=3e-4,
        n_steps=3200,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=os.path.join(LOGS, tag_A)
    )

    best_dir_A = os.path.join(MODELS, tag_A + '_best_tmp')
    os.makedirs(best_dir_A, exist_ok=True)
    stop_A = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=5, verbose=1)
    cb_A   = EvalCallback(
        eval_A,
        best_model_save_path=best_dir_A,
        log_path=os.path.join(LOGS, tag_A),
        eval_freq=16000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
        callback_after_eval=stop_A,
    )

    model_A.learn(total_timesteps=budgets['A'], callback=cb_A)
    model_A.save(os.path.join(MODELS, tag_A))

    best_src_A = os.path.join(best_dir_A, 'best_model.zip')
    if os.path.exists(best_src_A):
        shutil.move(best_src_A, os.path.join(MODELS, tag_A + '_best.zip'))
    shutil.rmtree(best_dir_A, ignore_errors=True)
    print(f"EXP A DONE  ->  {tag_A}.zip")

    # exp B: PPO, shaped reward, larger net — expect partial convergence
    tag_B = next_model_tag('B')
    print("\n" + "="*60)
    print("EXPERIMENT B: PPO | 256-128-64 | SHAPED REWARD | FIXED STARTS")
    print(f"model tag: {tag_B}  steps: {budgets['B']:,}")
    print("="*60)

    env_B  = Monitor(LunarOrbitEnv(reward_fn='shaped', exploring_starts=False))
    eval_B = Monitor(LunarOrbitEnv(reward_fn='shaped', exploring_starts=False))

    model_B = PPO(
        'MlpPolicy', env_B,
        policy_kwargs=dict(net_arch=[256, 128, 64]),
        learning_rate=3e-4,
        n_steps=3200,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=os.path.join(LOGS, tag_B)
    )

    best_dir_B = os.path.join(MODELS, tag_B + '_best_tmp')
    os.makedirs(best_dir_B, exist_ok=True)
    stop_B = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=5, verbose=1)
    cb_B   = EvalCallback(
        eval_B,
        best_model_save_path=best_dir_B,
        log_path=os.path.join(LOGS, tag_B),
        eval_freq=16000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
        callback_after_eval=stop_B,
    )

    model_B.learn(total_timesteps=budgets['B'], callback=cb_B)
    model_B.save(os.path.join(MODELS, tag_B))

    best_src_B = os.path.join(best_dir_B, 'best_model.zip')
    if os.path.exists(best_src_B):
        shutil.move(best_src_B, os.path.join(MODELS, tag_B + '_best.zip'))
    shutil.rmtree(best_dir_B, ignore_errors=True)
    print(f"EXP B DONE  ->  {tag_B}.zip")

    # exp C: SAC, multiobjective reward, exploring starts — should converge best
    tag_C = next_model_tag('C')
    print("\n" + "="*60)
    print("EXPERIMENT C: SAC | 128-128 | MULTI-OBJECTIVE | EXPLORING STARTS")
    print(f"model tag: {tag_C}  steps: {budgets['C']:,}")
    print("="*60)

    env_C  = Monitor(LunarOrbitEnv(reward_fn='multiobjective', exploring_starts=exploring_starts_C))
    eval_C = Monitor(LunarOrbitEnv(reward_fn='multiobjective', exploring_starts=exploring_starts_C))

    model_C = SAC(
        'MlpPolicy', env_C,
        policy_kwargs=dict(net_arch=[128, 128]),
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
        verbose=1,
        tensorboard_log=os.path.join(LOGS, tag_C)
    )

    best_dir_C = os.path.join(MODELS, tag_C + '_best_tmp')
    os.makedirs(best_dir_C, exist_ok=True)
    stop_C = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=5, verbose=1)
    cb_C   = EvalCallback(
        eval_C,
        best_model_save_path=best_dir_C,
        log_path=os.path.join(LOGS, tag_C),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
        callback_after_eval=stop_C,
    )

    model_C.learn(total_timesteps=budgets['C'], callback=cb_C)
    model_C.save(os.path.join(MODELS, tag_C))

    best_src_C = os.path.join(best_dir_C, 'best_model.zip')
    if os.path.exists(best_src_C):
        shutil.move(best_src_C, os.path.join(MODELS, tag_C + '_best.zip'))
    shutil.rmtree(best_dir_C, ignore_errors=True)
    print(f"EXP C DONE  ->  {tag_C}.zip")

    print("\nALL EXPERIMENTS COMPLETE")
    print(f"models saved to: {MODELS}")
    print(f"logs saved to:   {LOGS}")
    print("run: tensorboard --logdir logs/  to view training curves")


def finetune_exp_c(model_path, budget=2_000_000, exploring_starts_C=False):
    """load an existing exp C model and continue training from it."""
    tag_C = next_model_tag('C')
    print("\n" + "="*60)
    print(f"FINE-TUNE EXP C: SAC | warm-start from {os.path.basename(model_path)}")
    print(f"model tag: {tag_C}  additional steps: {budget:,}")
    print("="*60)

    env_C  = Monitor(LunarOrbitEnv(reward_fn='multiobjective', exploring_starts=exploring_starts_C))
    eval_C = Monitor(LunarOrbitEnv(reward_fn='multiobjective', exploring_starts=exploring_starts_C))

    model_C = SAC.load(model_path.replace('.zip', ''), env=env_C)

    best_dir_C = os.path.join(MODELS, tag_C + '_best_tmp')
    os.makedirs(best_dir_C, exist_ok=True)
    stop_C = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=5, verbose=1)
    cb_C = EvalCallback(
        eval_C,
        best_model_save_path=best_dir_C,
        log_path=os.path.join(LOGS, tag_C),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
        callback_after_eval=stop_C,
    )

    model_C.learn(total_timesteps=budget, callback=cb_C, reset_num_timesteps=True)
    model_C.save(os.path.join(MODELS, tag_C))

    best_src_C = os.path.join(best_dir_C, 'best_model.zip')
    if os.path.exists(best_src_C):
        shutil.move(best_src_C, os.path.join(MODELS, tag_C + '_best.zip'))
    shutil.rmtree(best_dir_C, ignore_errors=True)
    print(f"FINE-TUNE DONE  ->  {tag_C}.zip")


if __name__ == '__main__':
    run_experiments()
