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


SUCCESS_THRESH = {'A': -50.0, 'B': -50.0, 'C': 500.0, 'Cstar': 500.0}


def next_model_tag(exp, machine=None):
    """scan models/ for existing exp_X_[machine_]NNN*.zip and return the next tag.

    When *machine* is provided the tag embeds the machine name so two machines
    running concurrently cannot produce the same filename.
    Supports multi-word exp names like 'Cstar'.
    """
    if machine:
        pattern    = os.path.join(MODELS, f'exp_{exp}_{machine}_[0-9][0-9][0-9]*.zip')
        tag_prefix = f'exp_{exp}_{machine}'
        num_idx    = 3  # 'exp_C_main_003[_r810_success]'.split('_')[3] == '003'
                        # 'exp_Cstar_main_003[...]'.split('_')[3] == '003' (same index)
    else:
        pattern    = os.path.join(MODELS, f'exp_{exp}_[0-9][0-9][0-9]*.zip')
        tag_prefix = f'exp_{exp}'
        num_idx    = 2  # 'exp_C_003[_r810_success]'.split('_')[2] == '003'
                        # 'exp_Cstar_003[...]'.split('_')[2] == '003' (same index)

    existing = glob.glob(pattern)
    nums = []
    for f in existing:
        parts = os.path.basename(f).replace('.zip', '').split('_')
        try:
            nums.append(int(parts[num_idx]))
        except (ValueError, IndexError):
            pass
    n = max(nums) + 1 if nums else 1
    return f'{tag_prefix}_{n:03d}'


def _rename_with_reward(tag, success_thresh):
    """rename tag_best.zip (or tag.zip) to include reward and _success suffix.

    Prefers the *_best.zip* checkpoint (best during training) over the final
    weights.  Returns *(final_path, best_reward)*.
    """
    eval_log = os.path.join(LOGS, tag, 'evaluations.npz')
    best_reward = float('nan')
    if os.path.exists(eval_log):
        d = np.load(eval_log)
        best_reward = float(d['results'].mean(axis=1).max())

    if not np.isnan(best_reward):
        r_part = f'_r{int(round(best_reward))}'
        s_part = '_success' if best_reward >= success_thresh else ''
    else:
        r_part = ''
        s_part = ''

    final_name = f'{tag}{r_part}{s_part}.zip'
    final_path = os.path.join(MODELS, final_name)

    staged = os.path.join(MODELS, tag + '_best.zip')
    src = staged if os.path.exists(staged) else os.path.join(MODELS, tag + '.zip')
    if os.path.exists(src) and src != final_path:
        shutil.move(src, final_path)

    plain = os.path.join(MODELS, tag + '.zip')
    if os.path.exists(plain) and plain != final_path:
        os.remove(plain)

    return final_path, best_reward


def run_experiments(budgets=None, exploring_starts_C=True, machine=None):
    """run all three RL experiments. budgets dict maps exp letter to step count."""
    if budgets is None:
        budgets = {'A': 500_000, 'B': 2_000_000, 'C': 2_000_000}

    # exp A: PPO, sparse reward, small net — baseline, not expected to converge
    tag_A = next_model_tag('A', machine=machine)
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
    final_A, _ = _rename_with_reward(tag_A, SUCCESS_THRESH['A'])
    print(f"EXP A DONE  ->  {os.path.basename(final_A)}")

    # exp B: PPO, shaped reward, larger net — expect partial convergence
    tag_B = next_model_tag('B', machine=machine)
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
    final_B, _ = _rename_with_reward(tag_B, SUCCESS_THRESH['B'])
    print(f"EXP B DONE  ->  {os.path.basename(final_B)}")

    # exp C: SAC, multiobjective reward, exploring starts — should converge best
    tag_C = next_model_tag('C', machine=machine)
    print("\n" + "="*60)
    print("EXPERIMENT C: SAC | 128-128 | MULTI-OBJECTIVE | EXPLORING STARTS")
    print(f"model tag: {tag_C}  steps: {budgets['C']:,}")
    print("="*60)

    env_C  = Monitor(LunarOrbitEnv(reward_fn='multiobjective', exploring_starts=exploring_starts_C))
    eval_C = Monitor(LunarOrbitEnv(reward_fn='multiobjective', exploring_starts=False))  # eval always fixed start so scores are comparable across machines

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
    stop_C = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=5, verbose=1)  # 50 instead of 20 — exp C reward is flat for a long time before it clicks
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
    final_C, _ = _rename_with_reward(tag_C, SUCCESS_THRESH['C'])
    print(f"EXP C DONE  ->  {os.path.basename(final_C)}")

    print("\nALL EXPERIMENTS COMPLETE")
    print(f"models saved to: {MODELS}")
    print(f"logs saved to:   {LOGS}")
    print("run: tensorboard --logdir logs/  to view training curves")


def finetune_exp_c(model_path, budget=2_000_000, exploring_starts_C=False,
                   machine=None, tag=None, success_thresh=500.0,
                   sac_kwargs=None, env_kwargs=None, exp='C'):
    """load an existing exp C / C* model and continue training from it.

    Returns *(final_zip_path, best_reward)*.  The saved filename embeds the
    best evaluation reward and a ``_success`` suffix when the model meets the
    success criterion, e.g. ``exp_C_main_003_r810_success.zip``.

    *sac_kwargs* — extra keyword args merged into the SAC constructor.
                   Supports: learning_rate, buffer_size, batch_size, ent_coef,
                   policy_kwargs (net_arch).  When *policy_kwargs* / net_arch
                   differs from the saved model the function creates a new model
                   with those weights randomly initialised (warm-start is skipped
                   for architectural changes).
    *env_kwargs* — extra keyword args forwarded to LunarOrbitEnv
                   (reward_weights, gaussian_widths, …).
    *exp*        — experiment name prefix ('C' or 'Cstar').
    """
    sac_kwargs = dict(sac_kwargs) if sac_kwargs else {}
    env_kwargs = env_kwargs or {}
    tag_C = tag if tag is not None else next_model_tag(exp, machine=machine)
    print("\n" + "="*60)
    print(f"FINE-TUNE EXP {exp}: SAC | warm-start from {os.path.basename(model_path)}")
    print(f"model tag: {tag_C}  additional steps: {budget:,}")
    print("="*60)

    env_C  = Monitor(LunarOrbitEnv(reward_fn='multiobjective', exploring_starts=exploring_starts_C, **env_kwargs))
    eval_C = Monitor(LunarOrbitEnv(reward_fn='multiobjective', exploring_starts=False, **env_kwargs))  # eval always fixed start so scores are comparable across machines

    # split architectural kwargs (policy_kwargs / net_arch) from training kwargs
    policy_kwargs = sac_kwargs.pop('policy_kwargs', None)
    new_arch      = (policy_kwargs or {}).get('net_arch')
    train_kwargs  = sac_kwargs  # lr, buffer_size, batch_size, ent_coef (policy_kwargs already popped)

    if new_arch:
        # architecture change: create a fresh model with the requested architecture
        print(f"  note: new net_arch={new_arch} — creating fresh model (no warm-start)")
        create_kwargs = dict(train_kwargs)
        create_kwargs['policy_kwargs'] = policy_kwargs
        model_C = SAC(
            'MlpPolicy', env_C,
            policy_kwargs=policy_kwargs,
            learning_rate=create_kwargs.pop('learning_rate', 3e-4),
            buffer_size=create_kwargs.pop('buffer_size', 500_000),
            batch_size=create_kwargs.pop('batch_size', 256),
            gamma=0.99,
            tau=0.005,
            ent_coef=create_kwargs.pop('ent_coef', 'auto'),
            verbose=1,
            tensorboard_log=os.path.join(LOGS, tag_C),
            **create_kwargs,
        )
    else:
        # same architecture: load weights and apply non-architectural overrides
        custom_objects = {}
        if 'learning_rate' in train_kwargs:
            custom_objects['learning_rate'] = train_kwargs.pop('learning_rate')
        if 'buffer_size' in train_kwargs:
            custom_objects['buffer_size'] = train_kwargs.pop('buffer_size')
        if 'batch_size' in train_kwargs:
            custom_objects['batch_size'] = train_kwargs.pop('batch_size')
        if 'ent_coef' in train_kwargs:
            custom_objects['ent_coef'] = train_kwargs.pop('ent_coef')
        model_C = SAC.load(model_path.replace('.zip', ''), env=env_C,
                           custom_objects=custom_objects if custom_objects else None)

    best_dir_C = os.path.join(MODELS, tag_C + '_best_tmp')
    os.makedirs(best_dir_C, exist_ok=True)
    stop_C = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=5, verbose=1)  # 50 instead of 20 — exp C reward is flat for a long time before it clicks
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

    final_path, best_reward = _rename_with_reward(tag_C, success_thresh)
    return final_path, best_reward


if __name__ == '__main__':
    run_experiments()
