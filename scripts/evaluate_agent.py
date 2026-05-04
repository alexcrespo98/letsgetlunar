# scripts/evaluate_agent.py
# loads saved models, runs deterministic episodes, saves results to CSV
# alex crespo | 2026

import os, sys, csv
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from lunar_env import LunarOrbitEnv, VC, R_MOON
from stable_baselines3 import PPO, SAC

MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
OUT    = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUT, exist_ok=True)

EXPERIMENTS = [
    ('A', PPO, 'ppo_sparse_small/final',          'sparse',         False),
    ('B', PPO, 'ppo_shaped_large/final',           'shaped',         False),
    ('C', SAC, 'sac_multiobjective_medium/best_model',  'multiobjective', True),
]

summary_rows = []

for exp_id, ModelClass, model_path, reward_fn, exploring_starts in EXPERIMENTS:
    print(f"\n--- EVALUATING EXPERIMENT {exp_id} ---")
    full_path = os.path.join(MODELS, model_path)

    if not os.path.exists(full_path + '.zip'):
        print(f"  [!] Model not found: {full_path}.zip — skipping")
        continue

    model = ModelClass.load(full_path)
    env   = LunarOrbitEnv(reward_fn=reward_fn, exploring_starts=False)  # fixed ICs for fair eval

    # Run one full deterministic episode and record trajectory
    obs, _ = env.reset()
    done    = False
    traj    = []
    total_r = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_r += reward
        traj.append({
            't':        info['t'],
            'alt_km':   info['altitude_km'],
            'vr':       info['vr'],
            'vtan':     info['vtan'],
            'beta_deg': info['beta_deg'],
            'reward':   reward,
        })

    final = traj[-1]
    success = info['success']

    print(f"  Total reward:  {total_r:.2f}")
    print(f"  Final alt:     {final['alt_km']:.2f} km  (target: 400)")
    print(f"  Final Vr:      {final['vr']:.2f} m/s  (target: 0)")
    print(f"  Final Vtan:    {final['vtan']:.2f} m/s  (target: {VC:.2f})")
    print(f"  Success:       {success}")
    print(f"  Steps:         {len(traj)}")

    # Save trajectory CSV (for plotting)
    traj_file = os.path.join(OUT, f'exp_{exp_id}_trajectory.csv')
    with open(traj_file, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=traj[0].keys())
        w.writeheader()
        w.writerows(traj)
    print(f"  Saved: {traj_file}")

    summary_rows.append({
        'exp':        exp_id,
        'algorithm':  ModelClass.__name__,
        'reward_fn':  reward_fn,
        'success':    success,
        'total_reward': f'{total_r:.2f}',
        'final_alt_km': f'{final["alt_km"]:.2f}',
        'final_vr':   f'{final["vr"]:.2f}',
        'final_vtan': f'{final["vtan"]:.2f}',
        'steps':      len(traj),
    })

# Save summary table
summary_file = os.path.join(OUT, 'results_table.csv')
if summary_rows:
    with open(summary_file, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved summary: {summary_file}")

print("\nEVALUATION DONE")