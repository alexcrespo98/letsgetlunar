# scripts/evaluate_agent.py
# loads saved models, runs deterministic episodes, saves trajectory CSVs,
# generates per-experiment plots and a comparison summary, then shows them all.
# alex crespo | 2026
# usage: python scripts/evaluate_agent.py

import glob
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print('  evaluate_agent: matplotlib not available. plots will be skipped.')

from lunar_env import LunarOrbitEnv, VC, ALT_T, TOL_VR, TOL_VT
from stable_baselines3 import PPO, SAC

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
MODELS  = os.path.join(SCRIPTS, '..', 'models')
OUT     = os.path.join(SCRIPTS, '..', 'output')
os.makedirs(OUT, exist_ok=True)


_EXP_META = {
    'A':     (PPO, 'sparse',         False),
    'B':     (PPO, 'shaped',         False),
    'C':     (SAC, 'multiobjective', True),
    'Cstar': (SAC, 'multiobjective', True),
}


def _find_model_path(exp):
    """find the best model zip for the given experiment letter.

    Preference order:
    1. reward-encoded filename with _success suffix (highest reward wins)
    2. reward-encoded filename without _success
    3. any exp_X_*.zip (latest by name)
    """
    patterns_pref = [
        os.path.join(MODELS, f'exp_{exp}_*_success.zip'),
        os.path.join(MODELS, f'exp_{exp}_*.zip'),
    ]
    for pat in patterns_pref:
        matches = sorted(glob.glob(pat))
        matches = [m for m in matches
                   if not os.path.basename(m).endswith('_best.zip')
                   and 'best_model' not in os.path.basename(m)]
        if matches:
            # among reward-encoded names, prefer highest reward
            def _r(p):
                stem = os.path.basename(p).replace('.zip', '')
                for part in stem.split('_'):
                    if part.startswith('r') and len(part) > 1:
                        try:
                            return float(part[1:])
                        except ValueError:
                            pass
                return float('-inf')
            return max(matches, key=_r)
    return None


def _run_episode(model, reward_fn, exploring_starts=False):
    """run one deterministic episode and return (traj, total_r, info)."""
    env  = LunarOrbitEnv(reward_fn=reward_fn, exploring_starts=exploring_starts)
    obs, _ = env.reset()
    done    = False
    traj    = []
    total_r = 0.0
    info    = {}
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
    return traj, total_r, info


def _save_csv(traj, exp_id):
    path = os.path.join(OUT, f'exp_{exp_id}_trajectory.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=traj[0].keys())
        w.writeheader()
        w.writerows(traj)
    return path


def _plot_experiment(traj, exp_id):
    """generate per-experiment figures (5 subplots) and save to output/."""
    if not HAS_PLT:
        return []

    ts       = [r['t']        for r in traj]
    alts     = [r['alt_km']   for r in traj]
    vrs      = [r['vr']       for r in traj]
    vtans    = [r['vtan']     for r in traj]
    betas    = [r['beta_deg'] for r in traj]
    rewards  = [r['reward']   for r in traj]

    saved = []

    # ── Altitude vs time ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, alts, linewidth=1.5)
    ax.axhline(ALT_T / 1e3, color='red', linestyle='--', linewidth=1, label=f'target {ALT_T/1e3:.0f} km')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('altitude (km)')
    ax.set_title(f'exp {exp_id} — altitude vs time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, f'exp_{exp_id}_altitude.png')
    fig.savefig(p, dpi=120)
    saved.append(p)

    # ── Radial velocity vs time ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, vrs, linewidth=1.5, color='tab:orange')
    ax.axhline(0.0, color='red', linestyle='--', linewidth=1, label='target Vr = 0')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('radial velocity Vr (m/s)')
    ax.set_title(f'exp {exp_id} — radial velocity vs time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, f'exp_{exp_id}_vr.png')
    fig.savefig(p, dpi=120)
    saved.append(p)

    # ── Tangential velocity vs time ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, vtans, linewidth=1.5, color='tab:green')
    ax.axhline(VC, color='red', linestyle='--', linewidth=1, label=f'target Vc = {VC:.0f} m/s')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('tangential velocity Vtan (m/s)')
    ax.set_title(f'exp {exp_id} — tangential velocity vs time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, f'exp_{exp_id}_vtan.png')
    fig.savefig(p, dpi=120)
    saved.append(p)

    # ── Beta vs time ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, betas, linewidth=1.5, color='tab:purple')
    ax.axhline(0.0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('thrust angle β (deg)')
    ax.set_title(f'exp {exp_id} — thrust angle vs time')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, f'exp_{exp_id}_beta.png')
    fig.savefig(p, dpi=120)
    saved.append(p)

    # ── Reward vs time ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, rewards, linewidth=1.2, color='tab:red', alpha=0.8)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('step reward')
    ax.set_title(f'exp {exp_id} — reward vs time')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, f'exp_{exp_id}_reward.png')
    fig.savefig(p, dpi=120)
    saved.append(p)

    return saved


def _plot_summary(summary_rows):
    """bar-chart comparison of final alt, Vr, Vtan vs targets across experiments."""
    if not HAS_PLT or not summary_rows:
        return None

    exps     = [r['exp']          for r in summary_rows]
    alts     = [float(r['final_alt_km']) for r in summary_rows]
    vrs      = [abs(float(r['final_vr'])) for r in summary_rows]
    vtans    = [abs(float(r['final_vtan']) - VC) for r in summary_rows]

    x   = np.arange(len(exps))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(exps) * 2.5), 5))

    ax.bar(x - w,   alts,  w, label='final alt (km)',         color='tab:blue',   alpha=0.8)
    ax.bar(x,       vrs,   w, label='|Vr| (m/s)',             color='tab:orange', alpha=0.8)
    ax.bar(x + w,   vtans, w, label='|Vtan − Vc| (m/s)',      color='tab:green',  alpha=0.8)

    ax.axhline(ALT_T / 1e3,  color='tab:blue',   linestyle='--', linewidth=1, alpha=0.5, label=f'alt target {ALT_T/1e3:.0f} km')
    ax.axhline(TOL_VR,       color='tab:orange', linestyle='--', linewidth=1, alpha=0.5, label=f'Vr target < {TOL_VR:.0f} m/s')
    ax.axhline(TOL_VT,       color='tab:green',  linestyle='--', linewidth=1, alpha=0.5, label=f'ΔVtan target < {TOL_VT:.0f} m/s')

    ax.set_xticks(x)
    ax.set_xticklabels([f'exp {e}' for e in exps])
    ax.set_ylabel('value')
    ax.set_title('experiment comparison — final state vs targets')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    p = os.path.join(OUT, 'exp_comparison_summary.png')
    fig.savefig(p, dpi=120)
    return p


def run_all_experiments(show=True):
    """find, evaluate, and plot all available experiments (A, B, C, Cstar).

    Called when this script is run standalone.
    """
    summary_rows = []

    for exp_id, (ModelClass, reward_fn, exploring_starts) in _EXP_META.items():
        model_path = _find_model_path(exp_id)
        if model_path is None:
            print(f'  [!] no model found for exp {exp_id} — skipping')
            continue

        print(f'\n--- evaluating experiment {exp_id} ---')
        print(f'  model: {os.path.basename(model_path)}')

        model = ModelClass.load(model_path.replace('.zip', ''))
        traj, total_r, info = _run_episode(model, reward_fn, exploring_starts=False)
        final   = traj[-1]
        success = info.get('success', False)

        print(f'  total reward:  {total_r:.2f}')
        print(f'  final alt:     {final["alt_km"]:.2f} km  (target: 400)')
        print(f'  final Vr:      {final["vr"]:.2f} m/s  (target: 0)')
        print(f'  final Vtan:    {final["vtan"]:.2f} m/s  (target: {VC:.2f})')
        print(f'  success:       {success}')
        print(f'  steps:         {len(traj)}')

        csv_path = _save_csv(traj, exp_id)
        print(f'  saved CSV:     {csv_path}')

        saved_figs = _plot_experiment(traj, exp_id)
        for p in saved_figs:
            print(f'  saved figure:  {p}')

        summary_rows.append({
            'exp':          exp_id,
            'algorithm':    ModelClass.__name__,
            'reward_fn':    reward_fn,
            'success':      success,
            'total_reward': f'{total_r:.2f}',
            'final_alt_km': f'{final["alt_km"]:.2f}',
            'final_vr':     f'{final["vr"]:.2f}',
            'final_vtan':   f'{final["vtan"]:.2f}',
            'steps':        len(traj),
        })

    # summary CSV
    summary_file = os.path.join(OUT, 'results_table.csv')
    if summary_rows:
        with open(summary_file, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        print(f'\n  saved summary CSV: {summary_file}')

    # comparison figure
    cmp_fig = _plot_summary(summary_rows)
    if cmp_fig:
        print(f'  saved comparison:  {cmp_fig}')

    # terminal summary table
    if summary_rows:
        print('\n' + '='*90)
        print(f"  {'exp':<6}  {'algorithm':<6}  {'reward_fn':<14}  {'success':<8}  "
              f"{'total_reward':>12}  {'alt_km':>8}  {'Vr':>8}  {'Vtan':>8}")
        print('  ' + '-'*88)
        for r in summary_rows:
            print(f"  {r['exp']:<6}  {r['algorithm']:<6}  {r['reward_fn']:<14}  "
                  f"{str(r['success']):<8}  {r['total_reward']:>12}  "
                  f"{r['final_alt_km']:>8}  {r['final_vr']:>8}  {r['final_vtan']:>8}")
        print('='*90)

    if show and HAS_PLT:
        plt.show()

    print('\nevaluation done.')


def run_and_plot_single(model_zip, exp_id, algo_name, reward_fn, show=True):
    """evaluate a single model and show its plots.  called from letsgetlunar._run_pretrained_model()."""
    if algo_name == 'PPO':
        from stable_baselines3 import PPO as ModelClass
    else:
        from stable_baselines3 import SAC as ModelClass

    model_path = model_zip.replace('.zip', '')
    model = ModelClass.load(model_path)

    _, _, exploring_starts = _EXP_META.get(exp_id, (None, reward_fn, False))
    traj, total_r, info = _run_episode(model, reward_fn, exploring_starts=False)
    final   = traj[-1]
    success = info.get('success', False)

    csv_path = _save_csv(traj, exp_id)
    print(f'  saved CSV: {csv_path}')

    saved_figs = _plot_experiment(traj, exp_id)
    for p in saved_figs:
        print(f'  saved figure: {p}')

    # summary figure for this single experiment
    summary_row = [{
        'exp':          exp_id,
        'algorithm':    algo_name,
        'reward_fn':    reward_fn,
        'success':      success,
        'total_reward': f'{total_r:.2f}',
        'final_alt_km': f'{final["alt_km"]:.2f}',
        'final_vr':     f'{final["vr"]:.2f}',
        'final_vtan':   f'{final["vtan"]:.2f}',
        'steps':        len(traj),
    }]
    cmp_fig = _plot_summary(summary_row)
    if cmp_fig:
        print(f'  saved comparison: {cmp_fig}')

    if show and HAS_PLT:
        plt.show()


if __name__ == '__main__':
    run_all_experiments(show=True)
