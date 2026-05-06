# scripts/plot_rl_results.py
# RL training dashboard + trajectory visualiser
# reads logs/exp_X/evaluations.npz, loads best model, runs one episode
# alex crespo | 2026
#
# usage:
#   python scripts/plot_rl_results.py              # all exps
#   python scripts/plot_rl_results.py --exp C      # just one
#   python scripts/plot_rl_results.py --no-show    # save only

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT   = os.path.join(os.path.dirname(__file__), '..')
LOGS   = os.path.join(ROOT, 'logs')
MODELS = os.path.join(ROOT, 'models')
OUT    = os.path.join(ROOT, 'output')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from lunar_env import LunarOrbitEnv, R_MOON, D_T, VC, MU

# ── constants ─────────────────────────────────────────────────────────────────
SUCCESS_THRESH = {'A': -50.0, 'B': -50.0, 'C': 500.0, 'Cstar': 500.0}
EXP_LABELS     = {'A': 'Exp A — PPO / sparse', 'B': 'Exp B — PPO / shaped', 'C': 'Exp C — SAC / multi-obj'}
EXP_COLORS     = {'A': '#1f77b4', 'B': '#2ca02c', 'C': '#9467bd'}
EXP_REWARD_FN  = {'A': 'sparse', 'B': 'shaped', 'C': 'multiobjective'}

plt.rcParams.update({
    'font.size': 11,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 120,
})


# ── eval log helpers ───────────────────────────────────────────────────────────

def _find_eval_log(exp):
    tagged   = sorted(glob.glob(os.path.join(LOGS, f'exp_{exp}_[0-9][0-9][0-9]', 'evaluations.npz')))
    untagged = os.path.join(LOGS, f'exp_{exp}', 'evaluations.npz')
    cands    = tagged + ([untagged] if os.path.exists(untagged) else [])
    return cands[-1] if cands else None


def _load_eval(exp):
    p = _find_eval_log(exp)
    if p is None:
        return None, None, None, None, None
    d      = np.load(p)
    steps  = d['timesteps']
    means  = d['results'].mean(axis=1)
    stds   = d['results'].std(axis=1)
    best   = np.maximum.accumulate(means)
    thresh = SUCCESS_THRESH[exp]
    # SB3 only writes 'successes' when info has 'is_success'; fall back to reward threshold
    raw_succ = d.get('successes', None)
    if raw_succ is not None and raw_succ.mean() > 0:
        sr = raw_succ.mean(axis=1) * 100.0
    else:
        sr = (means >= thresh).astype(float) * 100.0   # 0 or 100 per eval window
        # smooth with a rolling window so it reads like a rate
        k  = min(10, len(sr))
        sr = np.convolve(sr, np.ones(k) / k, mode='same')
    return steps, means, stds, best, sr


# ── best model finder ──────────────────────────────────────────────────────────

def _find_best_model(exp):
    """return path (without .zip) to the highest-reward model for this exp."""
    all_zips = glob.glob(os.path.join(MODELS, f'exp_{exp}*.zip'))
    best_r, best_p = float('-inf'), None
    for z in all_zips:
        stem = os.path.basename(z).replace('.zip', '')
        r    = float('nan')
        for part in stem.split('_'):
            if part.startswith('r') and len(part) > 1:
                try:   r = float(part[1:]); break
                except ValueError: pass
        if np.isnan(r):
            # fall back to eval log
            d = _find_eval_log(exp)
            if d:
                dat = np.load(d)
                r   = float(dat['results'].mean(axis=1).max())
        if not np.isnan(r) and r > best_r:
            best_r, best_p = r, z.replace('.zip', '')
    return best_p, best_r


# ── run one episode ────────────────────────────────────────────────────────────

def _run_episode(exp):
    model_path, best_r = _find_best_model(exp)
    if model_path is None:
        print(f'  [{exp}] no model found — skipping trajectory plots')
        return None

    reward_fn = EXP_REWARD_FN[exp]
    if exp in ('A', 'B'):
        from stable_baselines3 import PPO as Algo
    else:
        from stable_baselines3 import SAC as Algo

    print(f'  [{exp}] loading {os.path.basename(model_path)} (best≈{best_r:.1f})')
    model = Algo.load(model_path)
    env   = LunarOrbitEnv(reward_fn=reward_fn, exploring_starts=False)

    obs, _  = env.reset()
    done    = False
    records = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        records.append({
            't':       info['t'],
            'alt_km':  info['altitude_km'],
            'r_m':     env.state[0],
            'theta':   env.state[1],
            'vr':      info['vr'],
            'vtan':    info['vtan'],
            'beta':    info['beta_deg'],
            'success': info['success'],
        })

    traj    = {k: np.array([r[k] for r in records]) for k in records[0]}
    success = bool(records[-1]['success'])
    print(f'  [{exp}] episode done — tf={traj["t"][-1]:.1f}s  '
          f'alt={traj["alt_km"][-1]:.1f}km  success={success}')
    return traj


# ── plot: per-exp RL curves ────────────────────────────────────────────────────

def plot_training(exp, show):
    steps, means, stds, best, sr = _load_eval(exp)
    if steps is None:
        print(f'  [{exp}] no eval log found — skipping training plots')
        return

    color = EXP_COLORS[exp]
    label = EXP_LABELS[exp]
    steps_m = steps / 1e6

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f'{label}  —  training curves', fontsize=13, fontweight='bold')

    # (a) mean reward ± 1σ
    ax = axes[0]
    ax.plot(steps_m, means, color=color, lw=2, label='mean reward')
    ax.fill_between(steps_m, means - stds, means + stds, color=color, alpha=0.15, label='±1σ')
    ax.axhline(SUCCESS_THRESH[exp], color='#ff7f0e', ls='--', lw=1.2, label=f'success threshold ({SUCCESS_THRESH[exp]:.0f})')
    ax.set_ylabel('Mean reward')
    ax.set_title('(a) Eval reward')
    ax.legend(fontsize=8)
    ax.grid(True)

    # (b) best so far
    ax = axes[1]
    ax.plot(steps_m, best, color=color, lw=2, ls='-')
    ax.axhline(SUCCESS_THRESH[exp], color='#ff7f0e', ls='--', lw=1.2)
    ax.set_ylabel('Best reward (cumulative)')
    ax.set_title('(b) Best so far')
    ax.grid(True)

    # (c) success rate
    ax = axes[2]
    ax.plot(steps_m, sr, color=color, lw=2)
    ax.fill_between(steps_m, 0, sr, color=color, alpha=0.12)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Success rate [%]')
    ax.set_xlabel('Timesteps [M]')
    ax.set_title('(c) Success rate (smoothed)')
    ax.grid(True)

    plt.tight_layout()
    fname = os.path.join(OUT, f'rl_{exp.lower()}_training.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'  saved: {os.path.basename(fname)}')
    if show: plt.show()
    plt.close()


# ── plot: trajectory (same style as plot_results.py) ──────────────────────────

def plot_trajectory(exp, traj, show):
    if traj is None:
        return

    color = EXP_COLORS[exp]
    label = EXP_LABELS[exp]
    t     = traj['t']
    tf    = t[-1]

    # fig 1: beta profile
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, traj['beta'], color=color, lw=2)
    ax.axhline(0, color='gray', lw=0.8)
    ax.fill_between(t, traj['beta'], 0, alpha=0.1, color=color)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('β [deg]')
    ax.set_title(f'{label}  —  Thrust angle profile β(t)')
    ax.set_ylim(-100, 100)
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.grid(True)
    plt.tight_layout()
    fname = os.path.join(OUT, f'rl_{exp.lower()}_beta.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'  saved: {os.path.basename(fname)}')
    if show: plt.show()
    plt.close()

    # fig 2: state history (2×2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'{label}  —  State history  (tf = {tf:.1f} s)', fontsize=13, fontweight='bold')

    axes[0, 0].plot(t, traj['alt_km'], color=color, lw=1.5)
    axes[0, 0].axhline(400, color='#ff7f0e', ls='--', lw=1.2, label='Target (400 km)')
    axes[0, 0].set_ylabel('Altitude [km]')
    axes[0, 0].set_title('(a) Altitude')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, traj['vr'], color=color, lw=1.5)
    axes[0, 1].axhline(0, color='gray', lw=0.8)
    axes[0, 1].set_ylabel('Radial velocity [m/s]')
    axes[0, 1].set_title('(b) Radial velocity')
    axes[0, 1].grid(True)

    axes[1, 0].plot(t, traj['vtan'], color=color, lw=1.5)
    axes[1, 0].axhline(VC, color='#ff7f0e', ls='--', lw=1.2, label=f'V_circ ({VC:.0f} m/s)')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Tangential velocity [m/s]')
    axes[1, 0].set_title('(c) Tangential velocity')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True)

    axes[1, 1].plot(t, traj['beta'], color=color, lw=1.5)
    axes[1, 1].axhline(0, color='gray', lw=0.8)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('β [deg]')
    axes[1, 1].set_title('(d) Thrust angle β')
    axes[1, 1].set_yticks([-90, -45, 0, 45, 90])
    axes[1, 1].grid(True)

    plt.tight_layout()
    fname = os.path.join(OUT, f'rl_{exp.lower()}_state_history.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'  saved: {os.path.basename(fname)}')
    if show: plt.show()
    plt.close()

    # fig 3: polar trajectory
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})
    r_km = traj['r_m'] / 1000.0
    th   = np.linspace(0, 2 * np.pi, 360)
    ax.plot(traj['theta'], r_km, color=color, lw=2.5, label='NN trajectory')
    ax.plot(th, np.full_like(th, R_MOON / 1000), '#7f7f7f', lw=2.5, label='Moon surface')
    ax.plot(th, np.full_like(th, D_T / 1000),   '#ff7f0e', lw=1.5, ls='--', label='Target orbit (400 km)')
    ax.plot(traj['theta'][0],  r_km[0],  'ko', ms=9,  label='Launch')
    ax.plot(traj['theta'][-1], r_km[-1], 'g*', ms=12, label='Orbit insertion')
    ax.set_rmin(R_MOON / 1000 - 200)
    ax.set_title(f'{label}  —  Polar trajectory', pad=25, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    fname = os.path.join(OUT, f'rl_{exp.lower()}_polar.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'  saved: {os.path.basename(fname)}')
    if show: plt.show()
    plt.close()

    # fig 4: altitude vs angular position (ground track)
    ang_deg = np.degrees(traj['theta'])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ang_deg, traj['alt_km'], color=color, lw=2)
    ax.plot(ang_deg[0],  traj['alt_km'][0],  'ko', ms=8,  label='Launch')
    ax.plot(ang_deg[-1], traj['alt_km'][-1], 'g*', ms=12,
            label=f'Insertion ({traj["alt_km"][-1]:.1f} km)')
    ax.axhline(400, color='#ff7f0e', ls='--', lw=1.2, label='Target (400 km)')
    ax.set_xlabel('Angular position [deg]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title(f'{label}  —  Ground track')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fname = os.path.join(OUT, f'rl_{exp.lower()}_ground_track.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'  saved: {os.path.basename(fname)}')
    if show: plt.show()
    plt.close()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',     default=None, choices=['A', 'B', 'C'],
                        help='single experiment to plot (default: all)')
    parser.add_argument('--no-show', action='store_true',
                        help='save figures without calling plt.show()')
    args = parser.parse_args()

    exps = [args.exp] if args.exp else ['A', 'B', 'C']
    show = not args.no_show

    for exp in exps:
        print(f'\n── Exp {exp} ──────────────────────────────────────')
        plot_training(exp, show=show)
        traj = _run_episode(exp)
        plot_trajectory(exp, traj, show=show)

    print('\ndone — figures saved to output/')


if __name__ == '__main__':
    main()
