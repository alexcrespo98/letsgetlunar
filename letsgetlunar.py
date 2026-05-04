# letsgetlunar.py
# project 3 - RL orbital insertion
# alex crespo | 2026
# usage: python letsgetlunar.py

import csv
import datetime
import glob
import os
import platform
import subprocess
import sys

import numpy as np

ROOT    = os.path.dirname(os.path.abspath(__file__))
LOGS    = os.path.join(ROOT, 'logs')
MODELS  = os.path.join(ROOT, 'models')
SCRIPTS = os.path.join(ROOT, 'scripts')

os.chdir(ROOT)
os.makedirs(LOGS,   exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

STEPS_PER_HOUR = 850_000
SNAP           = 16_384

BUDGETS_FULL = {'A': 500_000, 'B': 2_000_000, 'C': 2_000_000}
BUDGETS_MIN  = {'A':  50_000, 'B':   200_000,  'C':   200_000}
BUDGET_SHARE = {'A': 0.10,    'B': 0.40,       'C': 0.50}

SUCCESS_THRESH = {'A': -50.0, 'B': -50.0, 'C': 500.0}

EXP_CONFIG = {
    'A': ('PPO', 'sparse',        False),
    'B': ('PPO', 'shaped',        False),
    'C': ('SAC', 'multiobjective', True),
}

ABORT_REASONS = [
    'completed normally',
    'keyboard interrupt',
    'reward plateau',
    'reward decline',
    'time limit reached',
    'other',
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _snap(steps):
    return max(SNAP, round(steps / SNAP) * SNAP)


def _compute_budgets(hours):
    total = hours * STEPS_PER_HOUR
    budgets = {}
    for exp, share in BUDGET_SHARE.items():
        s = _snap(total * share)
        s = max(s, BUDGETS_MIN[exp])
        s = min(s, BUDGETS_FULL[exp])
        budgets[exp] = s
    return budgets


def _find_eval_log(exp):
    """return path to most recent evaluations.npz for exp, or None."""
    tagged = sorted(glob.glob(os.path.join(LOGS, f'exp_{exp}_[0-9][0-9][0-9]', 'evaluations.npz')))
    untagged = os.path.join(LOGS, f'exp_{exp}', 'evaluations.npz')
    candidates = tagged + ([untagged] if os.path.exists(untagged) else [])
    return candidates[-1] if candidates else None


def _read_eval_log(exp):
    """load evaluations.npz for exp, or return None."""
    p = _find_eval_log(exp)
    return np.load(p) if p else None


def _exp_from_zip(path):
    """guess exp letter (A/B/C) from a model zip path."""
    base = os.path.basename(path)
    # new style: exp_A_001.zip
    if base.startswith('exp_') and len(base) >= 6 and base[4] in 'ABC':
        return base[4]
    # old style: path contains known folder names
    lower = path.lower()
    if 'sparse' in lower:
        return 'A'
    if 'shaped' in lower:
        return 'B'
    if 'multiobjective' in lower or 'sac' in lower:
        return 'C'
    return '?'


def _all_model_zips():
    """return sorted list of all .zip paths under models/, best files excluded."""
    all_zips = glob.glob(os.path.join(MODELS, '**', '*.zip'), recursive=True)
    return sorted(z for z in all_zips if not os.path.basename(z).endswith('_best.zip')
                  and 'best_model' not in os.path.basename(z))


def _launch_monitor():
    """launch scripts/check_progress.py in a separate terminal (background)."""
    script = os.path.join(SCRIPTS, 'check_progress.py')
    plat = platform.system()
    try:
        if plat == 'Windows':
            subprocess.Popen(
                ['start', 'cmd', '/k', sys.executable, script],
                shell=True
            )
        elif plat == 'Darwin':
            subprocess.Popen([
                'osascript', '-e',
                f'tell app "Terminal" to do script "{sys.executable} {script}"'
            ])
        else:
            launched = False
            for term in ['xterm', 'gnome-terminal', 'xfce4-terminal']:
                try:
                    subprocess.Popen([term, '-e', f'{sys.executable} {script}'])
                    launched = True
                    break
                except FileNotFoundError:
                    continue
            if not launched:
                print('  note: could not auto-launch monitor. run scripts/check_progress.py manually.')
    except Exception as exc:
        print(f'  note: could not launch monitor ({exc}). run scripts/check_progress.py manually.')


# ── log attempt ───────────────────────────────────────────────────────────────

def _log_attempt(aborted, budgets=None):
    """prompt user to save this training run to logs/attempt_log.csv."""
    ans = input('\nsave this run to logs/attempt_log.csv? (y/n): ').strip().lower()
    if ans != 'y':
        return

    if budgets is None:
        budgets = dict(BUDGETS_FULL)

    log_file = os.path.join(LOGS, 'attempt_log.csv')
    fields = [
        'run_id', 'date', 'exp', 'algorithm', 'reward_fn',
        'hours_trained', 'steps_completed', 'steps_budget',
        'best_reward', 'final_reward', 'success', 'success_rate_pct',
        'abort_reason', 'notes',
    ]

    existing_rows = []
    if os.path.exists(log_file):
        with open(log_file, newline='') as f:
            existing_rows = list(csv.DictReader(f))
    run_id = len(existing_rows) + 1

    default_abort = 'keyboard interrupt' if aborted else 'completed normally'

    # show per-exp stats
    print('\n  eval summary:')
    for exp in ('A', 'B', 'C'):
        d = _read_eval_log(exp)
        if d is None:
            print(f'    exp {exp}: no eval log found')
            continue
        means = d['results'].mean(axis=1)
        steps = int(d['timesteps'][-1])
        best  = float(means.max())
        final = float(means[-1])
        thresh = SUCCESS_THRESH[exp]
        sr = 100.0 * (means >= thresh).sum() / max(len(means), 1)
        print(f'    exp {exp}: steps={steps:,}  best={best:.1f}  recent={final:.1f}  success={sr:.0f}%')

    print('\n  abort reasons:')
    for i, r in enumerate(ABORT_REASONS, 1):
        marker = '  <-- default' if r == default_abort else ''
        print(f'    {i}. {r}{marker}')

    hours_str = input('\n  hours trained (enter to skip): ').strip()

    abort_str = input(f'  abort reason 1-{len(ABORT_REASONS)} (enter for default): ').strip()
    if abort_str == '':
        abort_reason = default_abort
    else:
        try:
            abort_reason = ABORT_REASONS[int(abort_str) - 1]
        except (ValueError, IndexError):
            abort_reason = default_abort

    notes = input('  notes: ').strip()

    rows = []
    for exp, (algorithm, reward_fn, _) in EXP_CONFIG.items():
        d = _read_eval_log(exp)
        if d is None:
            continue
        means  = d['results'].mean(axis=1)
        steps  = int(d['timesteps'][-1])
        best   = float(means.max())
        final  = float(means[-1])
        thresh = SUCCESS_THRESH[exp]
        sr     = 100.0 * (means >= thresh).sum() / max(len(means), 1)
        rows.append({
            'run_id':           run_id,
            'date':             datetime.date.today().isoformat(),
            'exp':              exp,
            'algorithm':        algorithm,
            'reward_fn':        reward_fn,
            'hours_trained':    hours_str,
            'steps_completed':  steps,
            'steps_budget':     budgets.get(exp, BUDGETS_FULL[exp]),
            'best_reward':      f'{best:.1f}',
            'final_reward':     f'{final:.1f}',
            'success':          sr > 50.0,
            'success_rate_pct': f'{sr:.1f}',
            'abort_reason':     abort_reason,
            'notes':            notes,
        })

    if not rows:
        print('  no eval logs found. nothing saved.')
        return

    write_header = not os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerows(rows)

    print(f'\n  saved {len(rows)} row(s) to {log_file}')


# ── menu actions ──────────────────────────────────────────────────────────────

def _train_new_model():
    print('\ntrain a new model')
    hours_str = input('  how much time do you have in hours? (enter for no limit): ').strip()

    if hours_str == '':
        hours   = None
        budgets = dict(BUDGETS_FULL)
        print('  no time limit. using full budgets.')
    else:
        try:
            hours   = float(hours_str)
            budgets = _compute_budgets(hours)
        except ValueError:
            print('  invalid input. using full budgets.')
            hours   = None
            budgets = dict(BUDGETS_FULL)

    print('\n  step budgets:')
    for exp, steps in budgets.items():
        print(f'    exp {exp}: {steps:,} steps')

    print('\n  launching training monitor in background...')
    _launch_monitor()

    sys.path.insert(0, SCRIPTS)
    from train_agent import run_experiments  # noqa: PLC0415

    aborted = False
    try:
        run_experiments(budgets=budgets)
    except KeyboardInterrupt:
        print('\n  training interrupted.')
        aborted = True

    _log_attempt(aborted, budgets=budgets)


def _run_pretrained_model():
    zips = _all_model_zips()
    if not zips:
        print('\n  no models found in models/. train a model first.')
        return

    rows = []
    for z in zips:
        exp   = _exp_from_zip(z)
        d     = _read_eval_log(exp) if exp != '?' else None
        if d is not None:
            means     = d['results'].mean(axis=1)
            best      = float(means.max())
            thresh    = SUCCESS_THRESH.get(exp, 0.0)
            n_success = int((means >= thresh).sum())
        else:
            best      = float('nan')
            n_success = 0
        label = os.path.relpath(z, MODELS)
        rows.append((label, exp, best, n_success, z))

    print('\n  available models:')
    print(f"  {'#':>4}  {'model':<40}  {'exp':>4}  {'best reward':>11}  evals")
    print(f"  {'----':>4}  {'----------------------------------------':<40}  {'----':>4}  {'-----------':>11}  -----")
    for i, (label, exp, best, n_success, z) in enumerate(rows, 1):
        best_str = f'{best:.1f}' if not np.isnan(best) else 'n/a'
        print(f'  {i:>4}  {label:<40}  {exp:>4}  {best_str:>11}  ({n_success})')

    choice = input('\n  pick a model by number (or enter to cancel): ').strip()
    if not choice:
        return

    try:
        idx              = int(choice) - 1
        label, exp, best, n_success, z = rows[idx]
    except (ValueError, IndexError):
        print('  invalid choice.')
        return

    config = EXP_CONFIG.get(exp, ('SAC', 'multiobjective', True))
    algo_name, reward_fn, exploring_starts = config

    sys.path.insert(0, SCRIPTS)
    from lunar_env import LunarOrbitEnv, VC  # noqa: PLC0415

    if algo_name == 'PPO':
        from stable_baselines3 import PPO as AlgoClass  # noqa: PLC0415
    else:
        from stable_baselines3 import SAC as AlgoClass  # noqa: PLC0415

    model_path = z.replace('.zip', '')
    print(f'\n  loading {label}...')
    model = AlgoClass.load(model_path)
    env   = LunarOrbitEnv(reward_fn=reward_fn, exploring_starts=False)

    obs, _  = env.reset()
    done    = False
    total_r = 0.0
    steps   = 0
    info    = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_r += reward
        steps   += 1

    print(f'\n  total reward:  {total_r:.2f}')
    print(f'  final alt:     {info.get("altitude_km", 0.0):.2f} km  (target: 400)')
    print(f'  final Vr:      {info.get("vr", 0.0):.2f} m/s  (target: 0)')
    print(f'  final Vtan:    {info.get("vtan", 0.0):.2f} m/s  (target: {VC:.2f})')
    print(f'  success:       {info.get("success", False)}')
    print(f'  steps:         {steps}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print('letsgetlunar')
    print('project 3 - RL orbital insertion')
    print('alex crespo | 2026')
    print()

    MENU = [
        ('train a new model',      _train_new_model),
        ('run a pre-trained model', _run_pretrained_model),
        ('quit',                   None),
    ]

    while True:
        print('  menu:')
        for i, (label, _) in enumerate(MENU, 1):
            print(f'    {i}. {label}')

        choice = input('\n  pick an option: ').strip()
        try:
            idx        = int(choice) - 1
            label, fn  = MENU[idx]
        except (ValueError, IndexError):
            print('  invalid choice.\n')
            continue

        if fn is None:
            print('\n  goodbye.')
            break
        fn()
        print()


if __name__ == '__main__':
    main()
