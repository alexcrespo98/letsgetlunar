# scripts/check_progress.py - terminal training monitor
import glob
import numpy as np
import os
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(ROOT)

# ── detect fine-tune mode ────────────────────────────────────────────────────
SENTINEL = os.path.join('logs', '.finetune_mode')
FINETUNE_MODE = os.path.exists(SENTINEL)
FINETUNE_BUDGET = None
if FINETUNE_MODE:
    try:
        with open(SENTINEL) as _sf:
            FINETUNE_BUDGET = int(_sf.read().strip())
    except (ValueError, OSError):
        print('  warning: could not read fine-tune budget from sentinel file; defaulting to 2,000,000 steps.')
        FINETUNE_BUDGET = 2_000_000

TRAIN_START = None
_exps_to_check = ('C',) if FINETUNE_MODE else ('A', 'B', 'C')
for _exp in _exps_to_check:
    # check tagged dirs (exp_A_001) then untagged (exp_A)
    _tagged = sorted(glob.glob(os.path.join('logs', f'exp_{_exp}_[0-9][0-9][0-9]', 'evaluations.npz')))
    _plain  = os.path.join('logs', f'exp_{_exp}', 'evaluations.npz')
    _candidates = _tagged + ([_plain] if os.path.exists(_plain) else [])
    if _candidates:
        TRAIN_START = os.path.getmtime(_candidates[-1])
        break
if TRAIN_START is None:
    TRAIN_START = time.time()

TOTAL    = {'A': 500_000, 'B': 2_000_000, 'C': 2_000_000}
NEXT_EXP = {'A': 'B',    'B': 'C',        'C': None}

# reward threshold to count an eval period as "successful"
SUCCESS_THRESH = {'A': -50.0, 'B': -50.0, 'C': 500.0}


def _find_eval_path(exp):
    """return path to most recent evaluations.npz for exp, or None."""
    tagged = sorted(glob.glob(os.path.join('logs', f'exp_{exp}_[0-9][0-9][0-9]', 'evaluations.npz')))
    plain  = os.path.join('logs', f'exp_{exp}', 'evaluations.npz')
    candidates = tagged + ([plain] if os.path.exists(plain) else [])
    return candidates[-1] if candidates else None


def _next_started(exp):
    """check whether the next experiment has any eval log."""
    nxt = NEXT_EXP[exp]
    if nxt is None:
        return False
    return _find_eval_path(nxt) is not None


def bar(pct, width=30):
    filled = int(width * pct / 100)
    return '[' + '\u2588' * filled + '\u2591' * (width - filled) + f'] {pct:5.1f}%'


def elapsed(t):
    s = int(time.time() - t)
    return f'{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}'


def eta(steps, total, t0):
    if steps == 0:
        return '--:--:--'
    rate      = steps / (time.time() - t0)
    remaining = (total - steps) / rate
    return f'{int(remaining//3600):02d}:{int((remaining%3600)//60):02d}:{int(remaining%60):02d}'


def time_since_best(best_idx, total_evals, steps, t0):
    # estimate wall time of the best eval by interpolating through training
    if total_evals <= 1:
        return 'just now'
    frac_through   = best_idx / (total_evals - 1)
    elapsed_total  = time.time() - t0
    best_wall_time = t0 + frac_through * elapsed_total
    ago            = time.time() - best_wall_time
    if ago < 60:
        return f'{int(ago)}s ago'
    elif ago < 3600:
        return f'{int(ago//60)}m {int(ago%60)}s ago'
    else:
        return f'{int(ago//3600)}h {int((ago%3600)//60)}m ago'


while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    print('=' * 60)
    if FINETUNE_MODE:
        print(f'  fine-tune monitor (exp C)   training time: {elapsed(TRAIN_START)}')
    else:
        print(f'  RL training monitor   training time: {elapsed(TRAIN_START)}')
    print('=' * 60)

    active_exps = ('C',) if FINETUNE_MODE else ('A', 'B', 'C')
    for exp in active_exps:
        p = _find_eval_path(exp)
        print(f'\n  exp {exp}', end='')

        if p is None:
            print('  [ not started ]')
            continue

        d       = np.load(p)
        steps   = d['timesteps'][-1]
        results = d['results']          # shape: (n_evals, n_eval_eps)
        means   = results.mean(axis=1)
        recent  = means[-1]
        best    = means.max()
        best_idx = int(means.argmax())

        # successful eval periods (mean reward above threshold)
        thresh      = SUCCESS_THRESH[exp]
        n_success   = int((means >= thresh).sum())
        total_evals = len(means)

        # how long ago was the best found
        since_best = time_since_best(best_idx, total_evals, steps, TRAIN_START)

        # total steps for this run
        if FINETUNE_MODE and FINETUNE_BUDGET is not None:
            exp_total = FINETUNE_BUDGET
        else:
            exp_total = TOTAL[exp]

        done = False if FINETUNE_MODE else _next_started(exp)
        pct  = 100.0 if done else min(99.0, steps / exp_total * 100)

        status = ' [ done ]' if done else ''
        print(f'  {steps/1e6:.2f}M / {exp_total/1e6:.1f}M steps{status}')
        print(f'  {bar(pct)}')
        print(f'  recent: {recent:>10.1f}   best: {best:>10.1f}   ({since_best})')
        print(f'  successful evals: {n_success}/{total_evals}  ({100*n_success/max(total_evals,1):.0f}%)')
        if not done:
            print(f'  ETA: {eta(steps, exp_total, TRAIN_START)}')
        else:
            print('  ETA: complete')

    print('\n' + '=' * 60)
    print('  refreshing in 30s...  ctrl+c to quit')
    time.sleep(30)
