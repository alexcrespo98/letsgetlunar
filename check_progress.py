# live_plot.py - terminal training monitor
import numpy as np, os, time

TRAIN_START = None
for exp in ('A','B','C'):
    p = f'logs/exp_{exp}/evaluations.npz'
    if os.path.exists(p):
        TRAIN_START = os.path.getmtime(p)
        break
if TRAIN_START is None:
    TRAIN_START = time.time()

TOTAL    = {'A': 500_000, 'B': 2_000_000, 'C': 2_000_000}
NEXT_EXP = {'A': 'B',    'B': 'C',        'C': None}

# reward threshold to count an eval period as "successful"
SUCCESS_THRESH = {'A': -50.0, 'B': -50.0, 'C': 500.0}

def bar(pct, width=30):
    filled = int(width * pct / 100)
    return '[' + '█'*filled + '░'*(width-filled) + f'] {pct:5.1f}%'

def elapsed(t):
    s = int(time.time() - t)
    return f'{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}'

def eta(steps, total, t0):
    if steps == 0: return '--:--:--'
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
    os.system('cls')
    print('='*60)
    print(f'  RL TRAINING MONITOR   training time: {elapsed(TRAIN_START)}')
    print('='*60)

    for exp in ('A','B','C'):
        p = f'logs/exp_{exp}/evaluations.npz'
        print(f'\n  EXP {exp}', end='')

        if not os.path.exists(p):
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

        # done check
        nxt  = NEXT_EXP[exp]
        done = (nxt is not None and os.path.exists(f'logs/exp_{nxt}/evaluations.npz'))
        pct  = 100.0 if done else min(99.0, steps / TOTAL[exp] * 100)

        status = ' [ DONE ]' if done else ''
        print(f'  {steps/1e6:.2f}M / {TOTAL[exp]/1e6:.1f}M steps{status}')
        print(f'  {bar(pct)}')
        print(f'  recent: {recent:>10.1f}   best: {best:>10.1f}   ({since_best})')
        print(f'  successful evals: {n_success}/{total_evals}  ({100*n_success/max(total_evals,1):.0f}%)')
        if not done:
            print(f'  ETA: {eta(steps, TOTAL[exp], TRAIN_START)}')
        else:
            print(f'  ETA: complete')

    print('\n' + '='*60)
    print('  refreshing in 30s...  ctrl+c to quit')
    time.sleep(30)