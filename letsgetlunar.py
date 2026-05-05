# letsgetlunar.py
# project 3 - RL orbital insertion
# alex crespo | 2026
# usage: python letsgetlunar.py

import time
import base64
import csv
import datetime
import glob
import json
import os
import platform
import queue
import socket
import subprocess
import sys
import threading

import numpy as np

ROOT    = os.path.dirname(os.path.abspath(__file__))
LOGS    = os.path.join(ROOT, 'logs')
MODELS  = os.path.join(ROOT, 'models')
SCRIPTS = os.path.join(ROOT, 'scripts')

MACHINE = 'main'  # set by _select_machine() at startup

os.chdir(ROOT)
os.makedirs(LOGS,   exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

STEPS_PER_HOUR = 850_000
SNAP           = 16_384

COLLAB_PORT = 7777
MACHINE_IPS = {
    'main':   '192.168.4.56',
    'backup': '192.168.4.55',
}

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


# ── machine selection ────────────────────────────────────────────────────────

def _select_machine():
    """ask which machine this is and set the global MACHINE variable."""
    global MACHINE
    print('which machine is this?')
    print('  1. main (windows PC)')
    print('  2. backup (macbook air, ubuntu)')
    while True:
        choice = input('enter 1 or 2: ').strip()
        if choice == '1':
            MACHINE = 'main'
            break
        if choice == '2':
            MACHINE = 'backup'
            break
        print('  invalid choice. enter 1 or 2.')

    if MACHINE == 'backup':
        print()
        print('backup machine selected (macbook air, ubuntu)')
        print()
        print('before continuing, make sure you have:')
        print('  at least 8 GB RAM free (SAC replay buffer + model weights use ~4-6 GB)')
        print('  at least 4 CPU cores available (training is CPU bound, no GPU needed)')
        print('  python 3.10 or higher')
        print('  the following packages installed:')
        print()
        print('    pip install stable-baselines3[extra] gymnasium numpy matplotlib')
        print()
        print('  pull the latest repo before training:')
        print('    git pull origin main')
        print()
        input('press enter to continue...')
        print()


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

    if MACHINE == 'backup':
        try:
            subprocess.Popen(['bash', '-c', f'python3 {script}'], start_new_session=True)
        except Exception as exc:
            print(f'  note: could not launch monitor ({exc}). run scripts/check_progress.py manually.')
        return

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

    exploring_starts_C = True
    if MACHINE == 'backup':
        exploring_starts_C = False
        print('  backup mode: using fixed starts for exp C to match eval conditions')

    aborted = False
    try:
        run_experiments(budgets=budgets, exploring_starts_C=exploring_starts_C, machine=MACHINE)
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
        base_label = os.path.relpath(z, MODELS)
        if not np.isnan(best):
            label = f'{base_label}  (best: {best:.1f})'
        else:
            label = base_label
        rows.append((label, exp, best, n_success, z))

    print('\n  available models:')
    print(f"  {'#':>4}  {'model':<44}  {'exp':>4}  {'best reward':>11}  evals")
    print(f"  {'----':>4}  {'--------------------------------------------':<44}  {'----':>4}  {'-----------':>11}  -----")
    for i, (label, exp, best, n_success, z) in enumerate(rows, 1):
        best_str = f'{best:.1f}' if not np.isnan(best) else 'n/a'
        print(f'  {i:>4}  {label:<44}  {exp:>4}  {best_str:>11}  ({n_success})')

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


def _finetune_exp_c():
    print('\nfine-tune experiment C')

    # list only exp C models
    all_zips = _all_model_zips()
    c_zips = [z for z in all_zips if _exp_from_zip(z) == 'C']
    if not c_zips:
        print('\n  no exp C models found in models/. train a model first.')
        return

    rows = []
    for z in c_zips:
        d = _read_eval_log('C')
        if d is not None:
            means = d['results'].mean(axis=1)
            best  = float(means.max())
            n_evals = len(means)
        else:
            best    = float('nan')
            n_evals = 0
        base_label = os.path.relpath(z, MODELS)
        if not np.isnan(best):
            label = f'{base_label}  (best: {best:.1f})'
        else:
            label = base_label
        rows.append((label, best, n_evals, z))

    print('\n  available exp C models:')
    print(f"  {'#':>4}  {'model':<44}  {'best reward':>11}  evals")
    print(f"  {'----':>4}  {'--------------------------------------------':<44}  {'-----------':>11}  -----")
    for i, (label, best, n_evals, z) in enumerate(rows, 1):
        best_str = f'{best:.1f}' if not np.isnan(best) else 'n/a'
        print(f'  {i:>4}  {label:<44}  {best_str:>11}  ({n_evals})')

    choice = input('\n  pick a model by number (or enter to cancel): ').strip()
    if not choice:
        return
    try:
        idx = int(choice) - 1
        label, best, n_evals, model_path = rows[idx]
    except (ValueError, IndexError):
        print('  invalid choice.')
        return

    hours_str = input('  how many additional hours to train? (enter for default 2M steps): ').strip()
    if hours_str == '':
        budget = 2_000_000
    else:
        try:
            budget = int(float(hours_str) * STEPS_PER_HOUR)
            budget = max(budget, 50_000)
        except ValueError:
            print('  invalid input. using default 2M steps.')
            budget = 2_000_000

    print(f'\n  warm-starting from: {label}')
    print(f'  additional steps:   {budget:,}')

    # write sentinel so monitor knows which tag dir to watch
    sentinel = os.path.join(LOGS, '.finetune_mode')
    sys.path.insert(0, SCRIPTS)
    from train_agent import finetune_exp_c, next_model_tag  # noqa: PLC0415
    next_tag = next_model_tag('C', machine=MACHINE)
    with open(sentinel, 'w') as f:
        f.write(f'{budget} {next_tag}')

    print('\n  launching training monitor in background...')
    _launch_monitor()

    exploring_starts_C = MACHINE != 'backup'

    aborted = False
    try:
        final_path, best = finetune_exp_c(
            model_path, budget=budget,
            exploring_starts_C=exploring_starts_C,
            machine=MACHINE, tag=next_tag,
        )
        r_s = f'{best:.1f}' if not np.isnan(best) else '?'
        print(f'\n  saved: {os.path.basename(final_path)}  (best reward: {r_s})')
    except KeyboardInterrupt:
        print('\n  fine-tuning interrupted.')
        aborted = True
    finally:
        if os.path.exists(sentinel):
            os.remove(sentinel)

    _log_attempt(aborted, budgets={'A': 0, 'B': 0, 'C': budget})


# ── collab mode ───────────────────────────────────────────────────────────────

def _nan2none(x):
    return None if (isinstance(x, float) and np.isnan(x)) else x


def _reward_from_name(name):
    """parse the best-reward integer embedded in a model filename.

    Filenames produced by the new naming convention contain a ``_rNNN`` segment,
    e.g. ``exp_C_main_003_r810_success.zip`` → 810.0.
    Returns ``float('nan')`` when no such segment is found.
    """
    stem = os.path.basename(name).replace('.zip', '')
    for part in stem.split('_'):
        if part.startswith('r') and len(part) > 1:
            try:
                return float(part[1:])
            except ValueError:
                pass
    return float('nan')


class _CollabSession:
    """manages the socket connection and shared model pot for collab mode."""

    def __init__(self, machine, peer_ip):
        self.machine      = machine
        self.peer_ip      = peer_ip       # may be changed by UI thread at any time
        self.peer_machine = None
        self.connected    = False
        self.stopped      = False
        self.status       = f'trying {peer_ip}:{COLLAB_PORT}...'

        self._pot       = {}             # name -> {best_reward, local_path, being_trained}
        self._pot_lock  = threading.Lock()
        self._msg_q     = queue.Queue()
        self._sock      = None
        self._sock_lock = threading.Lock()

        self._thread = threading.Thread(target=self._connect_loop, daemon=True)
        self._thread.start()

    # ── public api ────────────────────────────────────────────────────────────

    def stop(self):
        self.stopped = True
        with self._sock_lock:
            s, self._sock = self._sock, None
        if s:
            try: s.close()
            except OSError: pass

    def send(self, msg):
        with self._sock_lock:
            s = self._sock
        if not s:
            return
        try:
            s.sendall((json.dumps(msg) + '\n').encode())
        except OSError:
            self._drop()

    def get_messages(self):
        msgs = []
        while True:
            try: msgs.append(self._msg_q.get_nowait())
            except queue.Empty: break
        return msgs

    def pot_snapshot(self):
        with self._pot_lock:
            return {n: dict(i) for n, i in self._pot.items()}

    def set_pot(self, name, **fields):
        with self._pot_lock:
            self._pot.setdefault(name, {}).update(fields)

    def best_untrained(self):
        with self._pot_lock:
            cands = [
                (n, dict(i)) for n, i in self._pot.items()
                if not i.get('being_trained')
                and not np.isnan(float(i.get('best_reward', float('nan'))))
            ]
        if not cands:
            return None, None
        return max(cands, key=lambda x: x[1].get('best_reward', float('-inf')))

    # ── internals ─────────────────────────────────────────────────────────────

    def _drop(self):
        with self._sock_lock:
            self._sock = None
        self.connected = False

    def _connect_loop(self):
        server = None
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('0.0.0.0', COLLAB_PORT))
            server.listen(1)
            server.settimeout(2)
        except OSError as e:
            self.status = f'port {COLLAB_PORT} unavailable ({e})'
            server = None

        while not self.stopped:
            sock = self._try_out()
            if sock is None and server:
                sock = self._try_in(server)
            if sock is None:
                self.status = f'trying {self.peer_ip}:{COLLAB_PORT}...'
                time.sleep(1)
                continue

            with self._sock_lock:
                self._sock = sock
            try:
                self._protocol(sock)
            except Exception:
                pass
            self._drop()
            if not self.stopped:
                self.status = 'disconnected — reconnecting...'
                time.sleep(2)

        if server:
            try: server.close()
            except OSError: pass

    def _try_out(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((self.peer_ip, COLLAB_PORT))
            return s
        except (OSError, socket.timeout):
            return None

    def _try_in(self, server):
        try:
            c, _ = server.accept()
            return c
        except (socket.timeout, OSError):
            return None

    def _protocol(self, sock):
        sock.settimeout(20)
        with self._pot_lock:
            local = [
                {'name': n, 'best_reward': _nan2none(i['best_reward'])}
                for n, i in self._pot.items() if i.get('local_path')
            ]
        sock.sendall((json.dumps({'type': 'hello', 'machine': self.machine}) + '\n').encode())
        sock.sendall((json.dumps({'type': 'model_list', 'models': local}) + '\n').encode())

        buf = ''
        while not self.stopped:
            try:
                chunk = sock.recv(131072)
                if not chunk:
                    break
                buf += chunk.decode('utf-8', errors='replace')
            except socket.timeout:
                try: sock.sendall((json.dumps({'type': 'ping'}) + '\n').encode())
                except OSError: break
                continue
            except OSError:
                break

            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    self._dispatch(json.loads(line), sock)
                except Exception:
                    pass

    def _dispatch(self, msg, sock):
        t = msg.get('type')
        if t == 'hello':
            self.peer_machine = msg.get('machine', 'peer')
            self.connected    = True
            self.status       = f'connected to {self.peer_machine} @ {self.peer_ip}'
            self._msg_q.put(msg)

        elif t == 'model_list':
            with self._pot_lock:
                for m in msg.get('models', []):
                    n = m['name']
                    r = m.get('best_reward')
                    r = float('nan') if r is None else float(r)
                    cur   = self._pot.get(n, {})
                    cur_r = float(cur.get('best_reward', float('nan')))
                    if np.isnan(cur_r) or (not np.isnan(r) and r > cur_r):
                        self._pot.setdefault(n, {}).update({
                            'best_reward':   r,
                            'local_path':    cur.get('local_path'),
                            'being_trained': cur.get('being_trained', False),
                        })
            self._msg_q.put(msg)

        elif t == 'model_request':
            self._serve(sock, msg.get('name'))

        elif t == 'model_data':
            self._save(msg)
            self._msg_q.put(msg)

        elif t == 'training_start':
            n = msg.get('model')
            with self._pot_lock:
                if n in self._pot:
                    self._pot[n]['being_trained'] = True
            self._msg_q.put(msg)

        elif t == 'training_done':
            old = msg.get('old_model')
            new = msg.get('new_model')
            r   = msg.get('best_reward')
            r   = float('nan') if r is None else float(r)
            with self._pot_lock:
                if old and old in self._pot:
                    self._pot[old]['being_trained'] = False
                if new:
                    self._pot[new] = {'best_reward': r, 'local_path': None, 'being_trained': False}
            self._msg_q.put(msg)

        elif t == 'ping':
            try: sock.sendall((json.dumps({'type': 'pong'}) + '\n').encode())
            except OSError: pass

    def _serve(self, sock, name):
        if not name:
            return
        with self._pot_lock:
            info = dict(self._pot.get(name, {}))
        path = info.get('local_path')
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, 'rb') as f:
                data = base64.b64encode(f.read()).decode()
            sock.sendall((json.dumps({'type': 'model_data', 'name': name, 'data': data}) + '\n').encode())
        except Exception:
            pass

    def _save(self, msg):
        name = msg.get('name')
        data = msg.get('data')
        if not name or not data:
            return
        path = os.path.join(MODELS, name)
        try:
            with open(path, 'wb') as f:
                f.write(base64.b64decode(data))
            with self._pot_lock:
                self._pot.setdefault(name, {})['local_path'] = path
        except Exception as e:
            print(f'\n  failed to save {name}: {e}')


def _collab_scan_models():
    """return (name, path, best_reward) for all model zips, preferring _best.zip."""
    all_zips = glob.glob(os.path.join(MODELS, '**', '*.zip'), recursive=True)
    by_base  = {}
    for z in all_zips:
        base_name = os.path.basename(z)
        if 'best_model' in base_name:
            continue
        is_best = base_name.endswith('_best.zip')
        base_key = base_name.replace('_best.zip', '').replace('.zip', '')
        existing = by_base.get(base_key)
        if existing is None or is_best:
            by_base[base_key] = (z, is_best)

    rows = []
    for base_key, (path, _) in by_base.items():
        exp  = _exp_from_zip(path)
        d    = _read_eval_log(exp) if exp != '?' else None
        best = float(d['results'].mean(axis=1).max()) if d is not None else float('nan')
        # fall back to reward encoded in the filename (new naming convention)
        if np.isnan(best):
            best = _reward_from_name(path)
        rows.append((os.path.basename(path), path, best))
    return rows


def _collab_print_pot(sess):
    pot = sess.pot_snapshot()
    if not pot:
        print('  (pot is empty)')
        return
    names = sorted(pot, key=lambda n: pot[n].get('best_reward', float('-inf')), reverse=True)
    print(f'  {"model":<42}  {"best reward":>11}  {"location":>8}  status')
    print(f'  {"─"*42}  {"─"*11}  {"─"*8}  ──────')
    for n in names:
        i   = pot[n]
        r   = i.get('best_reward', float('nan'))
        r_s = f'{r:.1f}' if not np.isnan(r) else 'n/a'
        loc = 'local' if i.get('local_path') else 'peer'
        st  = 'training' if i.get('being_trained') else ''
        print(f'  {n:<42}  {r_s:>11}  {loc:>8}  {st}')


def _collab_wait_for_connect(sess):
    """show connecting UI until connected or user quits. returns True if connected."""
    cmd_q = queue.Queue()

    def _reader():
        while not sess.stopped and not sess.connected:
            try: cmd_q.put(sys.stdin.readline().strip())
            except Exception: break

    threading.Thread(target=_reader, daemon=True).start()

    last_status = ''
    while not sess.stopped and not sess.connected:
        while True:
            try: cmd = cmd_q.get_nowait()
            except queue.Empty: break
            parts = cmd.split(None, 1)
            verb  = parts[0].lower() if parts else ''
            if verb == 'q':
                sess.stop()
                print('  cancelled.')
                return False
            elif verb == 'ip':
                new_ip = parts[1].strip() if len(parts) > 1 else input('  new ip: ').strip()
                if new_ip:
                    sess.peer_ip = new_ip
                    sess.status  = f'trying {new_ip}:{COLLAB_PORT}...'
                    print(f'  → switching to {new_ip}')
            elif verb:
                print('  commands: ip <address>, q')

        if sess.status != last_status:
            print(f'  {sess.status}')
            last_status = sess.status
        time.sleep(0.4)

    return sess.connected


def _collab_train_loop(sess):
    """coordinate fine-tuning with peer — picks best model, trains, shares results."""
    done_q   = queue.Queue()
    cmd_q    = queue.Queue()
    current  = None
    training = False

    def _reader():
        while not sess.stopped:
            try: cmd_q.put(sys.stdin.readline().strip())
            except Exception: break

    threading.Thread(target=_reader, daemon=True).start()

    def _train(name, path, budget=2_000_000):
        try:
            sys.path.insert(0, SCRIPTS)
            from train_agent import finetune_exp_c, next_model_tag  # noqa
            tag      = next_model_tag('C', machine=MACHINE)
            sentinel = os.path.join(LOGS, '.finetune_mode')
            with open(sentinel, 'w') as f:
                f.write(f'{budget} {tag}')
            final_path, best = finetune_exp_c(
                path.replace('.zip', ''),
                budget=budget,
                exploring_starts_C=(MACHINE == 'main'),
                machine=MACHINE,
                tag=tag,
            )
            done_q.put({'tag': tag, 'path': final_path, 'best': best})
        except Exception as e:
            done_q.put({'error': str(e)})
        finally:
            s = os.path.join(LOGS, '.finetune_mode')
            if os.path.exists(s):
                os.remove(s)

    while not sess.stopped:
        # handle messages from peer
        for msg in sess.get_messages():
            mt = msg.get('type')
            if mt == 'model_list':
                print('\n  peer updated model list')
                _collab_print_pot(sess)
            elif mt == 'training_start':
                print(f'  peer started training: {msg.get("model")}')
            elif mt == 'training_done':
                r   = msg.get('best_reward')
                r_s = f'{float(r):.1f}' if r is not None else '?'
                print(f'\n  peer done: {msg.get("new_model")}  best={r_s}')
                _collab_print_pot(sess)
            elif mt == 'model_data':
                n = msg.get('name')
                if n:
                    print(f'\n  received {n} from peer')
                    sess.set_pot(n, being_trained=False)

        # user commands
        while True:
            try: cmd = cmd_q.get_nowait()
            except queue.Empty: break
            v = cmd.strip().lower()
            if v == 'q':
                print('  stopping collab...')
                sess.stop()
                return
            elif v == 'pot':
                _collab_print_pot(sess)
            elif v == 'status':
                print(f'  {sess.status}')
                if current:
                    print(f'  training: {current}')
            elif v:
                print('  commands: pot, status, q')

        # training finished?
        try:
            result = done_q.get_nowait()
            training = False
            if 'error' in result:
                print(f'\n  training error: {result["error"]}')
                if current:
                    sess.set_pot(current, being_trained=False)
            else:
                tag      = result['tag']
                new_path = result['path']
                best     = result['best']
                new_name = os.path.basename(new_path)
                r_s      = f'{best:.1f}' if not np.isnan(best) else '?'
                print(f'\n  training done: {new_name}  (best: {r_s})')
                sess.set_pot(new_name, best_reward=best, local_path=new_path, being_trained=False)
                if current:
                    sess.set_pot(current, being_trained=False)
                sess.send({
                    'type': 'training_done',
                    'old_model':   current,
                    'new_model':   new_name,
                    'best_reward': _nan2none(best),
                })
                # push updated model list to peer
                pot   = sess.pot_snapshot()
                local = [{'name': n, 'best_reward': _nan2none(i['best_reward'])}
                         for n, i in pot.items() if i.get('local_path')]
                sess.send({'type': 'model_list', 'models': local})
                current = None
                _collab_print_pot(sess)
        except queue.Empty:
            pass

        # start training if idle
        if not training:
            name, info = sess.best_untrained()
            if name and info:
                local_path = info.get('local_path')
                if local_path and os.path.exists(local_path):
                    training = True
                    current  = name
                    sess.set_pot(name, being_trained=True)
                    r_disp = info.get('best_reward', float('nan'))
                    print(f'\n  starting fine-tune: {name}  (best: {r_disp:.1f})')
                    sess.send({'type': 'training_start', 'model': name})
                    threading.Thread(target=_train, args=(name, local_path), daemon=True).start()
                elif local_path is None:
                    # mark so we don't keep re-requesting every 0.5s
                    sess.set_pot(name, being_trained=True)
                    print(f'\n  requesting {name} from peer...')
                    sess.send({'type': 'model_request', 'name': name})

        time.sleep(0.5)


def _collab_mode():
    """peer-to-peer collab training entry point."""
    global MACHINE
    print()
    print('  collab mode')
    print(f'  main machine   (windows) → {MACHINE_IPS["main"]}')
    print(f'  backup machine (ubuntu)  → {MACHINE_IPS["backup"]}')
    print()
    print('  which machine is this?  1. main   2. backup')
    while True:
        ch = input('  pick 1 or 2: ').strip()
        if ch == '1':
            MACHINE = 'main'
            break
        if ch == '2':
            MACHINE = 'backup'
            break
        print('  enter 1 or 2.')

    peer_ip = MACHINE_IPS['backup' if MACHINE == 'main' else 'main']
    sess    = _CollabSession(MACHINE, peer_ip)

    for name, path, best in _collab_scan_models():
        sess.set_pot(name, best_reward=best, local_path=path, being_trained=False)

    print(f'\n  looking for peer at {peer_ip}:{COLLAB_PORT}...')
    print('  type "ip <address>" to change peer IP   "q" to cancel\n')

    if not _collab_wait_for_connect(sess):
        return

    print(f'\n  connected!  peer: {sess.peer_machine} @ {sess.peer_ip}')
    time.sleep(0.6)  # let model_list messages arrive and merge

    print('\n  current model pot:')
    _collab_print_pot(sess)
    print()
    print('  training starting — type "pot" to refresh, "status" for connection info, "q" to quit')
    print()

    _collab_train_loop(sess)
    sess.stop()
    print('  collab mode ended.')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print('letsgetlunar')
    print('project 3 - RL orbital insertion')
    print('alex crespo | 2026')
    print()

    grader = input('  are you a grader? (y/n): ').strip().lower()
    if grader == 'y':
        print()
        print('  hello! glad you\'re here.')
        print()
        print('  would you like to:')
        print('    1. train a model for experiments A, B, and C  (hope you have time — it took ~16 hours total!)')
        print('    2. run one of my pre-trained models  (higher reward = better performance)')
        print('    3. exit')
        print()
        grader_choice = input('  pick an option: ').strip()
        if grader_choice == '1':
            global MACHINE
            MACHINE = 'main'
            _train_new_model()
        elif grader_choice == '2':
            _run_pretrained_model()
        else:
            print('  see you later!')
        return

    collab = input('  would you like to go into collab mode? (y/n): ').strip().lower()
    if collab == 'y':
        _collab_mode()
        return

    _select_machine()

    MENU = [
        ('train a new model (experiments A, B, C from scratch)', _train_new_model),
        ('run a pre-trained model',                              _run_pretrained_model),
        ('fine-tune experiment C (warm-start from existing model)', _finetune_exp_c),
        ('quit',                                                 None),
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
