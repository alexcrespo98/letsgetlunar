# scripts/gui_monitor.py - live matplotlib GUI monitor for letsgetlunar training
# alex crespo | 2026
# usage: python scripts/gui_monitor.py
# also launched automatically by _launch_monitor() in letsgetlunar.py

import glob
import os
import time

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FuncAnimation
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError as _e:
    print(f'  gui_monitor: matplotlib not available ({_e}). run pip install matplotlib.')
    HAS_MATPLOTLIB = False

import numpy as np

# paths

ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LOGS     = os.path.join(ROOT, 'logs')
SENTINEL = os.path.join(LOGS, '.finetune_mode')
PARALLEL_SENTINEL = os.path.join(LOGS, '.parallel_mode')

SUCCESS_THRESH = 500.0
STEPS_PER_SEC  = 850_000 / 3600.0  # ~236 steps/sec, used to project wall-clock time

# record script start so we can estimate elapsed time
_T0 = time.time()


# data loading

def _find_eval_path():
    """return (tag, path) for the most recent evaluations.npz to display."""
    # finetune mode: read sentinel for exact tag
    if os.path.exists(SENTINEL):
        try:
            with open(SENTINEL) as sf:
                parts = sf.read().strip().split()
                tag = parts[1] if len(parts) > 1 else None
        except (OSError, IndexError):
            tag = None
        if tag:
            p = os.path.join(LOGS, tag, 'evaluations.npz')
            if os.path.exists(p):
                return tag, p

    # fall back: most recent tagged log dir (any exp)
    candidates = []
    for pattern in ('exp_C_[0-9][0-9][0-9]', 'exp_Cstar_*', 'exp_B_[0-9][0-9][0-9]', 'exp_A_[0-9][0-9][0-9]'):
        for d in sorted(glob.glob(os.path.join(LOGS, pattern))):
            p = os.path.join(d, 'evaluations.npz')
            if os.path.exists(p):
                candidates.append((os.path.getmtime(p), os.path.basename(d), p))
    if candidates:
        candidates.sort()
        _, tag, p = candidates[-1]
        return tag, p
    return None, None


def _load_data():
    """return (tag, timesteps, mean_rewards, file_mtime) or (None, ...)."""
    tag, path = _find_eval_path()
    if path is None:
        return None, None, None, None
    try:
        d      = np.load(path)
        steps  = d['timesteps'].astype(float)
        means  = d['results'].mean(axis=1)
        mtime  = os.path.getmtime(path)
        return tag, steps, means, mtime
    except Exception:
        return None, None, None, None


def _steps_to_wallclock(steps, mtime):
    """convert a steps array to wall-clock datetime objects.

    Strategy: assume the last eval point happened at file mtime.
    Project earlier points backward using STEPS_PER_SEC.
    """
    import datetime
    t_last = mtime
    wall   = []
    for s in steps:
        dt_back = (steps[-1] - s) / STEPS_PER_SEC
        ts = t_last - dt_back
        wall.append(datetime.datetime.fromtimestamp(ts))
    return wall


# plot setup

def _format_xaxis(ax, wall_times):
    """choose HH:MM vs date+time based on run duration."""
    if not wall_times:
        return
    span = (wall_times[-1] - wall_times[0]).total_seconds()
    if span < 7200:  # under 2 hours: just HH:MM
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')


WORKER_COLORS = [
    '#7ec8e3', '#f4a261', '#a8d8a8', '#d4a8d4', '#f4d4a8',
    '#a8c8f4', '#f4a8a8', '#a8f4d4', '#d4f4a8', '#f4f4a8',
]

MAX_LABEL_LENGTH = 50  # max chars for config summary in legend labels


def _run_parallel_gui(tags, budget, sweep_configs=None):
    """parallel mode GUI: one colored line per worker + CPU bar chart inset.

    When *sweep_configs* is provided (dict mapping tag → config summary string)
    the chart shows sweep-specific labels and title instead of plain tags.
    """
    try:
        import psutil
        HAS_PSUTIL = True
    except ImportError:
        HAS_PSUTIL = False

    n = len(tags)
    colors = WORKER_COLORS[:n]

    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor('#1e1e2e')

    if HAS_PSUTIL:
        ax_main = fig.add_axes([0.07, 0.12, 0.65, 0.78])
        ax_cpu  = fig.add_axes([0.76, 0.12, 0.20, 0.78])
    else:
        ax_main = fig.add_axes([0.07, 0.12, 0.88, 0.78])
        ax_cpu  = None

    for ax in ([ax_main] + ([ax_cpu] if ax_cpu else [])):
        ax.set_facecolor('#1e1e2e')
        for spine in ax.spines.values():
            spine.set_color('#555577')
        ax.tick_params(colors='#ccccdd')
        ax.xaxis.label.set_color('#ccccdd')
        ax.yaxis.label.set_color('#ccccdd')
        ax.title.set_color('#ccccdd')

    ax_main.axhline(SUCCESS_THRESH, color='#f4a261', linestyle='--',
                    linewidth=1.2, label=f'success ({SUCCESS_THRESH:.0f})', zorder=1)
    ax_main.set_xlabel('steps (M)')
    ax_main.set_ylabel('mean eval reward')

    scats  = []
    lines  = []
    for tag, color in zip(tags, colors):
        if sweep_configs:
            summary = sweep_configs.get(tag, '')
            label = f'{tag}  {summary[:MAX_LABEL_LENGTH]}' if summary else tag
        else:
            label = tag
        sc = ax_main.scatter([], [], s=14, color=color, zorder=3, alpha=0.7)
        ln, = ax_main.plot([], [], color=color, linewidth=1.5,
                           drawstyle='steps-post', zorder=2, label=label)
        scats.append(sc)
        lines.append(ln)

    ax_main.legend(facecolor='#2d2d3f', edgecolor='#555577', labelcolor='#ccccdd',
                   fontsize=7, loc='upper left')

    if ax_cpu:
        ax_cpu.set_title('CPU / core', color='#ccccdd', fontsize=8)
        ax_cpu.set_xlabel('load %', color='#ccccdd', fontsize=7)

    _last_mtimes = [None] * n

    def _update(_frame):
        changed = False
        for i, (tag, scat, line) in enumerate(zip(tags, scats, lines)):
            p = os.path.join(LOGS, tag, 'evaluations.npz')
            if not os.path.exists(p):
                continue
            try:
                mtime = os.path.getmtime(p)
                if mtime == _last_mtimes[i]:
                    continue
                _last_mtimes[i] = mtime
                d = np.load(p)
                steps = d['timesteps'].astype(float) / 1e6
                means = d['results'].mean(axis=1)
                best_so_far = np.maximum.accumulate(means)
                scat.set_offsets(np.column_stack([steps, means]))
                line.set_xdata(steps)
                line.set_ydata(best_so_far)
                changed = True
            except Exception:
                continue

        if changed:
            ax_main.relim()
            ax_main.autoscale_view()

        if ax_cpu and HAS_PSUTIL:
            try:
                import psutil
                percore = psutil.cpu_percent(percpu=True)
                ax_cpu.clear()
                ax_cpu.set_facecolor('#1e1e2e')
                for spine in ax_cpu.spines.values():
                    spine.set_color('#555577')
                ax_cpu.tick_params(colors='#ccccdd', labelsize=6)
                nc = len(percore)
                y_pos = list(range(nc))
                bar_colors = ['#e74c3c' if p > 80 else '#f4a261' if p > 60 else '#a8d8a8'
                              for p in percore]
                ax_cpu.barh(y_pos, percore, color=bar_colors, height=0.7)
                ax_cpu.set_xlim(0, 100)
                ax_cpu.set_yticks(y_pos)
                ax_cpu.set_yticklabels([f'c{j}' for j in range(nc)], fontsize=5)
                ax_cpu.set_xlabel('load %', color='#ccccdd', fontsize=7)
                ax_cpu.set_title('CPU / core', color='#ccccdd', fontsize=8)
                ax_cpu.xaxis.label.set_color('#ccccdd')
            except Exception:
                pass

        elapsed_s = int(time.time() - _T0)
        h, rem = divmod(elapsed_s, 3600)
        m, s   = divmod(rem, 60)
        if sweep_configs:
            title = (f'sweep exp C*   {n} configs   elapsed: {h:02d}:{m:02d}:{s:02d}')
        else:
            title = (f'parallel exp C*   {n} workers   elapsed: {h:02d}:{m:02d}:{s:02d}')
        ax_main.set_title(title, color='#ccccdd', pad=8, fontsize=9)
        fig.canvas.draw_idle()

    anim = FuncAnimation(fig, _update, interval=15_000, cache_frame_data=False)
    _update(0)
    plt.show()
    return anim


def _run_single_gui():
    """single-worker GUI (original behaviour)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')
    for spine in ax.spines.values():
        spine.set_color('#555577')
    ax.tick_params(colors='#ccccdd')
    ax.xaxis.label.set_color('#ccccdd')
    ax.yaxis.label.set_color('#ccccdd')
    ax.title.set_color('#ccccdd')

    scat   = ax.scatter([], [], s=18, color='#7ec8e3', zorder=3, label='eval reward')
    line_b,= ax.plot([], [], color='#a8d8a8', linewidth=1.5, drawstyle='steps-post',
                     label='best so far', zorder=2)
    thresh_line = ax.axhline(SUCCESS_THRESH, color='#f4a261', linestyle='--',
                             linewidth=1.2, label=f'success ({SUCCESS_THRESH:.0f})', zorder=1)
    ax.set_xlabel('wall-clock time')
    ax.set_ylabel('mean eval reward')
    ax.legend(facecolor='#2d2d3f', edgecolor='#555577', labelcolor='#ccccdd',
              fontsize=8, loc='upper left')

    _last_tag   = [None]
    _last_mtime = [None]

    def _update(_frame):
        tag, steps, means, mtime = _load_data()

        if steps is None or len(steps) == 0:
            ax.set_title('waiting for training data…', color='#ccccdd', pad=8)
            fig.canvas.draw_idle()
            return

        # skip redraw if nothing changed
        if tag == _last_tag[0] and mtime == _last_mtime[0]:
            return
        _last_tag[0]   = tag
        _last_mtime[0] = mtime

        wall = _steps_to_wallclock(steps, mtime)

        # scatter: all eval points
        xdata = mdates.date2num(wall)
        scat.set_offsets(np.column_stack([xdata, means]))

        # best-so-far step line
        best_so_far = np.maximum.accumulate(means)
        line_b.set_xdata(xdata)
        line_b.set_ydata(best_so_far)

        ax.relim()
        ax.autoscale_view()
        _format_xaxis(ax, wall)

        best   = float(means.max())
        recent = float(means[-1])
        n_steps = int(steps[-1])
        elapsed = int(time.time() - _T0)
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        ax.set_title(
            f'{tag}   best: {best:.1f}   recent: {recent:.1f}   '
            f'steps: {n_steps/1e6:.2f}M   elapsed: {h:02d}:{m:02d}:{s:02d}',
            color='#ccccdd', pad=8, fontsize=9
        )

        fig.tight_layout()
        fig.canvas.draw_idle()

    anim = FuncAnimation(fig, _update, interval=10_000, cache_frame_data=False)  # 10 s refresh

    # run an initial update immediately
    _update(0)

    plt.show()

    # keep anim alive (prevents garbage collection closing the window)
    return anim


# animation

def main():
    if not HAS_MATPLOTLIB:
        return

    # detect parallel mode
    parallel_tags = []
    parallel_budget = 2_000_000
    sweep_configs_dict = None
    if os.path.exists(PARALLEL_SENTINEL):
        try:
            import json as _json
            with open(PARALLEL_SENTINEL) as _pf:
                _pd = _json.load(_pf)
                parallel_tags = _pd.get('workers', [])
                parallel_budget = _pd.get('budget', 2_000_000)
                sweep_configs_dict = _pd.get('sweep_configs') or None
        except Exception:
            parallel_tags = []

    if parallel_tags:
        _run_parallel_gui(parallel_tags, parallel_budget, sweep_configs=sweep_configs_dict)
    else:
        _run_single_gui()


if __name__ == '__main__':
    _anim = main()
