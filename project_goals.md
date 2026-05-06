# project goals

not for submission. internal tracking only.
last updated: 2026-05-06

## overview

design an RL agent to control thrust angle β(t) for optimal lunar orbital insertion.
target orbit: 400 km circular orbit (alt ±20 km, Vr < 50 m/s, Vtan within 100 m/s of 1511 m/s).
deliverable: 2–4 page writeup summarizing reward functions, architectures, algorithms, and takeaways.

## project 2 baseline

NLP solver (scipy trust-constr) converges to optimal β(t) profile.
all 4 figures generated (beta profile, state history, polar trajectory, ground track).
output saved to output/ — optimal_trajectory.csv, optimal_beta_profile.csv, figs 1–4.
pipeline runner works end-to-end.

## RL infrastructure

interactive menu with 7 options (solo train, collab, run pretrained, fine-tune, quit, sweep, parallel).
time-budgeted training — input hours, auto-scales step budget per experiment using BUDGET_SHARE.
non-overwriting tagged model saves (exp_A_001, exp_C_main_003, exp_Cstar_main_w01_001, etc.).
attempt log — logs/attempt_log.csv records per-run stats (steps, best reward, success rate, notes).
auto-launch of check_progress.py and gui_monitor.py on training start.
early stopping via StopTrainingOnNoModelImprovement (wired in train_agent.py).
grader-friendly mode — simplified menu at startup for graders.

## experiments (run 1)

exp A — PPO, sparse reward, small net. no learning, confirmed as expected.
exp B — PPO, shaped reward, large net. no meaningful convergence, confirmed as expected.
exp C — SAC, multiobjective reward, medium net. best reward 789.4, partial progress.
all three model folders saved (ppo_sparse_small, ppo_shaped_large, sac_multiobjective_medium).
eval logs present at logs/exp_A, logs/exp_B, logs/exp_C.

## fine-tune / warm-start (option 4)

pick any saved exp C model and continue training from it.
configurable additional step budget.
tagged saves so warm-start runs don't overwrite originals.
still need to run a full 2M step warm-start from best C model once one exists.

## hyperparameter sweep (option 6)

originally 10 pre-built configs sampled via Latin hypercube across lr, net size, buffer, ent_coef, batch size, reward weights, gaussian altitude width.
replaced with 8 configs that bracket the current exp C parameters one dimension at a time (lr, net size, buffer, batch, ent_coef, gaussian width).
results logged to logs/sweep_results.csv after each config.
ranked summary + auto-suggestion of best config at end.
can immediately fine-tune with winning config (solo or collab).
still need to run a clean sweep to completion.

## parallel multi-core training (option 7)

spawns N worker processes (default: physical cores, detected via psutil — changed from phys_cores minus 2).
each worker gets a unique random seed for different exploration paths.
live status table refreshes every 30s showing steps, recent reward, best reward per worker.
all results ranked at end, winner identified.
winner can immediately hand off to collab mode.

fixed a tag collision bug where all 8 workers got the same model tag (exp_Cstar_main_001) because _next_worker_tag scanned for zip files before any existed.
fixed the monitoring loop exiting immediately on Windows because spawn processes weren't alive yet when is_alive() was first checked — replaced sleep(5) with a condition-based 60s wait.
gui monitor now filters out eval data from before the current session started so old runs don't show up in the live graph.
stripped verbose comment prose from train_agent.py, letsgetlunar.py, and gui_monitor.py.

first actual parallel sweep run attempted. 8 workers confirmed starting based on worker logs (SAC initialized, DummyVecEnv wrapped, logging started) but monitoring loop exited before data was collected due to the Windows spawn timing bug (now fixed).

## two-machine collab mode (option 2)

P2P socket connection between main PC (192.168.4.56) and MacBook (192.168.4.55) on port 7777.
shared model pot — both machines see each other's models ranked by best reward.
auto-trains best available model on each machine simultaneously (different seeds).
models transferred between machines as base64-encoded zip over socket.
config mismatch detection — refuses to share models if machines are on different sweep configs.
reconnects automatically if connection drops mid-training.
still need to run a collab session once a good C or C* base model exists.

## success criteria

altitude at insertion: 400 km ± 20 km.
radial velocity Vr: < 50 m/s.
tangential velocity Vtan: within 100 m/s of 1511 m/s.

convergence is not required for full credit — thorough documentation of what was tried and why is the primary grading criteria (40% thoroughness + critical thinking).

## writeup checklist

reward function designs and rationale (sparse vs shaped vs multiobjective).
network architectures tried (sizes, layer counts).
algorithm comparison — PPO vs SAC, why SAC fits continuous control better.
sweep results summary — which config won and why.
key observations and takeaways.
results / figures (include even if no full convergence — partial results count).
final PDF export.

## notes

NLP β(t) from project 2 is a useful sanity check — if RL agent is learning, its β profile should roughly match the NLP shape (starts high, sweeps toward 0).
exp C is the most promising base — keep iterating SAC before switching anything.
workflow once a model works: evaluate → generate plots → paste figures into writeup → export PDF.
