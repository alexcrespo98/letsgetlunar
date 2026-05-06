# project goals

not for submission. internal tracking only.  
last updated: 2026-05-06

---

## overview

design an RL agent to control thrust angle β(t) for optimal lunar orbital insertion.  
target orbit: **400 km circular orbit** (alt ±20 km, Vr < 50 m/s, Vtan within 100 m/s of 1511 m/s).  
deliverable: 2–4 page writeup summarizing reward functions, architectures, algorithms, and takeaways.

---

## ✅ done

### project 2 baseline
- [x] NLP solver (scipy trust-constr) converges to optimal β(t) profile
- [x] all 4 figures generated (beta profile, state history, polar trajectory, ground track)
- [x] output saved to `output/` — optimal_trajectory.csv, optimal_beta_profile.csv, figs 1–4
- [x] pipeline runner (`letsgetlunar.py` proj 2 version) works end-to-end

### RL infrastructure
- [x] interactive menu with 7 options (solo train, collab, run pretrained, fine-tune, quit, sweep, parallel)
- [x] time-budgeted training — input hours → auto-scales step budget per experiment using `BUDGET_SHARE`
- [x] non-overwriting tagged model saves (exp_A_001, exp_C_main_003, exp_Cstar_main_w01_001, etc.)
- [x] attempt log — `logs/attempt_log.csv` records per-run stats (steps, best reward, success rate, notes)
- [x] auto-launch of `check_progress.py` and `gui_monitor.py` on training start
- [x] early stopping via `StopTrainingOnNoModelImprovement` (wired in `train_agent.py`)
- [x] grader-friendly mode — simplified menu at startup for graders

### experiments (run 1)
- [x] **exp A** — PPO, sparse reward, small net → no learning, confirmed as expected
- [x] **exp B** — PPO, shaped reward, large net → no meaningful convergence, confirmed as expected
- [x] **exp C** — SAC, multiobjective reward, medium net → best reward 789.4, partial progress
- [x] all three model folders saved (`ppo_sparse_small`, `ppo_shaped_large`, `sac_multiobjective_medium`)
- [x] eval logs present at `logs/exp_A`, `logs/exp_B`, `logs/exp_C`

---

## 🔄 in progress

- [ ] exp C re-run with full 2M step budget (run 1 was cut short)
- [ ] finding a model that clears success criteria (alt 400±20 km, Vr < 50, Vtan within 100 of 1511)

---

## ⬜ not started

### evaluation + results
- [ ] run `evaluate_agent.py` on best saved model
- [ ] generate rollout plots (alt, Vr, Vtan, β vs time) from best RL agent
- [ ] compare run 1 vs run 2 using `attempt_log.csv`

### writeup sections
- [ ] reward function designs and rationale (sparse vs shaped vs multiobjective)
- [ ] network architectures tried (sizes, layer counts)
- [ ] algorithm comparison — PPO vs SAC, why SAC fits continuous control better
- [ ] sweep results summary — which config won and why
- [ ] key observations and takeaways
- [ ] results / figures (include even if no full convergence — partial results count)
- [ ] final PDF export

---

## 🚀 extra things to maximize parameters and compute

all of these are already fully implemented in `letsgetlunar.py` and ready to use.

### fine-tune / warm-start (option 4)
- [x] pick any saved exp C model and continue training from it
- [x] configurable additional step budget
- [x] tagged saves so warm-start runs don't overwrite originals
- [ ] run a full 2M step warm-start from best C model once one exists

### hyperparameter sweep — exp C* (option 6)
- [x] 10 pre-built configs sampled via Latin hypercube across:
  - learning rate: 1e-4, 3e-4, 1e-3
  - network size: [128,128], [256,256], [256,128,64]
  - buffer size: 500k, 1M
  - entropy coef: auto, 0.1, 0.2
  - batch size: 256, 512
  - reward weights: 3 different alt/Vr/Vtan splits
  - gaussian altitude width: 30 km vs 50 km
- [x] results logged to `logs/sweep_results.csv` after each config
- [x] ranked summary + auto-suggestion of best config at end
- [x] can immediately fine-tune with winning config (solo or collab)
- [ ] run the full sweep (6–10 configs, ~3 hrs) once base model is stable

### parallel multi-core training — exp C* (option 7)
- [x] spawns N worker processes (default: physical cores − 2, detected via psutil)
- [x] each worker gets a unique random seed → different exploration paths
- [x] live status table refreshes every 30s showing steps, recent reward, best reward per worker
- [x] all results ranked at end, winner identified
- [x] winner can immediately hand off to collab mode
- [ ] run a parallel session (4–6 workers) once a good base model exists

### two-machine collab mode (option 2)
- [x] P2P socket connection between main PC (192.168.4.56) and MacBook (192.168.4.55) on port 7777
- [x] shared model pot — both machines see each other's models ranked by best reward
- [x] auto-trains best available model on each machine simultaneously (different seeds)
- [x] models transferred between machines as base64-encoded zip over socket
- [x] config mismatch detection — refuses to share models if machines are on different sweep configs
- [x] reconnects automatically if connection drops mid-training
- [ ] run a collab session once a good C or C* base model exists

---

## success criteria

| metric | target |
|---|---|
| altitude at insertion | 400 km ± 20 km |
| radial velocity Vr | < 50 m/s |
| tangential velocity Vtan | within 100 m/s of 1511 m/s |

> convergence is not required for full credit — thorough documentation of what was tried and why is the primary grading criteria (40% thoroughness + critical thinking).

---

## notes

- NLP β(t) from project 2 is a useful sanity check — if RL agent is learning, its β profile should roughly match the NLP shape (starts high, sweeps toward 0)
- exp C is the most promising base — keep iterating SAC before switching anything
- workflow once a model works: evaluate → generate plots → paste figures into writeup → export PDF
