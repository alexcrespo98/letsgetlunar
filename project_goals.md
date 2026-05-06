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

- [x] project 2 NLP baseline complete — scipy trust-constr solver converges, produces clean β(t) profile and all 4 figures
- [x] dynamics validated — altitude, Vr, Vtan, and angular position all match expected physics
- [x] NLP output saved to `output/` (optimal_trajectory.csv, optimal_beta_profile.csv, figures)
- [x] pipeline runner (`letsgetlunar.py`) works end-to-end
- [x] RL infrastructure built — interactive menu, time-budgeted training, tagged model saves (exp_A_001, etc.)
- [x] attempt log wired up — `logs/attempt_log.csv` records per-experiment stats
- [x] training monitor auto-launches on training start
- [x] early stopping via `StopTrainingOnNoModelImprovement`
- [x] run 1 complete
  - [x] exp A (PPO sparse reward, small net) — confirmed no learning, expected
  - [x] exp B (PPO shaped reward, large net) — no meaningful convergence, expected
  - [x] exp C (SAC multiobjective, medium net) — best reward 789.4, partial progress

---

## 🔄 in progress

- [ ] exp C re-run with full 2M step budget (run 1 was capped short)
- [ ] finding a model that converges to orbit insertion criteria

---

## ⬜ not started

- [ ] run `evaluate_agent.py` on best saved model
- [ ] generate plots from RL agent rollout for writeup (alt, Vr, Vtan, β vs time)
- [ ] compare run 1 vs run 2 via `attempt_log.csv`
- [ ] try at least one additional RL algorithm (e.g. DDPG or TD3)
- [ ] try at least one additional network size variation
- [ ] try at least one additional reward function variant beyond exp A/B/C
- [ ] write up sections:
  - [ ] reward function designs and rationale
  - [ ] network architectures tried
  - [ ] algorithm comparison (PPO vs SAC vs ?)
  - [ ] key observations and takeaways
  - [ ] results / figures (even if no convergence)
- [ ] final PDF export of writeup

---

## success criteria

| metric | target |
|---|---|
| altitude at insertion | 400 km ± 20 km |
| radial velocity Vr | < 50 m/s |
| tangential velocity Vtan | within 100 m/s of 1511 m/s |

> convergence is not required for full credit — thorough documentation of what was tried and why is the main grading criteria.

---

## notes

- NLP β(t) profile from project 2 is a useful sanity check — RL agent output should roughly match its shape if it's learning correctly
- exp C is the most promising so far, keep iterating on SAC before switching algorithms
- when a model works, run evaluate → plot → drop figures straight into writeup
