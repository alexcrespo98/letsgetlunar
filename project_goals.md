# project goals

not for submission. internal tracking only.

## original starting point

letsgetlunar_rl.py ran train, evaluate, and plot sequentially with no user interaction. models were saved to hardcoded folder names (ppo_sparse_small, ppo_shaped_large, sac_multiobjective_medium). each run would overwrite the previous one.

## what we want to achieve

- interactive menu with time budgeted training
- non-overwriting model saves using tagged names (exp_A_001, exp_C_002, etc.)
- attempt log that records each run to logs/attempt_log.csv with per-experiment stats
- auto-launch of the training monitor when training starts
- early stopping via StopTrainingOnNoModelImprovement to avoid wasted compute

## current status (run 1)

run 1 complete. exp A and exp B failed as expected: sparse reward produced no learning, shaped reward showed no meaningful convergence either. exp C (SAC multiobjective) reached best reward 789.4 with 59% success rate over 121 evals. training stopped early when reward declined from 789 to 368 with no recovery. best model saved at models/sac_multiobjective_medium/best_model.zip.

## next steps

- run exp C again with the full 2M step budget now that infra is stable
- compare run 1 vs run 2 results using attempt_log.csv
- run evaluate_agent.py and plot results for write-up

## success criteria

exp C achieves orbit: altitude within 20 km of 400 km, radial velocity Vr under 50 m/s, tangential velocity Vtan within 100 m/s of circular velocity (1511 m/s).
