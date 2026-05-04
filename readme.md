# letsgetlunar

project 3 of the astrodynamics course. an RL agent learns to perform lunar orbital insertion, guiding a spacecraft from the lunar surface to a 400 km circular orbit by adjusting thrust angle over a 160 second burn.

three experiments compare reward functions and network sizes: experiment A uses PPO with sparse reward as a baseline, experiment B uses PPO with shaped reward, and experiment C uses SAC with a multiobjective reward and exploring starts.

## how to run

```
python letsgetlunar.py
```

the interactive menu lets you train a new model or run a pre-trained one. when training starts it automatically launches the progress monitor in a separate terminal window.

## folder structure

```
letsgetlunar.py        main interactive entry point
readme.md
project_goals.md
scripts/
    lunar_env.py       gymnasium environment (equations of motion, reward functions)
    train_agent.py     training code for all three experiments
    evaluate_agent.py  deterministic evaluation and CSV output
    check_progress.py  live training monitor (auto-launched during training)
logs/                  tensorboard logs and evaluations.npz files per run
models/                saved model weights (exp_A_001.zip, exp_A_001_best.zip, ...)
output/                trajectory CSV files and results table
```

## dependencies

- stable-baselines3
- gymnasium
- numpy
- matplotlib
