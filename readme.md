# letsgetlunar

project 3. an RL agent learns to perform lunar orbital insertion, guiding a spacecraft from the lunar surface to a 400 km circular orbit by adjusting thrust angle over a 160 second burn.

three experiments compare reward functions and network sizes: experiment A uses PPO with sparse reward as a baseline, experiment B uses PPO with shaped reward, and experiment C uses SAC with a multiobjective reward and exploring starts.

## how to run

```
python letsgetlunar.py
```

the interactive menu lets you train a new model or run a pre-trained one. when training starts it automatically launches the progress monitor in a separate terminal window.

## folder structure

```
letsgetlunar.py        main interactive entry point - run this
readme.md
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

## running on two machines

the project is designed to run on two machines simultaneously.

the main machine (windows PC) runs exp C with exploring starts enabled. the backup machine (macbook air, ubuntu) runs exp C with exploring starts disabled, which trains from the fixed initial condition that evaluation uses. this is the more targeted run.

models are not shared during training. after both finish, copy the better best_model.zip manually.

to set up the backup machine from scratch:

- git clone https://github.com/alexcrespo98/letsgetlunar
- cd letsgetlunar
- pip install stable-baselines3[extra] gymnasium numpy matplotlib
- python3 letsgetlunar.py
- select backup when prompted
