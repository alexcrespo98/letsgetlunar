# letsgetlunar

hi there — if you're a grader, the full write-up is in **project3.pdf**.

if you want to run the code yourself:

```
python letsgetlunar.py
```

it'll walk you through everything interactively. you can train from scratch, run a pre-trained model, or fine-tune experiment C. pretty self-explanatory once you're in the menu.

## what's in here

```
letsgetlunar.py          entry point — run this
scripts/
    lunar_env.py         the physics: equations of motion, reward functions, gymnasium environment
    train_agent.py       training logic for all three experiments
    evaluate_agent.py    deterministic evaluation, outputs CSV
    check_progress.py    live training monitor, auto-launched during training
logs/                    tensorboard logs and evaluations.npz per run
models/                  saved model weights (exp_A_001.zip, exp_C_002_best.zip, ...)
output/                  trajectory CSVs and results table
```

if you want to dig into the physics, start with `scripts/lunar_env.py`. if you want to see the pretrained models or run them yourself, they're in `models/` — pick option 2 from the menu and it'll list them with their best reward so you know which one performed best.

## dependencies

```
pip install stable-baselines3[extra] gymnasium numpy matplotlib
```

python 3.10+.

## the two-machine setup

someone was throwing out a macbook air with a broken screen. i took it, put ubuntu on it, and used it to run a second training process in parallel. each machine runs 2 million steps of experiment C simultaneously — one with exploring starts (random initial conditions), one fixed. when both finish i take whichever model is better and fine-tune it further.

i'm working on automating the model-sharing so they can hand off to each other without me babysitting it. right now i manually scp the best zip between machines.

running on two different OSes (Windows and Ubuntu) actually did add some friction. the terminal launching code had to be split — Windows uses `start cmd`, Linux tries xterm/gnome-terminal, and the python binary is `python` on Windows and `python3` on Ubuntu. small stuff but it adds up.

## status

still training. best reward so far: ~810. target is consistent orbit at 400 km altitude.
