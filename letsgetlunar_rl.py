# letsgetlunar_rl.py
# project 3 - rl orbital insertion
# alex crespo | 2026
# usage: python letsgetlunar_rl.py

import subprocess, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

STEPS = [
    ("training rl agents (3 experiments)",  os.path.join("scripts", "train_agent.py")),
    ("evaluating trained agents",            os.path.join("scripts", "evaluate_agent.py")),
    ("generating figures",                   os.path.join("scripts", "plot_rl_results.py")),
]

print()
print("moon time (rl edition)")
print("project 3 - rl orbital insertion")
print("alex crespo | 2026")
print()

for i, (desc, script) in enumerate(STEPS, 1):
    print(f"step {i}: {desc}")
    ret = subprocess.run([sys.executable, script])
    if ret.returncode != 0:
        print(f"  {script} failed, stopping.")
        sys.exit(1)
    print()

print("done. check output/ for figures and models/ for saved agents.")