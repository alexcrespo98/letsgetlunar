# letsgetlunar

howdy, if you're a grader, the full write-up is in **project3.pdf**.

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
models/                  saved model weights (exp_A_main_001_r-45.zip, exp_C_main_003_r810_success.zip, ...)
output/                  trajectory CSVs and results table
```

model filenames encode how well the model performed. after training finishes the file is renamed to include the best evaluation reward and a `_success` suffix when the model meets the success criterion for that experiment. so `exp_C_main_003_r810_success.zip` means: experiment C, windows PC, run 3, best eval reward 810, meets the 500-point success threshold.

## dependencies

```
pip install stable-baselines3[extra] gymnasium numpy matplotlib
```

python 3.10+.

## collab mode

collab mode lets two machines fine-tune experiment C models back-and-forth overnight, passing the best weights between them automatically over a TCP socket on port 7777.

### networking / firewall

for the two machines to reach each other you need:

**Windows (firewall rules — run in an elevated PowerShell):**
```powershell
# allow inbound ICMP (ping) so the other machine can test connectivity
New-NetFirewallRule -DisplayName "Allow ICMPv4 inbound" -Protocol ICMPv4 -IcmpType 8 -Direction Inbound -Action Allow

# allow inbound TCP on port 7777 (letsgetlunar collab port)
New-NetFirewallRule -DisplayName "letsgetlunar collab" -Protocol TCP -LocalPort 7777 -Direction Inbound -Action Allow
```

**Ubuntu / macOS:** `ufw` / `pf` are usually permissive by default; if needed open port 7777 inbound.

### ping sanity check

before starting collab mode, ping each machine from the other to confirm basic connectivity:

```bash
# from ubuntu → ping the windows machine
ping 192.168.4.56

# from windows PowerShell → ping ubuntu
ping 192.168.4.55
```

if ping fails, fix the firewall first (see above). collab mode will also fail if you can't ping.

### same LAN or Tailscale

collab works on any network where the two machines can reach each other on port 7777:

- **same Wi-Fi / LAN** — just use the local IP addresses (e.g. `192.168.x.x`). find them with `ipconfig` on Windows and `ip addr` or `ifconfig` on Linux.
- **Tailscale (anywhere)** — install [Tailscale](https://tailscale.com/) on both machines, log in with the same account, and use the `100.x.x.x` Tailscale IP instead of the LAN IP. this works even across different networks (e.g. one machine on Wi-Fi, one on ethernet, or one at a different location entirely).

### how collab mode works

1. **connect** — each machine listens on port 7777 and simultaneously tries to connect outward to the peer. whichever direction succeeds first becomes the session. you can type `ip <address>` at the prompt to change the target IP on the fly.

2. **model pot** — both machines share a list of available experiment C models ("the pot"). each entry shows the model name, its best evaluation reward, whether it's stored locally or on the peer, and whether it's currently being trained.

3. **requesting models** — if the best available model isn't local, the machine sends a `model_request` to the peer, which streams the zip back over the socket. once received, the file is saved to `models/` and training starts automatically.

4. **training loop** — each machine picks the highest-reward untrained model it has locally and launches a fine-tune run in a background thread (2 million steps by default). while training it broadcasts `training_start` so the peer knows what's happening.

5. **sharing results** — when training finishes the new model zip is renamed to include the reward (e.g. `exp_C_main_003_r810_success.zip`) and a `training_done` message is broadcast. the peer adds it to the pot and can request it immediately if it's the new best.

### model naming convention

saved filenames encode performance so you don't have to dig into logs:

```
exp_C_main_003_r810_success.zip
      │     │   │    └─ best eval reward rounded to nearest integer
      │     │   └─ run sequence number (per machine)
      │     └─ machine name (main / backup) — prevents collisions when both train at once
      └─ experiment letter
```

the `_success` suffix appears when the model's best reward meets the success threshold for that experiment (500 for experiment C). models without the suffix are still useful — they're just not there yet.

when checking out models from the pot, the filename tells you at a glance how well it performed.

## the story so far

i trained experiments A, B, and C on the Windows machine over about 16 hours, watching the reward curves climb while monitoring in tensorboard. experiment C with the SAC agent and multi-objective reward got to a best reward of ~789, which already clears the 500-point success threshold.

then i got another computer set up — a macbook air someone was throwing out, now running ubuntu — and spent a while getting the two machines talking to each other. once ping worked and the firewall rules were in place, collab mode connected on the first try. both machines started fine-tuning exp C back and forth overnight.

i checked back the next day after both machines had been running for about 20 hours. the initial improvement i saw early on had stopped. the reward was stuck around 850 and wasn't moving. the problem is all i had was a timestamp in the terminal — i couldn't tell whether the model had waves of near-success and then fell back, or whether it just flatlined there and stayed. 850 is above the success threshold (500) but i want to see if it can do better and actually nail the orbit consistently.

next step: build a proper live GUI for each machine — a window that stays open while training, shows the current reward curve updating in real time, and plots each eval result with a timestamp on the x-axis so i can actually see the shape of the learning curve over wall-clock time, not just step count. that way if it spikes and falls back i'll know, and i can make better decisions about when to change parameters instead of guessing from a single number in the morning.

i then looked into different ways to optimize my current setup on the windows machine and found that since it has 12 physical cores i could theoretically run 10 parallel workers and get roughly 10x the training throughput overnight. then the machine started getting hot, so i did some research on how to control cpu throttling — figuring out which power plan settings and core limits let me maximize training speed without cooking the hardware.

## status

plateau reached. best reward so far: ~850. target is consistent orbit at 400 km altitude (success threshold: 500 reward). both machines have been running collab mode; next move is a live GUI dashboard per machine and a hyperparameter sweep to try to push past the plateau.
