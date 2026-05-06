# letsgetlunar

RL agent for lunar orbital insertion — project 3. for the full write-up see **project3.md** (or **project3.pdf** if included).

## how to run

```
python letsgetlunar.py
```

the interactive menu walks you through everything: train from scratch, run a pre-trained model, fine-tune experiment C, or launch a hyperparameter sweep. graders can pick option 2 at the opening prompt for a guided "run pretrained model" flow that automatically generates and shows plots.

## what's in the repo

```
letsgetlunar.py          entry point — run this
project3.md              project write-up summary
scripts/
    lunar_env.py         physics: equations of motion, reward functions, gymnasium environment
    train_agent.py       training logic for experiments A, B, C
    evaluate_agent.py    evaluation, trajectory CSVs, and matplotlib plots (auto-launched on model run)
    check_progress.py    live terminal training monitor, auto-launched during training
    gui_monitor.py       live GUI reward curve, launched during fine-tune and sweep runs
logs/                    tensorboard logs and evaluations.npz per run
models/                  saved model weights (exp_A_main_001_r-45.zip, exp_C_main_003_r912_success.zip, ...)
output/                  trajectory CSVs, per-experiment PNG plots, and results_table.csv
```

## model naming convention

```
exp_C_main_003_r912_success.zip
      │     │   │    └─ best eval reward rounded to nearest integer
      │     │   └─ run sequence number (per machine)
      │     └─ machine name (main / backup) — prevents collisions when both train at once
      └─ experiment letter
```

the `_success` suffix appears when the best reward meets the success threshold (500 for experiment C). best reward achieved so far: **912**, found on the auxiliary machine during a collab session.

## dependencies

```
pip install stable-baselines3[extra] gymnasium numpy matplotlib
```

python 3.10+.

## collab mode

collab mode lets two machines fine-tune experiment C models back-and-forth overnight, passing the best weights between them automatically over a TCP socket on port 7777.

### networking / firewall

**Windows (run in elevated PowerShell):**
```powershell
New-NetFirewallRule -DisplayName "Allow ICMPv4 inbound" -Protocol ICMPv4 -IcmpType 8 -Direction Inbound -Action Allow
New-NetFirewallRule -DisplayName "letsgetlunar collab" -Protocol TCP -LocalPort 7777 -Direction Inbound -Action Allow
```

**Ubuntu / macOS:** open port 7777 inbound if needed (`ufw allow 7777/tcp`).

### ping sanity check

```bash
# from ubuntu → ping the windows machine
ping 192.168.4.56

# from windows PowerShell → ping ubuntu
ping 192.168.4.55
```

### same LAN or Tailscale

- **same Wi-Fi / LAN**: use local IP addresses (`ipconfig` on Windows, `ip addr` on Linux).
- **Tailscale**: install on both machines, log in with the same account, use the `100.x.x.x` Tailscale IP.

### how collab mode works

1. **connect**: each machine listens on port 7777 and simultaneously tries to connect outward to the peer. type `ip <address>` at the prompt to change the target IP on the fly.
2. **model pot**: both machines share a ranked list of experiment C models. each entry shows name, best reward, location (local/peer), and training status.
3. **requesting models**: if the best model isn't local, the machine sends a `model_request` to the peer, which streams the zip back over the socket.
4. **training loop**: each machine fine-tunes the highest-reward model it has locally (2 million steps by default) with a unique random seed.
5. **sharing results**: when training finishes the new zip is renamed to include the reward and a `training_done` message is broadcast so the peer can request it immediately.

