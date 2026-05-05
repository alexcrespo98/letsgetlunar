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
models/                  saved model weights (exp_A_main_001_r-45.zip, exp_C_main_003_r810_success.zip, ...)
output/                  trajectory CSVs and results table
```

model filenames encode how well the model performed. after training finishes the file is renamed to include the best evaluation reward and a `_success` suffix when the model meets the success criterion for that experiment. pick option 2 from the menu to list models — higher reward = better performance.

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
- **Tailscale (anywhere)** — install [Tailscale](https://tailscale.com/) on both machines, log in with the same account, and use the `100.x.x.x` Tailscale IP instead of the LAN IP. this works even when the machines are on different networks or behind NAT.

### how collab mode works

1. **connect** — each machine listens on port 7777 and simultaneously tries to connect outward to the peer. whichever direction succeeds first becomes the session. you can type `ip <address>` at the prompt to change the peer IP without restarting.

2. **model pot** — both machines share a list of available experiment C models ("the pot"). each entry shows the model name, its best evaluation reward, whether it's stored locally or on the peer, and whether it's currently being trained.

3. **requesting models** — if the best available model isn't local, the machine sends a `model_request` to the peer, which streams the zip back over the socket. once received, the file is saved to `models/` and is ready to train from.

4. **training loop** — each machine picks the highest-reward untrained model it has locally and launches a fine-tune run in a background thread (2 million steps by default). while training it broadcasts `training_start` so the peer knows not to train the same model.

5. **sharing results** — when training finishes the new model zip is renamed to include the reward (e.g. `exp_C_main_003_r810_success.zip`) and a `training_done` message is broadcast. the peer adds the new model to their pot and can request the zip to continue the chain.

### model naming convention

saved filenames encode performance so you don't have to dig into logs:

```
exp_C_main_003_r810_success.zip
      │     │   │    └─ best eval reward rounded to nearest integer
      │     │   └─ run sequence number (per machine)
      │     └─ machine name (main / backup) — prevents collisions when both train at once
      └─ experiment letter
```

the `_success` suffix appears when the model's best reward meets the success threshold for that experiment (500 for experiment C). models without the suffix are still useful — they're just not quite there yet.

when checking out models from the pot, the filename tells you at a glance how well it performed.

## the story so far

i trained experiments A, B, and C on the Windows machine over about 16 hours, watching the reward curves climb while monitoring in tensorboard. experiment C with the SAC agent and multi-objective reward was the clear winner, reaching a best reward around 810 before i had to stop.

then i got another computer set up — a macbook air someone was throwing out, now running ubuntu — and spent a while getting the two machines talking to each other. once ping worked and the firewall rules were in, collab mode connected and the model pot started populating on both sides. the plan is to leave both machines running for about four days, each one fine-tuning the best model the other produces and passing the result back. we'll see if it works.

the machine name in the filename means neither side can accidentally overwrite the other's output, so they can genuinely run unsupervised without me babysitting file names.

## status

still training. best reward so far: ~810. target is consistent orbit at 400 km altitude.
