# Running the CEDA-FGCS baselines on the GPU PC

For a fresh machine. A Claude Code session on that PC can execute all of this.

## 1. Clone

```bash
git clone <REPO_URL> CEDA_Baselines
cd CEDA_Baselines
```

## 2. Python env with CUDA PyTorch

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
python -m pip install -U pip
pip install numpy "pygame-ce>=2.5"
# pick the CUDA build matching the box's driver (cu121 / cu124 / ...):
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True`.

## 3. Wiring check (~2 min)

```bash
python run_all_baselines.py --device cuda:0 --smoke-test
```

Expect three `[... final evaluation: {...}]` lines and `runs/*/`.

## 4. Full runs

```bash
python run_all_baselines.py --device cuda:0 --episodes 12000
```

Runs VDN -> QMIX -> MADDPG in sequence (each its own subprocess; logs at
`runs/<algo>/train.log`), then scores CEDA-FGCS with the same harness and
writes `runs/baseline_table.tex` + `.md`.

**Make it survive an AnyDesk disconnect** — run it detached:

```bash
# Linux
nohup python run_all_baselines.py --device cuda:0 --episodes 12000 > runs_all.log 2>&1 &
# Windows PowerShell
Start-Process -NoNewWindow python "run_all_baselines.py --device cuda:0 --episodes 12000" -RedirectStandardOutput runs_all.log -RedirectStandardError runs_all.err
```

## 5. Progress / results

```bash
tail -f runs/vdn/train.log          # per-eval-interval metrics
python make_baseline_table.py runs/vdn runs/qmix runs/maddpg runs/ceda
```

Each baseline also checkpoints `runs/<algo>/<algo>_agent_FGCS.pth` and a
full `*_FGCS_metrics.json` (training + periodic-eval history) every 500
episodes.

## Notes

- Single GPU: sequential is fine. To parallelise, launch the three
  `*_FGCS.py` scripts yourself on different `--device` / MPS slices.
- `--episodes 12000` matches CEDA-FGCS. Lower it (e.g. `--episodes 6000`)
  if wall-clock is tight; report the budget in the paper.
- MADDPG uses `--train-every 2` by default here (10 networks/update).
