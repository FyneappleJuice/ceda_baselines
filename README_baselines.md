# CEDA-FGCS MARL baselines

Contemporary multi-agent RL baselines for the reviewer comparison, all
trained and evaluated on the **exact same environment** as CEDA-FGCS
(`CEDA-FGCS.py` — imported by path, its `Environment`, curriculum stages,
reward function and episode budget are reused verbatim).

| File | Algorithm | Notes |
|---|---|---|
| `VDN_FGCS.py` | **VDN** (Sunehag et al. 2018) | shared local Q net, `Q_tot = Σ_i Q_i`, double-DQN target, availability masks. This is QMIX with the mixing network replaced by a sum — the natural ablation of CEDA-FGCS's monotonic mixer. |
| `QMIX_FGCS.py` | **QMIX** (Rashid et al. 2018) | textbook monotonic mixing network with flat-state hypernetworks (`|w|=abs`). The ablation that isolates what CEDA-FGCS's entity-attention encoders add over vanilla QMIX. |
| `MADDPG_FGCS.py` | **MADDPG** (Lowe et al. 2017) | per-agent actor + per-agent centralised critic `Q_i(x, a_1..a_N)`, straight-through Gumbel-Softmax for the discrete action space, soft target updates. |
| `baseline_common.py` | shared infra | observation flattening, replay buffer, and the **single evaluation harness** used by every method. |
| `evaluate_ceda.py` | — | scores the trained CEDA-FGCS `.pth` with that same harness so every table row is produced by identical code / seeds. |
| `make_baseline_table.py` | — | collects `*_FGCS_eval.json` and emits `baseline_table.tex` + `.md`. |

QMIX and PPO/MAPPO baselines: plug their `*_FGCS_eval.json` (same schema —
see `baseline_common.evaluate`) into `make_baseline_table.py` and they line
up in the same table automatically.

## Quick check (any machine, ~1 min, CPU)

```bash
python VDN_FGCS.py --smoke-test
python MADDPG_FGCS.py --smoke-test
```

## Full runs (GPU)

```bash
# full 5-drone / 50-patient mission = curriculum stage 2 (default)
python VDN_FGCS.py    --episodes 12000 --device cuda:0 --output-dir runs/vdn
python QMIX_FGCS.py   --episodes 12000 --device cuda:0 --output-dir runs/qmix
python MADDPG_FGCS.py --episodes 12000 --device cuda:0 --output-dir runs/maddpg --train-every 2
```

CPU-only machines can smoke-test but not train at full scale — one
stage-2 episode is ~10^3 environment steps and the env step (A* + hazard
rasterisation) dominates. Use a GPU box for the real runs.

Useful flags (`baseline_common.base_arg_parser`): `--curriculum-stage`,
`--eval-every`, `--eval-episodes`, `--warmup`, `--buffer`, `--batch-size`,
`--lr`, `--reward-scale` (defaults to CEDA's `TD_REWARD_SCALE`),
`--train-every`, `--updates-per-step`, `--seed`.

Each run writes:
* `<algo>_agent_FGCS.pth` — weights
* `<algo>_FGCS_metrics.json` — full training + periodic-eval history
* `<algo>_FGCS_eval.json` — final evaluation (headline metrics)

## Build the comparison table

```bash
python evaluate_ceda.py --weights ctde_agent_marl_FGCS.pth --episodes 100 --device cuda:0 --output-dir runs/ceda
python make_baseline_table.py runs/ceda runs/vdn runs/maddpg runs/qmix runs/mappo runs/ppo
```

## Evaluation metrics (identical protocol for all methods)

`mean_reward`, `mean_delivery_rate` (delivered / spawned), `mean_triage_efficiency`
(weighted by patient acuity), `mission_success_rate`, `mean_died`,
`mean_landed`, `mean_obstacle_collisions`, `mean_agent_collisions`,
`mean_episode_length` — greedy (ε=0), fixed seed, `--eval-episodes` episodes
at the requested curriculum stage.
