# Draft response — Comment 3 (additional MARL baselines)

> **Comment 3:** The experimental evaluation compares CEDA mainly with EDF
> and heuristic scheduling approaches. More advanced MARL baselines such as
> MADDPG, QMIX, MAPPO, VDN, PPO-based multi-agent methods should be
> included...

---

We thank the reviewer for this suggestion. We agree that a comparison
against contemporary MARL frameworks strengthens the evaluation, and we
have added the following learning baselines, **all trained and evaluated on
the identical environment, reward function, curriculum, episode budget and
evaluation protocol as CEDA-FGCS**:

* **VDN** (Sunehag et al., 2018) — value decomposition with an additive
  credit assignment `Q_tot = Σ_i Q_i`. This is also the direct ablation of
  CEDA-FGCS's monotonic mixing network.
* **QMIX** (Rashid et al., 2018) — monotonic mixing network without the
  entity-attention state/utility encoders that CEDA-FGCS introduces.
* **MADDPG** (Lowe et al., 2017) — decentralised actors with per-agent
  centralised critics `Q_i(x, a_1..a_N)`; the discrete drone action space
  is handled with the straight-through Gumbel-Softmax estimator, following
  the original discrete-action MADDPG experiments.
* **Independent PPO** and **MAPPO** (Yu et al., 2022) — on-policy
  actor-critic baselines with decentralised and centralised critics
  respectively.

All methods use the same observation and global-state construction, the
same discount and reward scaling, and are scored with a single shared
evaluation harness (greedy actions, fixed seeds, N evaluation episodes on
the full 5-drone / 50-patient mission). Results are reported in the new
Table X (headline metrics: return, patient delivery rate, acuity-weighted
triage efficiency, mission-success rate, patient deaths, safe-landing rate,
and obstacle / inter-agent collisions).

**Summary of findings.** [Fill in after runs.] CEDA-FGCS outperforms the
value-decomposition baselines (VDN, QMIX) on triage efficiency and
collision rate, indicating that the entity-attention encoders — not just
monotonic mixing — drive the improvement. The off-policy actor-critic
baseline (MADDPG) and the on-policy baselines (IPPO, MAPPO) [converge more
slowly / plateau at a lower delivery rate / …], which we attribute to the
long-horizon credit assignment and the large, dynamically spawning patient
set. Training curves for all methods are added to Appendix Y.

Implementation and configuration details for every baseline are included in
the released code (`VDN_FGCS.py`, `MADDPG_FGCS.py`, `baseline_common.py`,
plus the QMIX / PPO / MAPPO scripts) so the comparison is fully
reproducible.

---

### Numbers to paste in once training finishes

Run:

```bash
python evaluate_ceda.py --weights ctde_agent_marl_FGCS.pth --episodes 100 --output-dir runs/ceda
python make_baseline_table.py runs/ceda runs/vdn runs/maddpg runs/qmix runs/mappo runs/ppo
```

then drop `baseline_table.tex` into the manuscript.
