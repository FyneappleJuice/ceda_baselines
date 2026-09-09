"""
Collect every ``*_FGCS_eval.json`` in the given directories and emit a
LaTeX table (and a Markdown copy) for the reviewer response / paper.

  python make_baseline_table.py runs/ceda runs/vdn runs/maddpg runs/qmix runs/mappo
"""

import argparse
import json
import sys
from pathlib import Path

ROWS = [
    ("mean_reward", "Return", "{:.1f}"),
    ("mean_delivery_rate", "Delivery rate", "{:.3f}"),
    ("mean_triage_efficiency", "Triage efficiency", "{:.3f}"),
    ("mission_success_rate", "Mission success", "{:.3f}"),
    ("mean_died", "Patient deaths", "{:.2f}"),
    ("mean_landed", "Drones landed", "{:.2f}"),
    ("mean_obstacle_collisions", "Obstacle collisions", "{:.1f}"),
    ("mean_agent_collisions", "Agent collisions", "{:.2f}"),
    ("mean_episode_length", "Episode length", "{:.0f}"),
]

ORDER = ["PPO", "VDN", "QMIX", "MADDPG", "MAPPO", "CEDA-FGCS"]


def find_evals(paths):
    found = {}
    for base in paths:
        base = Path(base)
        files = [base] if base.is_file() else base.rglob("*_FGCS_eval.json")
        for fp in files:
            data = json.loads(Path(fp).read_text())
            name = data.get("algorithm") or fp.stem.replace("_FGCS_eval", "")
            found[name] = data
    return found


def fmt(v, spec):
    try:
        return spec.format(v)
    except (TypeError, ValueError):
        return "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="baseline_table")
    args = ap.parse_args()

    evals = find_evals(args.paths)
    if not evals:
        sys.exit("No *_FGCS_eval.json files found.")
    methods = [m for m in ORDER if m in evals] + \
              [m for m in evals if m not in ORDER]

    # ---- Markdown ----
    md = ["| Metric | " + " | ".join(methods) + " |",
          "|" + "---|" * (len(methods) + 1)]
    for key, label, spec in ROWS:
        cells = [fmt(evals[m].get(key), spec) for m in methods]
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    Path(args.out + ".md").write_text("\n".join(md) + "\n")

    # ---- LaTeX ----
    col = "l" + "c" * len(methods)
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Comparison of CEDA-FGCS against contemporary "
           r"multi-agent RL baselines on the full 5-drone / 50-patient "
           r"mission (mean over evaluation episodes, identical seeds and "
           r"evaluation protocol).}",
           r"\label{tab:marl-baselines}",
           r"\begin{tabular}{" + col + "}", r"\toprule",
           "Metric & " + " & ".join(m.replace("_", r"\_") for m in methods)
           + r" \\", r"\midrule"]
    for key, label, spec in ROWS:
        cells = [fmt(evals[m].get(key), spec) for m in methods]
        tex.append(f"{label} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(args.out + ".tex").write_text("\n".join(tex) + "\n")

    print("\n".join(md))
    print(f"\nwrote {args.out}.md and {args.out}.tex")
    n_eps = {m: evals[m].get("episodes") for m in methods}
    print(f"evaluation episodes per method: {n_eps}")


if __name__ == "__main__":
    main()
