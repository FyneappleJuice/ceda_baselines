"""
One-shot driver for the GPU box: trains every MARL baseline in sequence,
then scores the trained CEDA-FGCS agent with the same harness and builds
the comparison table.

  python run_all_baselines.py --device cuda:0 --episodes 12000
  python run_all_baselines.py --device cuda:0 --only vdn qmix      # subset
  python run_all_baselines.py --device cuda:0 --smoke-test         # wiring check

Each baseline is launched as its own subprocess so a crash in one does not
lose the others; logs go to <output-root>/<algo>/train.log.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASELINES = {
    "vdn": "VDN_FGCS.py",
    "qmix": "QMIX_FGCS.py",
    "maddpg": "MADDPG_FGCS.py",
}
EXTRA_ARGS = {"maddpg": ["--train-every", "2"]}


def run(cmd, log_path):
    print(f"\n=== {' '.join(cmd)}\n    log -> {log_path}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--episodes", type=int, default=12000)
    p.add_argument("--output-root", default="runs")
    p.add_argument("--only", nargs="+", choices=list(BASELINES),
                   default=list(BASELINES))
    p.add_argument("--ceda-weights", default="ctde_agent_marl_FGCS.pth")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    root = Path(args.output_root)
    py = sys.executable
    t0 = time.time()

    for name in args.only:
        script = BASELINES[name]
        out = root / name
        cmd = [py, "-u", script, "--device", args.device,
               "--seed", str(args.seed), "--output-dir", str(out)]
        if args.smoke_test:
            cmd.append("--smoke-test")
        else:
            cmd += ["--episodes", str(args.episodes)]
        cmd += EXTRA_ARGS.get(name, [])
        rc = run(cmd, out / "train.log")
        print(f"--- {name} finished rc={rc}  "
              f"({(time.time() - t0) / 3600:.2f} h elapsed)", flush=True)

    # Score CEDA-FGCS with the identical harness.
    if Path(args.ceda_weights).exists() and not args.smoke_test:
        out = root / "ceda"
        rc = run([py, "-u", "evaluate_ceda.py",
                  "--weights", args.ceda_weights,
                  "--episodes", str(args.eval_episodes),
                  "--device", args.device,
                  "--output-dir", str(out)], out / "eval.log")
        print(f"--- ceda eval rc={rc}", flush=True)

    # Build the table from whatever eval JSONs exist.
    dirs = [str(root / n) for n in args.only] + [str(root / "ceda")]
    run([py, "-u", "make_baseline_table.py", *dirs,
         "--out", str(root / "baseline_table")], root / "table.log")
    print(f"\nAll done in {(time.time() - t0) / 3600:.2f} h. "
          f"Table: {root / 'baseline_table.tex'}", flush=True)


if __name__ == "__main__":
    main()
