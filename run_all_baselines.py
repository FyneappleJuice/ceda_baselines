"""
Driver for the GPU box: trains every MARL baseline, scores the trained
CEDA-FGCS agent with the same harness, and builds the comparison table.

Sequential (default) or --parallel (all baselines at once -- the env step
is single-threaded Python, so N processes ~= N x throughput on a multicore
box; the tiny MLPs share the GPU comfortably).

  # ~5-6 h on one RTX 4080-class GPU with an 8+ core CPU:
  python run_all_baselines.py --device cuda:0 --parallel \
      --episodes 3500 --warmup 3000 --buffer 80000 --eval-every 700

  python run_all_baselines.py --device cuda:0 --episodes 12000     # full, sequential
  python run_all_baselines.py --device cuda:0 --smoke-test         # wiring check

Logs: <output-root>/<algo>/train.log
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
# MADDPG does 10 network updates per learn(); amortise it.
EXTRA_ARGS = {"maddpg": ["--train-every", "2"]}


def build_cmd(name, args):
    out = Path(args.output_root) / name
    cmd = [sys.executable, "-u", BASELINES[name], "--device", args.device,
           "--seed", str(args.seed), "--output-dir", str(out)]
    if args.smoke_test:
        cmd.append("--smoke-test")
    else:
        cmd += ["--episodes", str(args.episodes)]
        for flag in ("warmup", "buffer", "batch_size", "eval_every",
                     "eval_episodes", "lr"):
            val = getattr(args, flag)
            if val is not None:
                cmd += [f"--{flag.replace('_', '-')}", str(val)]
    cmd += EXTRA_ARGS.get(name, [])
    return cmd, out / "train.log"


def run_sequential(names, args):
    t0 = time.time()
    for name in names:
        cmd, log = build_cmd(name, args)
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {' '.join(cmd)}\n    log -> {log}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"--- {name} rc={rc}  ({(time.time()-t0)/3600:.2f} h)", flush=True)


def run_parallel(names, args):
    procs = {}
    for name in names:
        cmd, log = build_cmd(name, args)
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"=== [{name}] {' '.join(cmd)}\n    log -> {log}", flush=True)
        procs[name] = (subprocess.Popen(cmd, stdout=open(log, "w"),
                                        stderr=subprocess.STDOUT), time.time())
    t0 = time.time()
    remaining = dict(procs)
    while remaining:
        for name, (proc, _) in list(remaining.items()):
            if proc.poll() is not None:
                print(f"--- {name} rc={proc.returncode}  "
                      f"({(time.time()-t0)/3600:.2f} h)", flush=True)
                del remaining[name]
        time.sleep(20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--episodes", type=int, default=12000)
    p.add_argument("--output-root", default="runs")
    p.add_argument("--only", nargs="+", choices=list(BASELINES),
                   default=list(BASELINES))
    p.add_argument("--parallel", action="store_true",
                   help="Run all baselines concurrently (needs RAM for "
                        "one replay buffer per baseline).")
    p.add_argument("--ceda-weights", default="ctde_agent_marl_FGCS.pth")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--smoke-test", action="store_true")
    # passthrough knobs (None -> use each script's own default)
    p.add_argument("--warmup", type=int)
    p.add_argument("--buffer", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--eval-episodes", type=int)
    p.add_argument("--lr", type=float)
    args = p.parse_args()

    root = Path(args.output_root)
    t0 = time.time()

    (run_parallel if args.parallel else run_sequential)(args.only, args)

    if Path(args.ceda_weights).exists() and not args.smoke_test:
        out = root / "ceda"
        out.mkdir(parents=True, exist_ok=True)
        print("\n=== CEDA-FGCS evaluation", flush=True)
        with open(out / "eval.log", "w") as fh:
            subprocess.run([sys.executable, "-u", "evaluate_ceda.py",
                            "--weights", args.ceda_weights,
                            "--episodes", str(args.eval_episodes or 100),
                            "--device", args.device,
                            "--output-dir", str(out)],
                           stdout=fh, stderr=subprocess.STDOUT)

    dirs = [str(root / n) for n in args.only] + [str(root / "ceda")]
    subprocess.run([sys.executable, "-u", "make_baseline_table.py", *dirs,
                    "--out", str(root / "baseline_table")])
    print(f"\nAll done in {(time.time()-t0)/3600:.2f} h. "
          f"Table: {root/'baseline_table.tex'} / .md", flush=True)


if __name__ == "__main__":
    main()
