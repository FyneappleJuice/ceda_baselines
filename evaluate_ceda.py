"""
Score the trained CEDA-FGCS agent with the *same* evaluation harness the
baselines use (baseline_common.evaluate), so every row of the paper's
comparison table is produced by identical code and identical seeds.

  python evaluate_ceda.py --weights ctde_agent_marl_FGCS.pth \
      --episodes 100 --device cuda:0 --output-dir runs/ceda
"""

import argparse
from pathlib import Path

import torch

import baseline_common as bc

ceda = bc.ceda


def load_agent(weights_path, device):
    agent = ceda.CTDEAgent(
        bc.ACTION_DIM, ceda.LEARNING_RATE, bc.GAMMA, device,
        mixed_precision=False,
    )
    ckpt = torch.load(weights_path, map_location=device)
    policy = ckpt.get("policy_state_dict", ckpt.get("policy"))
    agent.policy_net.load_state_dict(policy)
    if "mixer_state_dict" in ckpt:
        agent.mixer.load_state_dict(ckpt["mixer_state_dict"])
    agent.policy_net.eval()
    return agent


def main():
    p = argparse.ArgumentParser(description="Evaluate CEDA-FGCS with the "
                                            "shared baseline harness.")
    p.add_argument("--weights", default="ctde_agent_marl_FGCS.pth")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=bc.MAX_STEPS)
    p.add_argument("--curriculum-stage", type=int, default=bc.FULL_STAGE)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default=".")
    args = p.parse_args()

    device = bc.resolve_device(args.device)
    bc.set_global_seed(args.seed)
    agent = load_agent(args.weights, device)

    def select_fn(state):
        return agent.select_actions(state, epsilon=0.0)

    result = bc.evaluate(select_fn, args.episodes, args.seed,
                         args.max_steps, args.curriculum_stage)
    result["algorithm"] = "CEDA-FGCS"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bc.write_json(out / "CEDA_FGCS_eval.json", result)
    print(result)


if __name__ == "__main__":
    main()
