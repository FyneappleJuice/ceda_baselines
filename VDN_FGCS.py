"""
VDN (Value-Decomposition Networks, Sunehag et al. 2018) baseline for
CEDA-FGCS.

Standard formulation:
  * one shared local Q network Q(o_i, .) applied to every drone (the agent
    identity is part of o_i, so the shared net still specialises per drone);
  * the joint action value is the plain sum of per-agent utilities
        Q_tot(o, a) = sum_i Q(o_i, a_i)
    (this is exactly QMIX with the mixing network replaced by a sum, which
    is why VDN is the natural ablation to report next to CEDA-FGCS's
    monotonic mixer);
  * double-DQN bootstrapped target on the shared team reward, hard target
    network refresh, availability-masked greedy actions.

Run:
  python VDN_FGCS.py --smoke-test
  python VDN_FGCS.py --episodes 12000 --device cuda:0 --output-dir runs/vdn
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import baseline_common as bc


class SharedQNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=(512, 512, 256)):
        super().__init__()
        self.body = bc.mlp((obs_dim, *hidden), act=nn.ReLU)
        self.head = nn.Linear(hidden[-1], action_dim)

    def forward(self, obs):
        return self.head(F.relu(self.body(obs)))


class VDNAgent:
    def __init__(self, args, device):
        self.device = device
        self.gamma = bc.GAMMA
        self.batch_size = args.batch_size
        self.reward_scale = args.reward_scale
        self.target_update = 500
        self.grad_clip = 10.0

        self.q = SharedQNet(bc.LOCAL_OBS_DIM, bc.ACTION_DIM).to(device)
        self.target_q = SharedQNet(bc.LOCAL_OBS_DIM, bc.ACTION_DIM).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=args.lr, eps=1e-5)
        self.buffer = bc.ReplayBuffer(args.buffer)
        self.learn_steps = 0

    # -- acting -----------------------------------------------------------
    @torch.no_grad()
    def act(self, local_obs, action_masks, epsilon):
        obs = torch.as_tensor(local_obs, dtype=torch.float32, device=self.device)
        q = self.q(obs).cpu().numpy()                     # [N, A]
        greedy = bc.masked_greedy(q, action_masks)
        if epsilon <= 0.0:
            return greedy
        rand = bc.masked_random_actions(action_masks)
        take_rand = np.random.random(bc.NUM_AGENTS) < epsilon
        return np.where(take_rand, rand, greedy).astype(np.int64)

    def greedy_select(self, state):
        lo, _, am = bc.encode_state(state)
        return self.act(lo, am, epsilon=0.0)

    # -- learning -------------------------------------------------------
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None
        b = self.buffer.sample(self.batch_size)
        obs = b["obs"].to(self.device)               # [B, N, O]
        next_obs = b["next_obs"].to(self.device)     # [B, N, O]
        actions = b["actions"].long().to(self.device)  # [B, N]
        next_mask = b["next_mask"].to(self.device)   # [B, N, A]
        reward = b["reward"].to(self.device)         # [B]
        done = b["done"].to(self.device)             # [B]

        bsz = obs.shape[0]
        q_all = self.q(obs.reshape(-1, bc.LOCAL_OBS_DIM)).reshape(
            bsz, bc.NUM_AGENTS, bc.ACTION_DIM
        )
        chosen = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B, N]
        q_tot = chosen.sum(dim=1)

        with torch.no_grad():
            next_online = self.q(
                next_obs.reshape(-1, bc.LOCAL_OBS_DIM)
            ).reshape(bsz, bc.NUM_AGENTS, bc.ACTION_DIM)
            next_online = next_online.masked_fill(next_mask < 0.5, -1e9)
            next_act = next_online.argmax(dim=2, keepdim=True)      # [B, N, 1]
            next_target = self.target_q(
                next_obs.reshape(-1, bc.LOCAL_OBS_DIM)
            ).reshape(bsz, bc.NUM_AGENTS, bc.ACTION_DIM)
            next_q = next_target.gather(2, next_act).squeeze(-1).sum(dim=1)
            y = reward * self.reward_scale + self.gamma * (1.0 - done) * next_q

        loss = F.smooth_l1_loss(q_tot, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_q.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def state_dict(self):
        return {"q": self.q.state_dict(), "opt": self.opt.state_dict(),
                "learn_steps": self.learn_steps}


def epsilon_at(step, total):
    start, end, frac = 1.0, 0.05, 0.5
    t = min(1.0, step / max(1, total * frac))
    return start + (end - start) * t


def train(args):
    device = bc.resolve_device(args.device)
    bc.set_global_seed(args.seed)
    print(f"[VDN] device={device} stage={args.curriculum_stage} "
          f"episodes={args.episodes} local_obs_dim={bc.LOCAL_OBS_DIM}")

    agent = VDNAgent(args, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0
    est_total_steps = args.episodes * 250
    start = time.time()

    for episode in range(1, args.episodes + 1):
        env = bc.Environment(fixed_layout=False, episode_max_steps=args.max_steps)
        state = env.reset(curriculum_stage=args.curriculum_stage)
        lo, gs, am = bc.encode_state(state)
        ep_reward = 0.0
        losses = []

        for step in range(args.max_steps + bc.LANDING_GRACE):
            eps = epsilon_at(global_step, est_total_steps)
            actions = agent.act(lo, am, eps)
            nstate, _, done, sd = env.step([int(a) for a in actions])
            nlo, ngs, nam = bc.encode_state(nstate)
            agent.buffer.push({
                "obs": lo, "next_obs": nlo,
                "actions": actions.astype(np.float32),
                "next_mask": nam,
                "reward": np.float32(sd["team_reward"]),
                "done": np.float32(1.0 if done else 0.0),
            })
            ep_reward += sd["team_reward"]
            lo, gs, am = nlo, ngs, nam
            global_step += 1

            if global_step > args.warmup and global_step % args.train_every == 0:
                for _ in range(args.updates_per_step):
                    l = agent.learn()
                    if l is not None:
                        losses.append(l)
            if done:
                break

        rec = {
            "episode": episode,
            "global_step": global_step,
            "train_reward": ep_reward,
            "epsilon": epsilon_at(global_step, est_total_steps),
            "mean_loss": float(np.mean(losses)) if losses else None,
        }

        if args.eval_every and episode % args.eval_every == 0:
            ev = bc.evaluate(agent.greedy_select, args.eval_episodes,
                             args.seed + 1000, args.max_steps,
                             args.curriculum_stage)
            rec["evaluation"] = ev
            elapsed = time.time() - start
            print(f"[VDN] ep {episode:>6}  step {global_step:>9}  "
                  f"train_R {ep_reward:8.1f}  eval_R {ev['mean_reward']:8.1f}  "
                  f"deliv {ev['mean_delivery_rate']:.3f}  "
                  f"succ {ev['mission_success_rate']:.3f}  "
                  f"died {ev['mean_died']:.2f}  ({elapsed/60:.1f} min)")

        history.append(rec)

        if args.heartbeat and episode % args.heartbeat == 0 \
                and (not args.eval_every or episode % args.eval_every != 0):
            print(bc.heartbeat_line("VDN", episode, global_step, history,
                                    start, args.heartbeat), flush=True)

        if args.checkpoint_every and episode % args.checkpoint_every == 0:
            torch.save(agent.state_dict(), out_dir / "vdn_agent_FGCS.pth")
            bc.write_json(out_dir / "VDN_FGCS_metrics.json", {
                "algorithm": "VDN", "args": vars(args),
                "created": bc.utc_now(), "history": history,
            })

    final_eval = bc.evaluate(agent.greedy_select, max(args.eval_episodes, 50),
                             args.seed + 2000, args.max_steps,
                             args.curriculum_stage)
    torch.save(agent.state_dict(), out_dir / "vdn_agent_FGCS.pth")
    bc.write_json(out_dir / "VDN_FGCS_metrics.json", {
        "algorithm": "VDN", "args": vars(args), "created": bc.utc_now(),
        "history": history, "final_evaluation": final_eval,
    })
    bc.write_json(out_dir / "VDN_FGCS_eval.json", final_eval)
    print(f"[VDN] final evaluation: {final_eval}")


def main():
    parser = bc.base_arg_parser("VDN")
    args = parser.parse_args()
    if args.smoke_test:
        args.episodes = 3
        args.max_steps = 60
        args.warmup = 50
        args.batch_size = 16
        args.buffer = 2000
        args.eval_every = 3
        args.heartbeat = 2
        args.eval_episodes = 2
        args.checkpoint_every = 3
    train(args)


if __name__ == "__main__":
    main()
