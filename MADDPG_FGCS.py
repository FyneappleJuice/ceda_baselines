"""
MADDPG (Lowe et al. 2017) baseline for CEDA-FGCS.

Multi-Agent Deep Deterministic Policy Gradient adapted to the discrete
6-action drone control space in the standard way:

  * per-agent actor pi_i(o_i) -> action logits; discrete actions are made
    differentiable with the straight-through Gumbel-Softmax estimator
    (Jang et al. 2017) exactly as in the original MADDPG discrete-action
    experiments;
  * per-agent centralised critic Q_i(x, a_1..a_N) where x is the full
    environment state and a_j the (one-hot) action of every drone -- this
    is the "centralised training, decentralised execution" critic;
  * soft-updated target actors/critics, shared team reward, availability
    masks applied to every actor (behaviour, target and policy-gradient).

Run:
  python MADDPG_FGCS.py --smoke-test
  python MADDPG_FGCS.py --episodes 12000 --device cuda:0 --output-dir runs/maddpg
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import baseline_common as bc

NEG = -1e9


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=(512, 256)):
        super().__init__()
        self.net = bc.mlp((obs_dim, *hidden, action_dim), act=nn.ReLU)

    def forward(self, obs, avail_mask=None):
        logits = self.net(obs)
        if avail_mask is not None:
            logits = logits.masked_fill(avail_mask < 0.5, NEG)
        return logits


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents, hidden=(512, 512, 256)):
        super().__init__()
        self.net = bc.mlp((state_dim + action_dim * n_agents, *hidden, 1),
                          act=nn.ReLU)

    def forward(self, state, joint_actions):
        return self.net(torch.cat([state, joint_actions], dim=-1)).squeeze(-1)


def straight_through_gumbel(logits, tau=1.0):
    return F.gumbel_softmax(logits, tau=tau, hard=True)


def masked_onehot_greedy(logits):
    idx = logits.argmax(dim=-1)
    return F.one_hot(idx, num_classes=logits.shape[-1]).float()


class MADDPGAgent:
    def __init__(self, args, device):
        self.device = device
        self.gamma = bc.GAMMA
        self.batch_size = args.batch_size
        self.reward_scale = args.reward_scale
        self.tau = 0.01
        self.grad_clip = 10.0
        self.gumbel_tau = 1.0
        self.reg = 1e-3
        N, A = bc.NUM_AGENTS, bc.ACTION_DIM

        self.actors = nn.ModuleList(
            [Actor(bc.LOCAL_OBS_DIM, A) for _ in range(N)]
        ).to(device)
        self.critics = nn.ModuleList(
            [Critic(bc.GLOBAL_STATE_DIM, A, N) for _ in range(N)]
        ).to(device)
        self.target_actors = nn.ModuleList(
            [Actor(bc.LOCAL_OBS_DIM, A) for _ in range(N)]
        ).to(device)
        self.target_critics = nn.ModuleList(
            [Critic(bc.GLOBAL_STATE_DIM, A, N) for _ in range(N)]
        ).to(device)
        self.target_actors.load_state_dict(self.actors.state_dict())
        self.target_critics.load_state_dict(self.critics.state_dict())

        self.actor_opt = torch.optim.Adam(self.actors.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(),
                                           lr=args.lr)
        self.buffer = bc.ReplayBuffer(args.buffer)
        self.learn_steps = 0

    # -- acting ---------------------------------------------------------
    @torch.no_grad()
    def act(self, local_obs, action_masks, epsilon):
        obs = torch.as_tensor(local_obs, dtype=torch.float32, device=self.device)
        am = torch.as_tensor(action_masks, dtype=torch.float32,
                             device=self.device)
        actions = np.empty(bc.NUM_AGENTS, dtype=np.int64)
        for i in range(bc.NUM_AGENTS):
            logits = self.actors[i](obs[i], am[i])
            actions[i] = int(logits.argmax().item())
        if epsilon > 0.0:
            rand = bc.masked_random_actions(action_masks)
            take = np.random.random(bc.NUM_AGENTS) < epsilon
            actions = np.where(take, rand, actions).astype(np.int64)
        return actions

    def greedy_select(self, state):
        lo, _, am = bc.encode_state(state)
        return self.act(lo, am, epsilon=0.0)

    # -- learning -----------------------------------------------------
    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None
        b = self.buffer.sample(self.batch_size)
        N, A = bc.NUM_AGENTS, bc.ACTION_DIM
        obs = b["obs"].to(self.device)                 # [B, N, O]
        next_obs = b["next_obs"].to(self.device)       # [B, N, O]
        state = b["state"].to(self.device)             # [B, G]
        next_state = b["next_state"].to(self.device)   # [B, G]
        acts = b["actions_onehot"].to(self.device)     # [B, N, A]
        next_mask = b["next_mask"].to(self.device)     # [B, N, A]
        reward = b["reward"].to(self.device)           # [B]
        done = b["done"].to(self.device)               # [B]
        bsz = obs.shape[0]
        joint_actions = acts.reshape(bsz, N * A)

        # --- target joint next actions from target actors -------------
        with torch.no_grad():
            next_oh = []
            for j in range(N):
                logits = self.target_actors[j](next_obs[:, j, :],
                                               next_mask[:, j, :])
                next_oh.append(masked_onehot_greedy(logits))
            next_joint = torch.cat(next_oh, dim=-1)     # [B, N*A]

        critic_losses = []
        for i in range(N):
            with torch.no_grad():
                target_q = self.target_critics[i](next_state, next_joint)
                y = (reward * self.reward_scale
                     + self.gamma * (1.0 - done) * target_q)
            q = self.critics[i](state, joint_actions)
            critic_losses.append(F.smooth_l1_loss(q, y))
        critic_loss = torch.stack(critic_losses).sum()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), self.grad_clip)
        self.critic_opt.step()

        # --- actor updates ------------------------------------------
        actor_losses = []
        for i in range(N):
            pieces = []
            reg = 0.0
            for j in range(N):
                if j == i:
                    logits = self.actors[i](obs[:, i, :])
                    pieces.append(straight_through_gumbel(logits,
                                                          self.gumbel_tau))
                    reg = (logits ** 2).mean()
                else:
                    pieces.append(acts[:, j, :])
            joint = torch.cat(pieces, dim=-1)
            actor_losses.append(-self.critics[i](state, joint).mean()
                                + self.reg * reg)
        actor_loss = torch.stack(actor_losses).sum()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actors.parameters(), self.grad_clip)
        self.actor_opt.step()

        self._soft_update(self.actors, self.target_actors)
        self._soft_update(self.critics, self.target_critics)
        self.learn_steps += 1
        return {"critic_loss": float(critic_loss.item()),
                "actor_loss": float(actor_loss.item())}

    def state_dict(self):
        return {
            "actors": self.actors.state_dict(),
            "critics": self.critics.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "learn_steps": self.learn_steps,
        }


def epsilon_at(step, total):
    start, end, frac = 1.0, 0.05, 0.5
    t = min(1.0, step / max(1, total * frac))
    return start + (end - start) * t


def train(args):
    device = bc.resolve_device(args.device)
    bc.set_global_seed(args.seed)
    print(f"[MADDPG] device={device} stage={args.curriculum_stage} "
          f"episodes={args.episodes} local_obs_dim={bc.LOCAL_OBS_DIM} "
          f"global_state_dim={bc.GLOBAL_STATE_DIM}")

    agent = MADDPGAgent(args, device)
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
        c_losses, a_losses = [], []

        for step in range(args.max_steps + bc.LANDING_GRACE):
            eps = epsilon_at(global_step, est_total_steps)
            actions = agent.act(lo, am, eps)
            nstate, _, done, sd = env.step([int(a) for a in actions])
            nlo, ngs, nam = bc.encode_state(nstate)
            onehot = np.eye(bc.ACTION_DIM, dtype=np.float32)[actions]
            agent.buffer.push({
                "obs": lo, "next_obs": nlo,
                "state": gs, "next_state": ngs,
                "actions_onehot": onehot,
                "next_mask": nam,
                "reward": np.float32(sd["team_reward"]),
                "done": np.float32(1.0 if done else 0.0),
            })
            ep_reward += sd["team_reward"]
            lo, gs, am = nlo, ngs, nam
            global_step += 1

            if global_step > args.warmup and global_step % args.train_every == 0:
                for _ in range(args.updates_per_step):
                    out = agent.learn()
                    if out is not None:
                        c_losses.append(out["critic_loss"])
                        a_losses.append(out["actor_loss"])
            if done:
                break

        rec = {
            "episode": episode,
            "global_step": global_step,
            "train_reward": ep_reward,
            "epsilon": epsilon_at(global_step, est_total_steps),
            "mean_critic_loss": float(np.mean(c_losses)) if c_losses else None,
            "mean_actor_loss": float(np.mean(a_losses)) if a_losses else None,
        }

        if args.eval_every and episode % args.eval_every == 0:
            ev = bc.evaluate(agent.greedy_select, args.eval_episodes,
                             args.seed + 1000, args.max_steps,
                             args.curriculum_stage)
            rec["evaluation"] = ev
            elapsed = time.time() - start
            print(f"[MADDPG] ep {episode:>6}  step {global_step:>9}  "
                  f"train_R {ep_reward:8.1f}  eval_R {ev['mean_reward']:8.1f}  "
                  f"deliv {ev['mean_delivery_rate']:.3f}  "
                  f"succ {ev['mission_success_rate']:.3f}  "
                  f"died {ev['mean_died']:.2f}  ({elapsed/60:.1f} min)")

        history.append(rec)

        if args.checkpoint_every and episode % args.checkpoint_every == 0:
            torch.save(agent.state_dict(), out_dir / "maddpg_agent_FGCS.pth")
            bc.write_json(out_dir / "MADDPG_FGCS_metrics.json", {
                "algorithm": "MADDPG", "args": vars(args),
                "created": bc.utc_now(), "history": history,
            })

    final_eval = bc.evaluate(agent.greedy_select, max(args.eval_episodes, 50),
                             args.seed + 2000, args.max_steps,
                             args.curriculum_stage)
    torch.save(agent.state_dict(), out_dir / "maddpg_agent_FGCS.pth")
    bc.write_json(out_dir / "MADDPG_FGCS_metrics.json", {
        "algorithm": "MADDPG", "args": vars(args), "created": bc.utc_now(),
        "history": history, "final_evaluation": final_eval,
    })
    bc.write_json(out_dir / "MADDPG_FGCS_eval.json", final_eval)
    print(f"[MADDPG] final evaluation: {final_eval}")


def main():
    parser = bc.base_arg_parser("MADDPG")
    args = parser.parse_args()
    if args.smoke_test:
        args.episodes = 3
        args.max_steps = 60
        args.warmup = 50
        args.batch_size = 16
        args.buffer = 2000
        args.eval_every = 3
        args.eval_episodes = 2
        args.checkpoint_every = 3
    train(args)


if __name__ == "__main__":
    main()
