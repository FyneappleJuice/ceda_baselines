"""
Shared infrastructure for the CEDA-FGCS MARL baselines (VDN, MADDPG, ...).

Every baseline trains and is evaluated on exactly the same environment,
episode budget, curriculum stage and headline metrics as the CEDA-FGCS
agent, so the numbers drop straight into the paper's comparison table.

The CEDA-FGCS environment lives in ``CEDA-FGCS.py`` (hyphenated, not an
importable module name), so it is loaded here by path with importlib and
re-exported.  Importing it only defines classes/constants -- the training
entry point is guarded by ``if __name__ == '__main__'``.
"""

import argparse
import importlib.util
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Load the CEDA-FGCS environment module by file path.
# --------------------------------------------------------------------------
os.environ.setdefault("CEDA_HEADLESS", "1")  # never open a pygame window here

# ``CEDA-FGCS.py`` imports the Unix-only ``resource`` module (used purely for
# peak-RSS diagnostics).  Provide a harmless stub so the baselines can also be
# smoke-tested on Windows; on Linux the real module is used.
if "resource" not in sys.modules:
    try:
        import resource  # noqa: F401
    except ImportError:
        import types

        _stub = types.ModuleType("resource")
        _stub.RUSAGE_SELF = 0
        _stub.getrusage = lambda who=0: types.SimpleNamespace(ru_maxrss=0)
        sys.modules["resource"] = _stub

_CEDA_PATH = Path(__file__).resolve().parent / "CEDA-FGCS.py"


def _load_ceda_module():
    if not _CEDA_PATH.exists():
        raise FileNotFoundError(f"Cannot find CEDA environment at {_CEDA_PATH}")
    spec = importlib.util.spec_from_file_location("ceda_fgcs", _CEDA_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ceda_fgcs"] = module
    spec.loader.exec_module(module)
    return module


ceda = _load_ceda_module()

# Re-export the constants the baselines need.
NUM_AGENTS = ceda.NUM_AGENTS
ACTION_DIM = ceda.ACTION_DIM
ACTION_HOVER = ceda.ACTION_HOVER
MAX_PATIENTS = ceda.MAX_PATIENTS
LOCAL_GRID_SIZE = ceda.LOCAL_GRID_SIZE
DRONE_STATE_DIM = ceda.DRONE_STATE_DIM
PATIENT_STATE_DIM = ceda.PATIENT_STATE_DIM
MISSION_STATE_DIM = ceda.MISSION_STATE_DIM
MAX_STEPS = ceda.MAX_STEPS
LANDING_GRACE = ceda.DEFAULT_LANDING_GRACE_STEPS
GAMMA = ceda.GAMMA
TD_REWARD_SCALE = ceda.TD_REWARD_SCALE
CURRICULUM_STAGES = ceda.CURRICULUM_STAGES
FULL_STAGE = len(CURRICULUM_STAGES) - 1
Environment = ceda.Environment

_GRID_FLAT = 3 * LOCAL_GRID_SIZE * LOCAL_GRID_SIZE
# Per-agent (decentralised) observation fed to every actor / local Q network.
LOCAL_OBS_DIM = (
    DRONE_STATE_DIM              # this drone's own state row
    + _GRID_FLAT                 # this drone's egocentric 3x(21x21) hazard grid
    + MISSION_STATE_DIM          # shared mission summary
    + MAX_PATIENTS * PATIENT_STATE_DIM  # patient roster (padded)
    + MAX_PATIENTS               # patient active mask
    + NUM_AGENTS                 # one-hot agent id
)
# Centralised state used by mixers / centralised critics (training only).
GLOBAL_STATE_DIM = (
    NUM_AGENTS * DRONE_STATE_DIM
    + MISSION_STATE_DIM
    + MAX_PATIENTS * PATIENT_STATE_DIM
    + MAX_PATIENTS
)


# --------------------------------------------------------------------------
# Observation encoding.
# --------------------------------------------------------------------------
def encode_state(state):
    """Convert an Environment observation dict into flat baseline tensors.

    Returns
    -------
    local_obs      : float32 [NUM_AGENTS, LOCAL_OBS_DIM]
    global_state   : float32 [GLOBAL_STATE_DIM]
    action_masks   : float32 [NUM_AGENTS, ACTION_DIM]  (1 == available)
    """
    drones = state["drones"].astype(np.float32)                 # [N, 22]
    grids = state["local_grids"].astype(np.float32).reshape(NUM_AGENTS, -1)
    mission = state["mission"].astype(np.float32)               # [12]
    patients = state["patients"].astype(np.float32).reshape(-1)  # [P*10]
    pmask = state["patient_masks"].astype(np.float32)           # [P]
    amask = state["action_masks"].astype(np.float32)            # [N, A]

    agent_id = np.eye(NUM_AGENTS, dtype=np.float32)
    shared = np.concatenate([mission, patients, pmask])
    shared_tiled = np.broadcast_to(shared, (NUM_AGENTS, shared.shape[0]))
    local_obs = np.concatenate(
        [drones, grids, shared_tiled, agent_id], axis=1
    ).astype(np.float32)

    global_state = np.concatenate(
        [drones.reshape(-1), mission, patients, pmask]
    ).astype(np.float32)
    return local_obs, global_state, amask


def masked_greedy(q_values, action_masks):
    """Greedy action per agent under availability masks. q_values: [N, A]."""
    neg = np.where(action_masks > 0.5, q_values, -1e9)
    return neg.argmax(axis=1).astype(np.int64)


def masked_random_actions(action_masks):
    out = np.empty(NUM_AGENTS, dtype=np.int64)
    for i in range(NUM_AGENTS):
        valid = np.flatnonzero(action_masks[i] > 0.5)
        out[i] = int(random.choice(valid.tolist())) if valid.size else ACTION_HOVER
    return out


# --------------------------------------------------------------------------
# Networks.
# --------------------------------------------------------------------------
def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------
# Replay buffer (uniform).  Stores whole joint transitions.
# --------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = []
        self.pos = 0

    def __len__(self):
        return len(self.data)

    def push(self, transition):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.data), size=batch_size)
        batch = [self.data[i] for i in idx]
        return _collate(batch)


def _collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.as_tensor(
            np.stack([b[k] for b in batch]), dtype=torch.float32
        )
    return out


# --------------------------------------------------------------------------
# Evaluation harness -- mirrors CEDA-FGCS.evaluate_policy headline metrics.
# --------------------------------------------------------------------------
def evaluate(select_fn, episodes, seed, max_steps, curriculum_stage):
    """Greedy, epsilon=0 evaluation, identical protocol for every method.

    ``select_fn(state_dict) -> iterable[int] of length NUM_AGENTS`` where
    ``state_dict`` is the raw Environment observation.  Baseline agents wrap
    this with their own :func:`encode_state` call; the CEDA-FGCS agent is
    passed straight through.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        rewards, delivered, died, landed, success = [], [], [], [], []
        triage_eff, delivery_rate = [], []
        obstacle_col, agent_col, lengths = [], [], []
        spawned_counts = []
        for _ in range(episodes):
            env = Environment(fixed_layout=False, episode_max_steps=max_steps)
            state = env.reset(curriculum_stage=curriculum_stage)
            total_r = 0.0
            oc = ac = 0
            steps = 0
            for step in range(max_steps + LANDING_GRACE):
                actions = select_fn(state)
                state, _, done, sd = env.step([int(a) for a in actions])
                total_r += sd["team_reward"]
                oc += sd["obstacle_collisions"]
                ac += sd["agent_collisions"]
                steps = step + 1
                if done:
                    break
            spawned = sum(env.patient_active[: env.episode_max_patients])
            deliv = sum(env.patients_actually_delivered[: env.episode_max_patients])
            outcome = env.mission_outcome_metrics()
            rewards.append(total_r)
            delivered.append(deliv)
            died.append(sum(env.patients_died[: env.episode_max_patients]))
            landed.append(sum(env.landed))
            success.append(1 if env.mission_success() else 0)
            triage_eff.append(outcome["triage_efficiency"])
            delivery_rate.append(deliv / max(1, spawned))
            spawned_counts.append(spawned)
            obstacle_col.append(oc)
            agent_col.append(ac)
            lengths.append(steps)

        return {
            "episodes": int(episodes),
            "curriculum_stage": int(curriculum_stage),
            "suite_seed": int(seed),
            "mean_reward": float(np.mean(rewards)),
            "mean_spawned": float(np.mean(spawned_counts)),
            "mean_delivered": float(np.mean(delivered)),
            "mean_delivery_rate": float(np.mean(delivery_rate)),
            "mean_died": float(np.mean(died)),
            "mean_landed": float(np.mean(landed)),
            "mission_success_rate": float(np.mean(success)),
            "mean_triage_efficiency": float(np.mean(triage_eff)),
            "mean_obstacle_collisions": float(np.mean(obstacle_col)),
            "mean_agent_collisions": float(np.mean(agent_col)),
            "mean_episode_length": float(np.mean(lengths)),
        }
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


# --------------------------------------------------------------------------
# Misc.
# --------------------------------------------------------------------------
def resolve_device(name):
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[baseline] wrote {path}")


def base_arg_parser(algo_name):
    p = argparse.ArgumentParser(description=f"Train the {algo_name} baseline "
                                            f"on the CEDA-FGCS environment.")
    p.add_argument("--episodes", type=int, default=12000)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    p.add_argument("--output-dir", default=os.environ.get("CEDA_OUTPUT_DIR", "."))
    p.add_argument("--curriculum-stage", type=int, default=FULL_STAGE,
                   help=f"0..{FULL_STAGE}; default {FULL_STAGE} (full mission).")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--train-every", type=int, default=1,
                   help="Gradient update every N environment steps.")
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--buffer", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--reward-scale", type=float, default=TD_REWARD_SCALE)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--smoke-test", action="store_true",
                   help="Two short episodes to check the wiring end to end.")
    return p


def utc_now():
    return datetime.now(timezone.utc).isoformat()
