import argparse
import os
import random
import math
import numpy as np 
import sys
import signal
from pathlib import Path

HEADLESS_MODE = (
    "--headless" in sys.argv
    or os.environ.get("CEDA_HEADLESS", "").lower() in {"1", "true", "yes"}
    or (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))
)

if HEADLESS_MODE:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import json
from datetime import datetime
import heapq
import hashlib
import platform
import time
import tracemalloc
import resource
from collections import deque
from contextlib import nullcontext

# Configuration
MODEL_VERSION = "CEDA-FGCS"

GRID_SIZE = 100
CELL_SIZE = 10
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

NUM_OBSTACLES = 800
NUM_WIND_ZONE_RECTANGLES = 6
NUM_LOW_SIGNAL_ZONE_RECTANGLES = 5
HAZARD_RECTANGLE_MIN_WIDTH = 6
HAZARD_RECTANGLE_MAX_WIDTH = 12
HAZARD_RECTANGLE_MIN_HEIGHT = 6
HAZARD_RECTANGLE_MAX_HEIGHT = 10

HAZARD_ROUTE_MIN_FRACTION = 0.30
HAZARD_ROUTE_MAX_FRACTION = 0.75
HAZARD_MAX_RECTANGLE_OVERLAP = 0.20
HAZARD_MAX_SAFE_DETOUR_RATIO = 1.50
HAZARD_PLACEMENT_ATTEMPTS = 24
WIND_APPEAR_INTERVAL = 1500
LOW_SIGNAL_APPEAR_INTERVAL = 1500

LOCAL_GRID_RADIUS = 10
LOCAL_GRID_SIZE = 2 * LOCAL_GRID_RADIUS + 1

MAX_PATIENT_TIMER = 300

NUM_AGENTS = 5
NUM_INITIAL_PATIENTS = 20
MAX_PATIENTS = 50
NEW_PATIENT_SPAWN_INTERVAL = 60
PATIENT_SPAWN_INTERVAL_JITTER = 5
MIN_PATIENT_SPAWN_INTERVAL = (
    NEW_PATIENT_SPAWN_INTERVAL - PATIENT_SPAWN_INTERVAL_JITTER
)
MAX_PATIENT_SPAWN_INTERVAL = (
    NEW_PATIENT_SPAWN_INTERVAL + PATIENT_SPAWN_INTERVAL_JITTER
)
FINAL_PATIENT_SPAWN_STEP = 400
MIN_PATIENT_SPAWN_BATCH = 4
MAX_PATIENT_SPAWN_BATCH = 6
LEVEL1_PATIENT_WEIGHT = 1
LEVEL2_PATIENT_WEIGHT = 2
LEVEL3_PATIENT_WEIGHT = 3
MAX_PATIENT_WEIGHT = 3

SURVIVAL_DECAY_RANGES = {
    1: (0.02, 0.05),
    2: (0.05, 0.10),
    3: (0.10, 0.20),
}
SURVIVAL_OFFSET_RANGES = {
    1: (3.0, 5.0),
    2: (2.0, 3.5),
    3: (1.0, 2.5),
}
SERIOUS_SURVIVAL_THRESHOLD_RANGE = (0.40, 0.70)
CRITICAL_SURVIVAL_THRESHOLD_RANGE = (0.10, 0.30)

ACTION_NORTH = 0
ACTION_SOUTH = 1
ACTION_WEST = 2
ACTION_EAST = 3
ACTION_HOVER = 4
ACTION_LAND = 5
ACTION_DIM = 6
ACTION_NAMES = ("north", "south", "west", "east", "hover", "land")
MOVEMENT_OFFSETS = ((0, -1), (0, 1), (-1, 0), (1, 0))
DRONE_STATE_DIM = 16 + ACTION_DIM
PATIENT_STATE_DIM = 10

NUM_EPISODES = 12000
MAX_STEPS = 800
DEFAULT_LANDING_GRACE_STEPS = 400

MAX_BATTERY = 100.0
BATTERY_DRAIN_PER_STEP = 0.20
BATTERY_DRAIN_IN_WIND = 2.30
BATTERY_DRAIN_AT_LANDING_ZONE = 0.02
LOW_BATTERY_THRESHOLD = 20.0
SAFE_RETURN_BATTERY_BUFFER = 18.0

ENERGY_RETURN_TRIGGER_MARGIN = 0.0
RETURN_ENERGY_RISK_MULTIPLIER = 1.0

POST_DEPLETION_REMINDER_STEPS = 2
POST_DEPLETION_REMINDER_PENALTY = -20.0

DEAD_LANDING_PENALTY = -300.0
LOW_SIGNAL_FAILURE_PROB = 0.5
WIND_MOVEMENT_FAILURE_PROB = 0.15

LEARNING_RATE = 5.0e-5
GAMMA = 0.995
BATCH_SIZE = 256
BUFFER_CAPACITY = 500000
REPLAY_WARMUP = 50000
N_STEP = 5
PER_ALPHA = 0.4
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_EPSILON = 1e-5
PER_PRIORITY_MAX = 15.0
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_MID = 0.15

EPSILON_MID_STEP = 1250000
EPSILON_END_STEP = 3000000

CURRICULUM_EPSILON_RESET = 0.25
CURRICULUM_EPSILON_FLOOR = 0.03
CURRICULUM_EPSILON_DECAY_EPISODES = 750
TARGET_UPDATE_STEPS = 625
GRADIENT_CLIP = 7.5
TRAIN_EVERY_STEPS = 32
UPDATES_PER_TRAIN = 1
TD_REWARD_SCALE = 0.05
LOCAL_TD_LOSS_WEIGHT = 0.25
REPLAY_UNIFORM_FRACTION = 0.20
REPLAY_RESCUE_FRACTION = 0.60
REPLAY_LANDING_FRACTION = 0.20
ATTENTION_DIAGNOSTIC_INTERVAL = 250
ROLLOUT_WORKERS = 8
ROLLOUT_CHUNK_STEPS = 64
ROLLOUT_QUEUE_SIZE = 16
POLICY_MAX_LAG = 8

CURRICULUM_STAGES = (
    {
        'stage': 0, 'name': 'five_patient_navigation_100x100',
        'initial_patients': 5, 'max_patients': 5,
        'dynamic_spawning': False, 'maximum_path_distance': 20,
        'maximum_landing_path_distance': 20,
        'patient_timer': 300, 'spawn_interval': 60,
        'spawn_jitter': 5, 'spawn_batch_min': 4, 'spawn_batch_max': 6,
        'final_spawn_step': 400,
        'hazard_fraction': 0.50, 'hazard_penalty_scale': 0.25,
        'minimum_episodes': 250,
        'maximum_episodes': 750, 'delivery_gate': 0.70,
        'triage_gate': 0.70, 'landing_gate': 0.85,
        'lower_triage_gate': 0.50, 'collision_rate_gate': 0.020,
        'acuity_priority_gate': 0.0, 'priority_fairness_gate': 0.65,
        'triage_ordering_gate': 0.0,
        'wind_avoidance_gate': 0.50, 'low_signal_avoidance_gate': 0.50,
        'minimum_hazard_opportunities': 20,
        'response_ordering_gate': 0.0,
        'battery_depletion_free_gate': 0.90,
        'reserve_violation_gate': 0.050,
    },
    {
        'stage': 1, 'name': 'twenty_five_patient_cross_layer_100x100',
        'initial_patients': 15, 'max_patients': 25,
        'dynamic_spawning': True, 'maximum_path_distance': 64,
        'maximum_landing_path_distance': 45,
        'patient_timer': 260, 'spawn_interval': 45,
        'spawn_jitter': 4, 'spawn_batch_min': 5, 'spawn_batch_max': 7,
        'final_spawn_step': 120,
        'hazard_fraction': 1.00, 'hazard_penalty_scale': 0.90,
        'minimum_episodes': 500,
        'maximum_episodes': 1750, 'delivery_gate': 0.68,
        'triage_gate': 0.72, 'landing_gate': 0.85,
        'lower_triage_gate': 0.40, 'collision_rate_gate': 0.010,
        'acuity_priority_gate': 0.50, 'priority_fairness_gate': 0.65,
        'triage_ordering_gate': 0.50,
        'wind_avoidance_gate': 0.90, 'low_signal_avoidance_gate': 0.90,
        'minimum_hazard_opportunities': 50,
        'response_ordering_gate': 0.55,
        'battery_depletion_free_gate': 0.93,
        'reserve_violation_gate': 0.030,
    },
    {
        'stage': 2, 'name': 'full_5_drone_50_patient_100x100_cross_layer',
        'initial_patients': NUM_INITIAL_PATIENTS, 'max_patients': MAX_PATIENTS,
        'dynamic_spawning': True, 'maximum_path_distance': 0,
        'maximum_landing_path_distance': 50,
        'patient_timer': 220, 'spawn_interval': 25,
        'spawn_jitter': 2, 'spawn_batch_min': 8, 'spawn_batch_max': 10,
        'final_spawn_step': 90,
        'hazard_fraction': 1.00, 'hazard_penalty_scale': 1.75,
        'minimum_episodes': 0,
        'maximum_episodes': NUM_EPISODES, 'delivery_gate': 0.70,
        'triage_gate': 0.75, 'landing_gate': 0.85,
        'lower_triage_gate': 0.40, 'collision_rate_gate': 0.003,
        'acuity_priority_gate': 0.53, 'priority_fairness_gate': 0.70,
        'triage_ordering_gate': 0.54,
        'wind_avoidance_gate': 0.98, 'low_signal_avoidance_gate': 0.98,
        'minimum_hazard_opportunities': 75,
        'response_ordering_gate': 0.62,
        'battery_depletion_free_gate': 0.97,
        'reserve_violation_gate': 0.010,
    },
)
CURRICULUM_REQUIRED_PASSES = 2
CURRICULUM_FORCE_PROMOTION_AT_MAXIMUM = True
CURRICULUM_CURRENT_PROBABILITY = 0.80
CURRICULUM_PREVIOUS_PROBABILITY = 0.10
CURRICULUM_FULL_PROBABILITY = 0.10

for _stage in CURRICULUM_STAGES:
    _landing_radius = int(_stage['maximum_landing_path_distance'])
    _required_start_return_battery = (
        _landing_radius
        * BATTERY_DRAIN_PER_STEP
        * RETURN_ENERGY_RISK_MULTIPLIER
        + SAFE_RETURN_BATTERY_BUFFER
        + ENERGY_RETURN_TRIGGER_MARGIN
    )
    if _required_start_return_battery > MAX_BATTERY + 1e-9:
        raise RuntimeError(
            f"Curriculum stage {_stage['stage']} requires "
            f"{_required_start_return_battery:.3f} battery units to keep its "
            f"start-to-pad route above the return trigger, exceeding the "
            f"{MAX_BATTERY:.3f}-unit budget"
        )
del _stage, _landing_radius, _required_start_return_battery

EVALUATION_EVERY_EPISODES = 250
EVALUATION_EPISODES = 20

EVALUATION_TRACE_EPISODES = 1
EVALUATION_TRACE_MAX_RECORDS = 128
MAX_RECORDED_OBSTACLE_EVENTS_PER_EPISODE = 128

ENTITY_EMBED_DIM = 64
ATTENTION_HEADS = 4
SET_ATTENTION_BLOCKS = 2
SELF_EMBED_DIM = 128
GRID_EMBED_DIM = 64
AGENT_ID_EMBED_DIM = 16
CENTRAL_GRID_EMBED_DIM = 32
MISSION_STATE_DIM = 12
FUSION_HIDDEN_DIM = 256
MIXER_EMBED_DIM = 64
GLOBAL_EMBED_DIM = 128
AGENT_MIX_CONTEXT_DIM = 128

DELIVERY_TRIAGE_ALPHA = 0.50
GOAL_REWARD = 10.0
DELIVERY_COMPLETION_BONUS = 5.0
STEP_PENALTY = -0.02
CLEAN_STEP_BONUS = 0.01
COLLISION_PENALTY = -45.0
AGENT_COLLISION_PENALTY = -45.0
DOMINATED_OBSTACLE_SELECTION_PENALTY = -8.0
DOMINATED_AGENT_CONFLICT_PENALTY = -8.0
BATTERY_DEPLETION_PENALTY = -100.0
LOW_BATTERY_PENALTY = -0.1
ENERGY_USAGE_PENALTY_PER_UNIT = -0.05
SAFE_RETURN_RESERVE_PENALTY = -3.00
ENERGY_MARGIN_SHAPING_FACTOR = 5.0
WIND_PENALTY = -3.00
LOW_SIGNAL_PENALTY = -4.00
WIND_ENTRY_PENALTY = -4.00
LOW_SIGNAL_ENTRY_PENALTY = -5.00

WIND_DOMINATED_SELECTION_PENALTY = -6.00
WIND_SHORTCUT_SELECTION_PENALTY = -3.00
LOW_SIGNAL_DOMINATED_SELECTION_PENALTY = -8.00
LOW_SIGNAL_SHORTCUT_SELECTION_PENALTY = -4.00

SHAPING_FACTOR = 0.5
PATIENT_DEATH_PENALTY = -30.0
RESPONSE_WAIT_PENALTY_PER_PATIENT = -0.015
RESPONSE_TIME_DELIVERY_REWARD = 12.0
LANDING_REWARD = 30.0
EARLY_LANDING_PENALTY = -2.0
LAND_WRONG_PENALTY = -0.1
RESCUE_OUTCOME_LINEAR_REWARD = 100.0
RESCUE_OUTCOME_QUADRATIC_REWARD = 100.0
SAFE_RETURN_BASE_REWARD = 20.0
SAFE_RETURN_QUALITY_REWARD = 50.0
MISSION_SUCCESS_REWARD = 125.0
MISSION_FAILURE_PENALTY = -40.0
FAIRNESS_OUTCOME_REWARD = 50.0
TRIAGE_ORDERING_OUTCOME_REWARD = 50.0
TRIAGE_RESPONSE_OUTCOME_REWARD = 100.0

PRIORITY_SERVICE_POTENTIAL_SCALE = 300.0
PRIORITY_DEATH_PENALTY_GROWTH = 0.25
APPLICATION_DELIVERY_SUCCESS_THRESHOLD = 0.70
APPLICATION_TRIAGE_SUCCESS_THRESHOLD = 0.80
APPLICATION_LOWER_TRIAGE_FLOOR = 0.40
APPLICATION_ACUITY_PRIORITY_THRESHOLD = 0.55

TRIAGE_CLASS_DELIVERY_TARGETS = {
    1: 0.50,
    2: 0.70,
    3: 0.90,
}
APPLICATION_PRIORITY_FAIRNESS_THRESHOLD = 0.60
APPLICATION_TRIAGE_ORDERING_THRESHOLD = 0.55
APPLICATION_TRIAGE_RESPONSE_THRESHOLD = 0.60
CLOSENESS_PENALTY = -0.5
CLOSENESS_RADIUS  = 4
HOVER_PENALTY = -1.0
LANDING_HOVER_PENALTY = -1.0
ENERGY_STANDBY_HOVER_PENALTY = -0.02
COLLISION_STREAK_CAP = 4
COLLISION_STREAK_PENALTY_GROWTH = 1.0
MIXER_MIN_RAW_WEIGHT = 0.10
REWARD_COMPONENT_NAMES = (
    'step', 'clean', 'delivery', 'patient_death',
    'obstacle_collision', 'agent_collision', 'battery',
    'wind', 'low_signal', 'wind_routing', 'low_signal_routing',
    'low_battery', 'energy_navigation', 'response_time', 'landing', 'hover',
    'closeness', 'potential', 'rescue_outcome', 'fairness',
    'triage_ordering', 'triage_response', 'safe_return', 'mission'
)

EVENT_DELIVERY = 1
EVENT_COLLISION = 2
EVENT_PROGRESS = 4
EVENT_LANDING = 8
EVENT_TERMINAL = 16
EVENT_LANDING_PHASE = 32
EVENT_OBSTACLE_COLLISION = 64
EVENT_HAZARD = 128
EVENT_PRIORITY_MULTIPLIERS = {
    EVENT_DELIVERY: 1.75,
    EVENT_COLLISION: 4.00,
    EVENT_PROGRESS: 1.10,
    EVENT_LANDING: 2.0,
    EVENT_TERMINAL: 1.10,
    
    
    EVENT_LANDING_PHASE: 1.0,
    EVENT_OBSTACLE_COLLISION: 4.50,
    
    
    EVENT_HAZARD: 4.00,
}

# Environment and mission dynamics
class Environment:

    def __init__(
        self,
        fixed_layout,
        episode_max_steps=MAX_STEPS,
        landing_grace_steps=DEFAULT_LANDING_GRACE_STEPS,
    ):
        self.grid_size = GRID_SIZE
        self.cell_size = CELL_SIZE
        self.fixed_layout = fixed_layout
        self.episode_max_steps = episode_max_steps
        self.landing_grace_steps = int(landing_grace_steps)
        self.landing_deadline = -1
        self.landing_completion_step = -1
        self.curriculum_stage = len(CURRICULUM_STAGES) - 1
        self.curriculum_stage_name = CURRICULUM_STAGES[-1]['name']
        self.episode_initial_patients = NUM_INITIAL_PATIENTS
        self.episode_max_patients = MAX_PATIENTS
        self.dynamic_spawning = True
        self.hazard_fraction = 1.0
        default_stage = CURRICULUM_STAGES[-1]
        self.hazard_penalty_scale = float(
            default_stage['hazard_penalty_scale']
        )
        self.episode_patient_timer = int(default_stage['patient_timer'])
        self.patient_spawn_interval = int(default_stage['spawn_interval'])
        self.patient_spawn_jitter = int(default_stage['spawn_jitter'])
        self.minimum_patient_spawn_batch = int(
            default_stage['spawn_batch_min']
        )
        self.maximum_patient_spawn_batch = int(
            default_stage['spawn_batch_max']
        )
        self.final_patient_spawn_step = int(
            default_stage['final_spawn_step']
        )

        
        self.obstacles = set(self.generate_obstacles())
        self.obstacle_raster = self.cells_to_raster(self.obstacles)

        
        self.start_positions = [(2, 2), (2, 26), (4, 38), (6, 52), (8, 64)]
        self.patient_positions = [
            (26, 26), (26, 2), (50, 50), (50, 2),
            (70, 20), (20, 70), (80, 60), (60, 80),
        ]
        self.landing_zones = [
            (96, 96), (96, 90), (92, 92), (92, 96), (94, 94)
        ]

        
        
        
        if (NUM_AGENTS > len(self.start_positions)
                or MAX_PATIENTS > len(self.patient_positions)):
            self.generate_random_positions(
                maximum_path_distance=int(
                    default_stage['maximum_path_distance']
                ),
                active_patient_count=MAX_PATIENTS,
                maximum_landing_path_distance=int(
                    default_stage['maximum_landing_path_distance']
                ),
            )

        self.agents = list(self.start_positions)
        self.batteries = [MAX_BATTERY] * NUM_AGENTS
        self.landed = [False] * NUM_AGENTS
        self.battery_depleted = [False] * NUM_AGENTS
        self.drone_died = [False] * NUM_AGENTS
        self.death_reminder_steps_remaining = [0] * NUM_AGENTS
        self.dead_landing_penalized = [False] * NUM_AGENTS
        self.previous_actions = [ACTION_HOVER] * NUM_AGENTS
        self.previous_displacements = [(0, 0)] * NUM_AGENTS
        self.previous_collision_flags = [False] * NUM_AGENTS
        self.collision_streaks = [0] * NUM_AGENTS
        self.obstacle_collision_streaks = [0] * NUM_AGENTS
        self.collision_pair_streaks = {}
        self.landing_zone_arrival_steps = [-1] * NUM_AGENTS
        self.episode_mode = "full_mission"
        self.curriculum_max_distance = 0
        self.curriculum_max_landing_distance = int(
            default_stage['maximum_landing_path_distance']
        )
        self.curriculum_initial_distances = [0] * NUM_AGENTS
        self.curriculum_start_step = 0
        self.agent_path_lengths = [0] * NUM_AGENTS
        self.agent_unique_positions = [set([self.agents[i]]) for i in range(NUM_AGENTS)]

        self.patients_delivered = [False] * MAX_PATIENTS
        self.patients_actually_delivered = [False] * MAX_PATIENTS
        self.patients_died = [False] * MAX_PATIENTS
        self.patient_timers = [self.episode_patient_timer] * MAX_PATIENTS
        self.patient_initial_timers = (
            [self.episode_patient_timer] * MAX_PATIENTS
        )
        self.patient_weights = [self.random_weight() for _ in range(MAX_PATIENTS)]
        self.initial_patient_weights = self.patient_weights.copy()
        self.updated_patient_weights = [float(value) for value in self.patient_weights]
        self.patient_decay_rates = [0.0] * MAX_PATIENTS
        self.patient_survival_offsets = [0.0] * MAX_PATIENTS
        self.patient_serious_thresholds = [0.0] * MAX_PATIENTS
        self.patient_critical_thresholds = [0.0] * MAX_PATIENTS
        self.patient_survival_probabilities = [1.0] * MAX_PATIENTS
        for patient_idx in range(MAX_PATIENTS):
            self.sample_patient_survival_profile(patient_idx)
        self.patient_active = [i < NUM_INITIAL_PATIENTS for i in range(MAX_PATIENTS)]
        self.patient_spawn_steps = [
            0 if i < NUM_INITIAL_PATIENTS else -1 for i in range(MAX_PATIENTS)
        ]
        self.patient_resolution_steps = [-1] * MAX_PATIENTS
        self.patient_delivery_agents = [-1] * MAX_PATIENTS
        self.episode_step = 0
        self.new_patient_timer = (
            self.sample_spawn_interval()
            if self.dynamic_spawning and not self.all_patients_spawned()
            else 0
        )
        self.first_delivery_step = -1
        self.last_delivery_step = -1
        self.all_patients_resolved_step = -1
        self.irrecoverable_step = -1
        self.termination_reason = "in_progress"

        
        self.wind_zones = set()
        self.wind_rectangles = []
        self.wind_raster = self.cells_to_raster(self.wind_zones)
        self.wind_timer = WIND_APPEAR_INTERVAL
        self.low_signal_zones = set()
        self.low_signal_rectangles = []
        self.low_signal_raster = self.cells_to_raster(self.low_signal_zones)
        self.low_signal_timer = LOW_SIGNAL_APPEAR_INTERVAL
        self.hazard_route_records = []
        self.hazard_route_challenges = []
        self.hazard_reserved_cells = set()
 
        self.astar_paths = self.compute_astar_paths()
        self.hazard_candidates = list({
            position for path in self.astar_paths for position in path
        })
        self.refresh_landing_distance_maps()

    def random_weight(self):
        x = random.randint(1, 3)
        if x == 1: return LEVEL1_PATIENT_WEIGHT
        if x == 2: return LEVEL2_PATIENT_WEIGHT
        return LEVEL3_PATIENT_WEIGHT

    def cells_to_raster(self, cells):

        raster = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.uint8
        )
        if cells:
            coordinates = np.asarray(list(cells), dtype=np.int64)
            raster[coordinates[:, 1], coordinates[:, 0]] = 1
        return raster

    def sample_spawn_interval(self):
        minimum_interval = max(
            1, self.patient_spawn_interval - self.patient_spawn_jitter
        )
        maximum_interval = max(
            minimum_interval,
            self.patient_spawn_interval + self.patient_spawn_jitter,
        )
        sampled_interval = random.randint(
            minimum_interval, maximum_interval
        )
        
        
        steps_until_final_spawn = max(
            1,
            self.final_patient_spawn_step
            - int(getattr(self, 'episode_step', 0)),
        )
        return min(sampled_interval, steps_until_final_spawn)

    def sample_patient_survival_profile(self, patient_idx):

        weight = int(self.initial_patient_weights[patient_idx])
        self.patient_decay_rates[patient_idx] = random.uniform(
            *SURVIVAL_DECAY_RANGES[weight]
        )
        self.patient_survival_offsets[patient_idx] = random.uniform(
            *SURVIVAL_OFFSET_RANGES[weight]
        )
        serious = random.uniform(*SERIOUS_SURVIVAL_THRESHOLD_RANGE)
        critical_upper = min(
            CRITICAL_SURVIVAL_THRESHOLD_RANGE[1], serious - 0.05
        )
        critical = random.uniform(
            CRITICAL_SURVIVAL_THRESHOLD_RANGE[0], critical_upper
        )
        self.patient_serious_thresholds[patient_idx] = serious
        self.patient_critical_thresholds[patient_idx] = critical
        self.patient_survival_probabilities[patient_idx] = 1.0 / (
            1.0 + math.exp(-self.patient_survival_offsets[patient_idx])
        )

    def update_patient_deterioration(self, patient_idx):

        elapsed = max(
            0, self.episode_step - self.patient_spawn_steps[patient_idx]
        )
        exponent = (
            self.patient_decay_rates[patient_idx] * elapsed
            - self.patient_survival_offsets[patient_idx]
        )
        survival = 1.0 / (1.0 + math.exp(min(60.0, exponent)))
        self.patient_survival_probabilities[patient_idx] = survival
        if survival <= self.patient_critical_thresholds[patient_idx]:
            deterioration_class = LEVEL3_PATIENT_WEIGHT
        elif survival <= self.patient_serious_thresholds[patient_idx]:
            deterioration_class = LEVEL2_PATIENT_WEIGHT
        else:
            deterioration_class = LEVEL1_PATIENT_WEIGHT
        
        current_weight = max(
            int(self.initial_patient_weights[patient_idx]),
            deterioration_class,
        )
        self.patient_weights[patient_idx] = current_weight
        self.updated_patient_weights[patient_idx] = float(current_weight)

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def direction_vector(self, pos, target):
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 0:
            return dx / dist, dy / dist
        return 0.0, 0.0

    def shortest_path_distance_map(self, goal):

        unreachable = self.grid_size * self.grid_size
        distances = np.full(
            (self.grid_size, self.grid_size), unreachable, dtype=np.int16
        )
        if goal in self.obstacles:
            raise RuntimeError(f'Landing zone {goal} overlaps an obstacle')
        distances[goal[0], goal[1]] = 0
        search_queue = deque([goal])
        while search_queue:
            x, y = search_queue.popleft()
            next_distance = int(distances[x, y]) + 1
            for dx, dy in MOVEMENT_OFFSETS:
                neighbor = (x + dx, y + dy)
                if not (
                    0 <= neighbor[0] < self.grid_size
                    and 0 <= neighbor[1] < self.grid_size
                ):
                    continue
                if neighbor in self.obstacles:
                    continue
                if distances[neighbor[0], neighbor[1]] <= next_distance:
                    continue
                distances[neighbor[0], neighbor[1]] = next_distance
                search_queue.append(neighbor)
        return distances

    def refresh_landing_distance_maps(self):

        self.landing_distance_maps = [
            self.shortest_path_distance_map(zone)
            for zone in self.landing_zones
        ]
        unreachable = self.grid_size * self.grid_size
        self.landing_distance_scales = []
        for distance_map in self.landing_distance_maps:
            reachable = distance_map[distance_map < unreachable]
            self.landing_distance_scales.append(
                max(1, int(reachable.max()))
            )

    def movement_success_probability(self, position):

        success_probability = 1.0
        if position in self.low_signal_zones:
            success_probability *= 1.0 - LOW_SIGNAL_FAILURE_PROB
        if position in self.wind_zones:
            success_probability *= 1.0 - WIND_MOVEMENT_FAILURE_PROB
        return max(1e-6, success_probability)

    def expected_transition_battery_cost(self, origin, destination):

        success_probability = self.movement_success_probability(origin)
        expected_failed_attempts = (
            (1.0 - success_probability) / success_probability
        )
        failed_attempt_cost = (
            BATTERY_DRAIN_PER_STEP
            + (BATTERY_DRAIN_IN_WIND if origin in self.wind_zones else 0.0)
        )
        successful_attempt_cost = (
            BATTERY_DRAIN_PER_STEP
            + (
                BATTERY_DRAIN_IN_WIND
                if destination in self.wind_zones else 0.0
            )
        )
        return (
            successful_attempt_cost
            + expected_failed_attempts * failed_attempt_cost
        )

    def return_energy_cost_map(self, goal):

        distances = np.full(
            (self.grid_size, self.grid_size), np.inf, dtype=np.float64
        )
        distances[goal[0], goal[1]] = 0.0
        open_set = [(0.0, goal)]
        while open_set:
            current_cost, current = heapq.heappop(open_set)
            x, y = current
            if current_cost > float(distances[x, y]) + 1e-10:
                continue
            for dx, dy in MOVEMENT_OFFSETS:
                predecessor = (x + dx, y + dy)
                if not (
                    0 <= predecessor[0] < self.grid_size
                    and 0 <= predecessor[1] < self.grid_size
                ):
                    continue
                if predecessor in self.obstacles:
                    continue
                transition_cost = self.expected_transition_battery_cost(
                    predecessor, current
                )
                candidate_cost = current_cost + transition_cost
                if candidate_cost + 1e-8 >= float(
                        distances[predecessor[0], predecessor[1]]):
                    continue
                distances[predecessor[0], predecessor[1]] = candidate_cost
                heapq.heappush(open_set, (candidate_cost, predecessor))
        return distances

    def refresh_return_energy_maps(self):

        self.return_energy_maps = [
            self.return_energy_cost_map(zone) for zone in self.landing_zones
        ]

    def landing_shortest_distance(self, agent_idx, position):
        return int(
            self.landing_distance_maps[agent_idx][position[0], position[1]]
        )

    def required_return_battery(self, agent_idx, position):

        if hasattr(self, 'return_energy_maps'):
            route_energy = float(
                self.return_energy_maps[agent_idx][position[0], position[1]]
            )
        else:
            route_energy = (
                self.landing_shortest_distance(agent_idx, position)
                * BATTERY_DRAIN_PER_STEP
            )
        if not math.isfinite(route_energy):
            return float('inf')
        return route_energy + SAFE_RETURN_BATTERY_BUFFER

    def safe_return_battery_margin(self, agent_idx, position=None):

        if position is None:
            position = self.agents[agent_idx]
        return (
            self.batteries[agent_idx]
            - self.required_return_battery(agent_idx, position)
        )

    def energy_return_required(self, agent_idx, position=None):

        if self.landed[agent_idx] or self.battery_depleted[agent_idx]:
            return False
        return bool(
            self.batteries[agent_idx] <= LOW_BATTERY_THRESHOLD
            or self.safe_return_battery_margin(agent_idx, position)
            <= 0.0
        )

    def landing_permitted(self, agent_idx):

        return bool(
            not self.landed[agent_idx]
            and not self.drone_died[agent_idx]
            and tuple(self.agents[agent_idx])
            == tuple(self.landing_zones[agent_idx])
            and (
                self.all_patients_resolved()
                or self.energy_return_required(agent_idx)
            )
        )

    def largest_traversable_component(self):

        if hasattr(self, '_largest_traversable_component'):
            return self._largest_traversable_component
        unvisited = {
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.obstacles
        }
        largest_component = set()
        while unvisited:
            start = unvisited.pop()
            component = {start}
            search_queue = deque([start])
            while search_queue:
                x, y = search_queue.popleft()
                for dx, dy in MOVEMENT_OFFSETS:
                    neighbor = (x + dx, y + dy)
                    if neighbor not in unvisited:
                        continue
                    unvisited.remove(neighbor)
                    component.add(neighbor)
                    search_queue.append(neighbor)
            if len(component) > len(largest_component):
                largest_component = component
        self._largest_traversable_component = frozenset(largest_component)
        return self._largest_traversable_component

    def configure_curriculum_stage(self, stage_index):

        stage_index = int(stage_index)
        if not 0 <= stage_index < len(CURRICULUM_STAGES):
            raise ValueError(f'Invalid curriculum stage {stage_index}')
        stage = CURRICULUM_STAGES[stage_index]
        self.curriculum_stage = stage_index
        self.curriculum_stage_name = str(stage['name'])
        self.episode_initial_patients = int(stage['initial_patients'])
        self.episode_max_patients = int(stage['max_patients'])
        self.dynamic_spawning = bool(stage['dynamic_spawning'])
        self.hazard_fraction = float(stage['hazard_fraction'])
        self.hazard_penalty_scale = float(stage['hazard_penalty_scale'])
        self.curriculum_max_distance = int(stage['maximum_path_distance'])
        self.curriculum_max_landing_distance = int(
            stage['maximum_landing_path_distance']
        )
        self.episode_patient_timer = int(stage['patient_timer'])
        self.patient_spawn_interval = int(stage['spawn_interval'])
        self.patient_spawn_jitter = int(stage['spawn_jitter'])
        self.minimum_patient_spawn_batch = int(stage['spawn_batch_min'])
        self.maximum_patient_spawn_batch = int(stage['spawn_batch_max'])
        self.final_patient_spawn_step = int(stage['final_spawn_step'])
        self.episode_mode = (
            'full_mission'
            if stage_index == len(CURRICULUM_STAGES) - 1
            else f'curriculum_stage_{stage_index}'
        )

    def current_episode_deadline(self):
        return (
            self.landing_deadline
            if self.landing_deadline >= 0 else self.episode_max_steps
        )

    def nearest_undelivered_patient(self, pos):
        best_idx = None
        best_dist = float('inf')
        for p in range(MAX_PATIENTS):
            if not self.patient_active[p]:
                continue
            if not self.patients_delivered[p]:
                d = self.manhattan_distance(pos, self.patient_positions[p])
                if d < best_dist:
                    best_dist = d
                    best_idx = p
        return best_idx

    def generate_obstacles(self):
        protected = {
            (2, 2), (2, 26), (4, 38), (6, 52), (8, 64),
            (26, 26), (26, 2), (50, 50), (50, 2),
            (70, 20), (20, 70), (80, 60), (60, 80),
            (96, 96), (96, 90), (92, 92), (92, 96), (94, 94),
        }
        obstacles = set()

        
        
        
        for y in range(6, 24):
            if y not in [10, 11, 12, 18, 19, 20]:
                obstacles.add((14, y))

        while len(obstacles) < NUM_OBSTACLES:
            x = random.randint(2, self.grid_size - 3)
            y = random.randint(2, self.grid_size - 3)
            if (x, y) not in protected:
                neighbors = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
                blocked = sum(
                    1 for n in neighbors
                    if n in obstacles or n[0] < 0 or n[0] >= self.grid_size
                       or n[1] < 0 or n[1] >= self.grid_size
                )
                if blocked < 3:
                    obstacles.add((x, y))

        return list(obstacles)

    def a_star(self, start, goal, additional_blocked=None):

        additional_blocked = set(additional_blocked or ())
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score   = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            x, y = current
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = (x+dx, y+dy)
                if not (0 <= nb[0] < self.grid_size and 0 <= nb[1] < self.grid_size):
                    continue
                if nb in self.obstacles or nb in additional_blocked:
                    continue
                tg = g_score[current] + 1
                if nb not in g_score or tg < g_score[nb]:
                    came_from[nb] = current
                    g_score[nb]   = tg
                    heapq.heappush(open_set, (tg + self.manhattan_distance(nb, goal), nb))
        return []

    def compute_astar_paths(self):

        best_patient_routes = {}
        for agent_idx, ag_pos in enumerate(self.agents):
            parents = {ag_pos: None}
            search_queue = deque([ag_pos])
            while search_queue:
                current = search_queue.popleft()
                x, y = current
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    neighbor = (x + dx, y + dy)
                    if not (
                        0 <= neighbor[0] < self.grid_size
                        and 0 <= neighbor[1] < self.grid_size
                    ):
                        continue
                    if neighbor in self.obstacles or neighbor in parents:
                        continue
                    parents[neighbor] = current
                    search_queue.append(neighbor)

            for patient_idx in range(self.episode_max_patients):
                pat_pos = self.patient_positions[patient_idx]
                if pat_pos not in parents:
                    continue
                path = []
                current = pat_pos
                while current is not None:
                    path.append(current)
                    current = parents[current]
                path.reverse()
                candidate = {
                    'agent': int(agent_idx),
                    'patient': int(patient_idx),
                    'initial_weight': int(
                        self.initial_patient_weights[patient_idx]
                    ),
                    'path': path,
                }
                previous = best_patient_routes.get(patient_idx)
                if (previous is None
                        or len(path) < len(previous['path'])
                        or (
                            len(path) == len(previous['path'])
                            and agent_idx < previous['agent']
                        )):
                    best_patient_routes[patient_idx] = candidate
        self.hazard_route_records = [
            best_patient_routes[patient_idx]
            for patient_idx in sorted(best_patient_routes)
        ]
        return [record['path'] for record in self.hazard_route_records]

    def rectangle_cells(self, center, width, height):

        center_x, center_y = center
        left = max(0, min(center_x - width // 2, self.grid_size - width))
        top = max(0, min(center_y - height // 2, self.grid_size - height))
        rectangle = (left, top, width, height)
        cells = {
            (x, y)
            for x in range(left, left + width)
            for y in range(top, top + height)
        }
        return rectangle, cells

    def generate_hazard_rectangles(self, rectangle_count, hazard_kind):

        rectangle_count = int(rectangle_count)
        if rectangle_count <= 0:
            return set(), []

        route_groups = {weight: [] for weight in (1, 2, 3)}
        for record in getattr(self, 'hazard_route_records', []):
            if len(record['path']) >= 3:
                route_groups[record['initial_weight']].append(record)
        for routes in route_groups.values():
            random.shuffle(routes)
        weight_order = [
            weight for weight in (1, 2, 3) if route_groups[weight]
        ]
        random.shuffle(weight_order)
        all_routes = [
            route for weight in weight_order for route in route_groups[weight]
        ]

        protected = set(self.start_positions)
        protected.update(self.patient_positions[:self.episode_max_patients])
        protected.update(self.landing_zones)
        reserved_cells = set(getattr(self, 'hazard_reserved_cells', set()))
        cells = set()
        rectangles = []
        challenge_records = []
        for rectangle_index in range(rectangle_count):
            accepted = None
            for _attempt in range(HAZARD_PLACEMENT_ATTEMPTS):
                if not all_routes:
                    break
                desired_weight = weight_order[
                    rectangle_index % len(weight_order)
                ]
                
                
                
                candidate_routes = (
                    route_groups[desired_weight]
                    if _attempt < int(0.75 * HAZARD_PLACEMENT_ATTEMPTS)
                    else all_routes
                )
                route = candidate_routes[
                    (rectangle_index + _attempt) % len(candidate_routes)
                ]
                path = route['path']
                route_fraction = random.uniform(
                    HAZARD_ROUTE_MIN_FRACTION,
                    HAZARD_ROUTE_MAX_FRACTION,
                )
                path_index = int(round(route_fraction * (len(path) - 1)))
                path_index = max(1, min(len(path) - 2, path_index))
                
                
                
                
                
                route_steps = max(1, len(path) - 1)
                if self.curriculum_stage == 0:
                    maximum_width = min(
                        HAZARD_RECTANGLE_MAX_WIDTH,
                        max(2, route_steps // 3),
                    )
                    maximum_height = min(
                        HAZARD_RECTANGLE_MAX_HEIGHT,
                        max(2, route_steps // 3),
                    )
                    minimum_width = 2
                    minimum_height = 2
                else:
                    minimum_width = HAZARD_RECTANGLE_MIN_WIDTH
                    maximum_width = HAZARD_RECTANGLE_MAX_WIDTH
                    minimum_height = HAZARD_RECTANGLE_MIN_HEIGHT
                    maximum_height = HAZARD_RECTANGLE_MAX_HEIGHT
                width = random.randint(minimum_width, maximum_width)
                height = random.randint(minimum_height, maximum_height)
                rectangle, rectangle_cell_set = self.rectangle_cells(
                    path[path_index], width, height
                )
                if rectangle_cell_set & protected:
                    continue
                overlap_fraction = (
                    len(rectangle_cell_set & reserved_cells)
                    / max(1, len(rectangle_cell_set))
                )
                if overlap_fraction > HAZARD_MAX_RECTANGLE_OVERLAP:
                    continue
                path_intersection = [
                    position for position in path
                    if position in rectangle_cell_set
                ]
                if not path_intersection:
                    continue
                safe_path = self.a_star(
                    path[0],
                    path[-1],
                    additional_blocked=(
                        reserved_cells | cells | rectangle_cell_set
                    ),
                )
                if not safe_path:
                    continue
                baseline_steps = max(1, len(path) - 1)
                safe_steps = max(1, len(safe_path) - 1)
                detour_ratio = safe_steps / baseline_steps
                if detour_ratio > HAZARD_MAX_SAFE_DETOUR_RATIO:
                    continue
                accepted = (
                    route,
                    rectangle,
                    rectangle_cell_set,
                    path_intersection,
                    safe_steps,
                    detour_ratio,
                    route_fraction,
                )
                break
            if accepted is None:
                continue
            (
                route,
                rectangle,
                rectangle_cell_set,
                path_intersection,
                safe_steps,
                detour_ratio,
                route_fraction,
            ) = accepted
            rectangles.append(rectangle)
            cells.update(rectangle_cell_set)
            reserved_cells.update(rectangle_cell_set)
            challenge_records.append({
                'hazard_kind': str(hazard_kind),
                'agent': int(route['agent']),
                'patient': int(route['patient']),
                'initial_weight': int(route['initial_weight']),
                'baseline_path_steps': int(len(route['path']) - 1),
                'safe_detour_steps': int(safe_steps),
                'safe_detour_ratio': float(detour_ratio),
                'route_fraction': float(route_fraction),
                'path_intersection_cells': int(len(path_intersection)),
                'rectangle': [int(value) for value in rectangle],
                'safe_detour_exists': True,
            })
        self.hazard_reserved_cells = reserved_cells
        self.hazard_route_challenges.extend(challenge_records)
        return cells, rectangles

    def update_wind_zones(self, force=False):
        if self.wind_timer > 0:
            if not force:
                self.wind_timer -= 1
                if self.wind_timer > 0:
                    return
        rectangle_count = int(round(
            NUM_WIND_ZONE_RECTANGLES * self.hazard_fraction
        ))
        self.hazard_reserved_cells = set()
        self.hazard_route_challenges = []
        self.wind_zones, self.wind_rectangles = (
            self.generate_hazard_rectangles(rectangle_count, 'wind')
        )
        self.wind_raster = self.cells_to_raster(self.wind_zones)
        self.wind_timer = WIND_APPEAR_INTERVAL

    def update_low_signal_zones(self, force=False):
        if self.low_signal_timer > 0:
            if not force:
                self.low_signal_timer -= 1
                if self.low_signal_timer > 0:
                    return
        rectangle_count = int(round(
            NUM_LOW_SIGNAL_ZONE_RECTANGLES * self.hazard_fraction
        ))
        self.low_signal_zones, self.low_signal_rectangles = (
            self.generate_hazard_rectangles(rectangle_count, 'low_signal')
        )
        self.low_signal_raster = self.cells_to_raster(
            self.low_signal_zones
        )
        self.low_signal_timer = LOW_SIGNAL_APPEAR_INTERVAL

    def get_state(self):
        if any(
                bool(self.drone_died[agent_idx])
                != bool(self.battery_depleted[agent_idx])
                for agent_idx in range(NUM_AGENTS)):
            raise RuntimeError(
                'Drone died flags diverged from battery-depletion status'
            )
        drones = np.zeros((NUM_AGENTS, DRONE_STATE_DIM), dtype=np.float32)
        local_grids = np.zeros(
            (NUM_AGENTS, 3, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE),
            dtype=np.uint8,
        )
        action_masks = np.zeros((NUM_AGENTS, ACTION_DIM), dtype=np.uint8)

        
        
        
        
        radius = LOCAL_GRID_RADIUS
        padded_local_layers = np.stack([
            np.pad(
                self.obstacle_raster,
                radius,
                mode='constant',
                constant_values=1,
            ),
            np.pad(self.wind_raster, radius, mode='constant'),
            np.pad(self.low_signal_raster, radius, mode='constant'),
        ])

        for agent_idx in range(NUM_AGENTS):
            x, y = self.agents[agent_idx]
            lz_x, lz_y = self.landing_zones[agent_idx]
            previous_action_one_hot = [0.0] * ACTION_DIM
            previous_action_one_hot[self.previous_actions[agent_idx]] = 1.0
            drones[agent_idx] = [
                x / self.grid_size,
                y / self.grid_size,
                self.batteries[agent_idx] / MAX_BATTERY,
                1.0 if self.landed[agent_idx] else 0.0,
                
                
                1.0 if self.drone_died[agent_idx] else 0.0,
                lz_x / self.grid_size,
                lz_y / self.grid_size,
                float(self.previous_displacements[agent_idx][0]),
                float(self.previous_displacements[agent_idx][1]),
                1.0 if self.previous_collision_flags[agent_idx] else 0.0,
                min(1.0, self.collision_streaks[agent_idx] / COLLISION_STREAK_CAP),
                *previous_action_one_hot,
                self.landing_shortest_distance(agent_idx, (x, y))
                / self.landing_distance_scales[agent_idx],
                float(np.clip(
                    self.safe_return_battery_margin(agent_idx, (x, y))
                    / MAX_BATTERY,
                    -1.0,
                    1.0,
                )),
                1.0 if (x, y) in self.wind_zones else 0.0,
                1.0 if (x, y) in self.low_signal_zones else 0.0,
                1.0 if (
                    self.all_patients_resolved()
                    or self.energy_return_required(agent_idx, (x, y))
                ) else 0.0,
            ]
            if self.landed[agent_idx] or self.drone_died[agent_idx]:
                action_masks[agent_idx, ACTION_HOVER] = 1
            elif self.landing_permitted(agent_idx):
                
                
                action_masks[agent_idx, ACTION_LAND] = 1
            else:
                action_masks[agent_idx, :ACTION_HOVER + 1] = 1
                action_masks[agent_idx, ACTION_HOVER] = 1

            local_grids[agent_idx] = padded_local_layers[
                :,
                y:y + LOCAL_GRID_SIZE,
                x:x + LOCAL_GRID_SIZE,
            ]

        patients = np.zeros((MAX_PATIENTS, PATIENT_STATE_DIM), dtype=np.float32)
        patient_masks = np.zeros(MAX_PATIENTS, dtype=np.uint8)
        pending_patient_masks = np.zeros(MAX_PATIENTS, dtype=np.uint8)
        for patient_idx in range(MAX_PATIENTS):
            if not self.patient_active[patient_idx]:
                continue

            patient_masks[patient_idx] = 1
            px, py = self.patient_positions[patient_idx]
            pending = not self.patients_delivered[patient_idx]
            pending_patient_masks[patient_idx] = int(pending)
            elapsed_response_fraction = float(np.clip(
                (
                    self.episode_step
                    - self.patient_spawn_steps[patient_idx]
                ) / max(1, self.patient_initial_timers[patient_idx]),
                0.0,
                1.0,
            ))
            patients[patient_idx] = [
                px / self.grid_size,
                py / self.grid_size,
                max(0.0, self.patient_timers[patient_idx]) / MAX_PATIENT_TIMER,
                self.patient_weights[patient_idx] / MAX_PATIENT_WEIGHT,
                self.initial_patient_weights[patient_idx] / MAX_PATIENT_WEIGHT,
                1.0,
                1.0 if pending else 0.0,
                1.0 if self.patients_actually_delivered[patient_idx] else 0.0,
                1.0 if self.patients_died[patient_idx] else 0.0,
                elapsed_response_fraction,
            ]

        spawned_count = sum(self.patient_active)
        resolved_count = sum(self.patients_delivered)
        delivered_count = sum(self.patients_actually_delivered)
        died_count = sum(self.patients_died)
        pending_count = spawned_count - resolved_count
        service_state = self.triage_service_state()
        mission = np.asarray([
            min(1.0, self.episode_step / max(1, self.current_episode_deadline())),
            max(0.0, self.new_patient_timer) / max(
                1,
                self.patient_spawn_interval + self.patient_spawn_jitter,
            ),
            spawned_count / MAX_PATIENTS,
            pending_count / MAX_PATIENTS,
            delivered_count / MAX_PATIENTS,
            died_count / MAX_PATIENTS,
            sum(self.landed) / NUM_AGENTS,
            1.0 if self.all_patients_spawned() else 0.0,
            1.0 if self.all_patients_resolved() else 0.0,
            service_state[1]['debt_fraction'],
            service_state[2]['debt_fraction'],
            service_state[3]['debt_fraction'],
        ], dtype=np.float32)

        return {
            'drones': drones,
            'patients': patients,
            'patient_masks': patient_masks,
            'pending_patient_masks': pending_patient_masks,
            'local_grids': local_grids,
            'mission': mission,
            'action_masks': action_masks,
        }

    def all_patients_spawned(self):
        return all(
            self.patient_active[patient_idx]
            for patient_idx in range(self.episode_max_patients)
        )

    def all_patients_resolved(self):
        return self.all_patients_spawned() and all(
            self.patients_delivered[patient_idx]
            for patient_idx in range(self.episode_max_patients)
        )

    def all_patients_delivered_successfully(self):
        return self.all_patients_spawned() and all(
            self.patients_actually_delivered[patient_idx]
            for patient_idx in range(self.episode_max_patients)
        )

    def perfect_rescue(self):
        return self.all_patients_delivered_successfully()

    def triage_service_state(self):

        result = {}
        for weight in (1, 2, 3):
            spawned = sum(
                self.patient_active[patient_idx]
                and self.initial_patient_weights[patient_idx] == weight
                for patient_idx in range(self.episode_max_patients)
            )
            delivered = sum(
                self.patient_active[patient_idx]
                and self.initial_patient_weights[patient_idx] == weight
                and self.patients_actually_delivered[patient_idx]
                for patient_idx in range(self.episode_max_patients)
            )
            target_count = TRIAGE_CLASS_DELIVERY_TARGETS[weight] * spawned
            debt_count = max(0.0, target_count - delivered)
            result[weight] = {
                'spawned': int(spawned),
                'delivered': int(delivered),
                'target_count': float(target_count),
                'debt_count': float(debt_count),
                'debt_fraction': float(
                    debt_count / max(1.0, target_count)
                ),
                'target_attainment': float(
                    min(1.0, delivered / max(1.0, target_count))
                ),
            }
        return result

    def priority_service_potential(self):

        service_state = self.triage_service_state()
        return float(sum(
            (weight / MAX_PATIENT_WEIGHT)
            * min(
                service_state[weight]['delivered'],
                service_state[weight]['target_count'],
            )
            / MAX_PATIENTS
            for weight in (1, 2, 3)
        ))

    def rescue_success(self):
        if not self.all_patients_resolved():
            return False
        outcome = self.mission_outcome_metrics()
        return bool(
            outcome['delivery_utilization']
            >= APPLICATION_DELIVERY_SUCCESS_THRESHOLD
            and outcome['triage_efficiency']
            >= APPLICATION_TRIAGE_SUCCESS_THRESHOLD
            and outcome['lower_triage_delivery_floor']
            >= APPLICATION_LOWER_TRIAGE_FLOOR
            and outcome['acuity_priority_score']
            >= APPLICATION_ACUITY_PRIORITY_THRESHOLD
            and outcome['priority_fairness_attainment']
            >= APPLICATION_PRIORITY_FAIRNESS_THRESHOLD
            and outcome['triage_delivery_ordering_score']
            >= APPLICATION_TRIAGE_ORDERING_THRESHOLD
            and outcome['triage_response_time_ordering_score']
            >= APPLICATION_TRIAGE_RESPONSE_THRESHOLD
        )

    def safe_return_complete(self):
        return self.all_patients_resolved() and all(self.landed)

    def mission_success(self):
        return self.rescue_success() and self.safe_return_complete()

    def mission_outcome_metrics(self):

        spawned = [
            patient_idx for patient_idx in range(self.episode_max_patients)
            if self.patient_active[patient_idx]
        ]
        delivered_count = sum(
            self.patients_actually_delivered[patient_idx]
            for patient_idx in spawned
        )
        died_count = sum(self.patients_died[patient_idx] for patient_idx in spawned)
        weighted_spawned = float(sum(
            self.initial_patient_weights[patient_idx]
            for patient_idx in spawned
        ))
        weighted_delivered = float(sum(
            self.initial_patient_weights[patient_idx]
            for patient_idx in spawned
            if self.patients_actually_delivered[patient_idx]
        ))
        delivery_utilization = (
            delivered_count / len(spawned) if spawned else 0.0
        )
        triage_efficiency = (
            weighted_delivered / weighted_spawned
            if weighted_spawned > 0.0 else 0.0
        )
        rescue_quality = (
            (1.0 - DELIVERY_TRIAGE_ALPHA) * delivery_utilization
            + DELIVERY_TRIAGE_ALPHA * triage_efficiency
        )
        spawned_by_weight = {weight: 0 for weight in (1, 2, 3)}
        delivered_by_weight = {weight: 0 for weight in (1, 2, 3)}
        died_by_weight = {weight: 0 for weight in (1, 2, 3)}
        response_times = {weight: [] for weight in (1, 2, 3)}
        response_time_ratios = {weight: [] for weight in (1, 2, 3)}
        patient_outcomes = []
        for patient_idx in spawned:
            weight = int(self.initial_patient_weights[patient_idx])
            spawned_by_weight[weight] += 1
            delivered = bool(self.patients_actually_delivered[patient_idx])
            died = bool(self.patients_died[patient_idx])
            if delivered:
                delivered_by_weight[weight] += 1
            elif died:
                died_by_weight[weight] += 1
            response_time = (
                max(
                    0,
                    self.patient_resolution_steps[patient_idx]
                    - self.patient_spawn_steps[patient_idx],
                )
                if self.patient_resolution_steps[patient_idx] >= 0 else None
            )
            response_time_ratio = (
                response_time
                / max(1, self.patient_initial_timers[patient_idx])
                if response_time is not None else None
            )
            if delivered and response_time is not None:
                response_times[weight].append(response_time)
                response_time_ratios[weight].append(response_time_ratio)
            patient_outcomes.append(
                (weight, delivered, response_time_ratio)
            )

        present_weights = [
            weight for weight in (1, 2, 3)
            if spawned_by_weight[weight] > 0
        ]
        class_delivery_rates = {
            weight: (
                delivered_by_weight[weight] / spawned_by_weight[weight]
                if spawned_by_weight[weight] > 0 else 1.0
            )
            for weight in (1, 2, 3)
        }
        target_fulfillment = {
            weight: (
                min(
                    1.0,
                    class_delivery_rates[weight]
                    / TRIAGE_CLASS_DELIVERY_TARGETS[weight],
                )
                if spawned_by_weight[weight] > 0 else 1.0
            )
            for weight in (1, 2, 3)
        }
        present_fulfillment = [
            target_fulfillment[weight] for weight in present_weights
        ]
        priority_jain_denominator = len(present_fulfillment) * sum(
            value * value for value in present_fulfillment
        )
        priority_normalized_jain = (
            sum(present_fulfillment) ** 2 / priority_jain_denominator
            if priority_jain_denominator > 0.0 else 0.0
        )
        priority_target_attainment = (
            float(np.mean(present_fulfillment))
            if present_fulfillment else 0.0
        )
        priority_fairness_attainment = (
            priority_normalized_jain * priority_target_attainment
        )
        mean_response_ratios = {
            weight: (
                float(np.mean(response_time_ratios[weight]))
                if response_time_ratios[weight] else 1.0
            )
            for weight in (1, 2, 3)
        }
        mean_response_times = {
            weight: (
                float(np.mean(response_times[weight]))
                if response_times[weight] else -1.0
            )
            for weight in (1, 2, 3)
        }
        median_response_times = {
            weight: (
                float(np.median(response_times[weight]))
                if response_times[weight] else -1.0
            )
            for weight in (1, 2, 3)
        }
        p90_response_times = {
            weight: (
                float(np.percentile(response_times[weight], 90))
                if response_times[weight] else -1.0
            )
            for weight in (1, 2, 3)
        }
        first_response_times = {
            weight: (
                float(min(response_times[weight]))
                if response_times[weight] else -1.0
            )
            for weight in (1, 2, 3)
        }
        all_delivered_response_times = [
            response_time
            for weight in (1, 2, 3)
            for response_time in response_times[weight]
        ]
        w3_before_w1_points = 0.0
        w3_before_w1_pairs = 0
        for high_response in response_times[3]:
            for low_response in response_times[1]:
                w3_before_w1_pairs += 1
                if high_response < low_response:
                    w3_before_w1_points += 1.0
                elif high_response == low_response:
                    w3_before_w1_points += 0.5
        w3_before_w1_fraction = (
            w3_before_w1_points / w3_before_w1_pairs
            if w3_before_w1_pairs else 0.5
        )

        
        
        
        
        
        
        
        ordering_points = 0.0
        rate_ordering_points = 0.0
        response_ordering_points = 0.0
        ordering_pairs = 0
        response_tiebreak_pairs = 0
        for high_weight in present_weights:
            for low_weight in present_weights:
                if high_weight <= low_weight:
                    continue
                ordering_pairs += 1
                difference = (
                    class_delivery_rates[high_weight]
                    - class_delivery_rates[low_weight]
                )
                if difference > 1e-8:
                    ordering_points += 1.0
                    rate_ordering_points += 1.0
                elif abs(difference) <= 1e-8:
                    rate_ordering_points += 0.5
                    high_has_response = bool(
                        response_time_ratios[high_weight]
                    )
                    low_has_response = bool(
                        response_time_ratios[low_weight]
                    )
                    if high_has_response and low_has_response:
                        response_tiebreak_pairs += 1
                        response_difference = (
                            mean_response_ratios[low_weight]
                            - mean_response_ratios[high_weight]
                        )
                        if response_difference > 1e-8:
                            ordering_points += 1.0
                        elif abs(response_difference) <= 1e-8:
                            ordering_points += 0.5
                    else:
                        
                        ordering_points += 0.5

                high_has_response = bool(response_time_ratios[high_weight])
                low_has_response = bool(response_time_ratios[low_weight])
                if high_has_response and low_has_response:
                    response_difference = (
                        mean_response_ratios[low_weight]
                        - mean_response_ratios[high_weight]
                    )
                    if response_difference > 1e-8:
                        response_ordering_points += 1.0
                    elif abs(response_difference) <= 1e-8:
                        response_ordering_points += 0.5
                else:
                    response_ordering_points += 0.5
        triage_delivery_ordering_score = (
            ordering_points / ordering_pairs if ordering_pairs else 0.5
        )
        triage_delivery_rate_ordering_score = (
            rate_ordering_points / ordering_pairs
            if ordering_pairs else 0.5
        )
        triage_response_time_ordering_score = (
            response_ordering_points / ordering_pairs
            if ordering_pairs else 0.5
        )
        lower_present_weights = [
            weight for weight in (1, 2)
            if spawned_by_weight[weight] > 0
        ]
        lower_triage_floor = min(
            (class_delivery_rates[weight] for weight in lower_present_weights),
            default=1.0,
        )
        present_rates = [class_delivery_rates[weight] for weight in present_weights]
        class_delivery_gap = (
            max(present_rates) - min(present_rates)
            if present_rates else 0.0
        )
        jain_denominator = len(present_rates) * sum(
            rate * rate for rate in present_rates
        )
        class_jain_fairness = (
            sum(present_rates) ** 2 / jain_denominator
            if jain_denominator > 0.0 else 1.0
        )

        priority_points = 0.0
        priority_pairs = 0
        for high_weight, high_delivered, high_time in patient_outcomes:
            for low_weight, low_delivered, low_time in patient_outcomes:
                if high_weight <= low_weight:
                    continue
                priority_pairs += 1
                if high_delivered and not low_delivered:
                    priority_points += 1.0
                elif high_delivered and low_delivered:
                    if high_time < low_time:
                        priority_points += 1.0
                    elif high_time == low_time:
                        priority_points += 0.5
                elif not high_delivered and not low_delivered:
                    priority_points += 0.5
        acuity_priority_score = (
            priority_points / priority_pairs if priority_pairs else 0.5
        )
        workload = [
            sum(agent_idx == owner for owner in self.patient_delivery_agents)
            for agent_idx in range(NUM_AGENTS)
        ]
        workload_denominator = NUM_AGENTS * sum(
            value * value for value in workload
        )
        workload_jain_fairness = (
            sum(workload) ** 2 / workload_denominator
            if workload_denominator > 0 else 1.0
        )
        return {
            'delivery_utilization': float(delivery_utilization),
            'triage_efficiency': float(triage_efficiency),
            'rescue_quality': float(rescue_quality),
            'death_fraction': float(died_count / len(spawned) if spawned else 0.0),
            'spawned_count': int(len(spawned)),
            'delivered_count': int(delivered_count),
            'died_count': int(died_count),
            'weighted_spawned': float(weighted_spawned),
            'weighted_delivered': float(weighted_delivered),
            'acuity_priority_score': float(acuity_priority_score),
            'minimum_class_delivery_rate': float(
                min(present_rates) if present_rates else 1.0
            ),
            'lower_triage_delivery_floor': float(lower_triage_floor),
            'class_delivery_rate_gap': float(class_delivery_gap),
            'class_delivery_jain_fairness': float(class_jain_fairness),
            'priority_normalized_jain_fairness': float(
                priority_normalized_jain
            ),
            'priority_target_attainment': float(priority_target_attainment),
            'priority_fairness_attainment': float(
                priority_fairness_attainment
            ),
            'triage_delivery_ordering_score': float(
                triage_delivery_ordering_score
            ),
            'triage_delivery_rate_ordering_score': float(
                triage_delivery_rate_ordering_score
            ),
            'triage_response_time_ordering_score': float(
                triage_response_time_ordering_score
            ),
            'triage_ordering_pairs': int(ordering_pairs),
            'triage_response_tiebreak_pairs': int(
                response_tiebreak_pairs
            ),
            'high_vs_low_response_advantage': float(
                mean_response_ratios[1] - mean_response_ratios[3]
            ),
            'w3_vs_w1_response_advantage_steps': float(
                mean_response_times[1] - mean_response_times[3]
                if mean_response_times[1] >= 0.0
                and mean_response_times[3] >= 0.0 else 0.0
            ),
            'w3_before_w1_response_fraction': float(
                w3_before_w1_fraction
            ),
            'w3_before_w1_response_pairs': int(w3_before_w1_pairs),
            'mean_delivered_response_time': float(
                np.mean(all_delivered_response_times)
                if all_delivered_response_times else -1.0
            ),
            'delivery_workload_jain_fairness': float(workload_jain_fairness),
            'perfect_rescue': bool(self.perfect_rescue()),
            **{
                f'spawned_w{weight}': int(spawned_by_weight[weight])
                for weight in (1, 2, 3)
            },
            **{
                f'delivered_w{weight}': int(delivered_by_weight[weight])
                for weight in (1, 2, 3)
            },
            **{
                f'died_w{weight}': int(died_by_weight[weight])
                for weight in (1, 2, 3)
            },
            **{
                f'delivery_rate_w{weight}': float(class_delivery_rates[weight])
                for weight in (1, 2, 3)
            },
            **{
                f'delivery_target_w{weight}': float(
                    TRIAGE_CLASS_DELIVERY_TARGETS[weight]
                )
                for weight in (1, 2, 3)
            },
            **{
                f'target_fulfillment_w{weight}': float(
                    target_fulfillment[weight]
                )
                for weight in (1, 2, 3)
            },
            **{
                f'mean_response_ratio_w{weight}': float(
                    mean_response_ratios[weight]
                )
                for weight in (1, 2, 3)
            },
            **{
                f'mean_response_time_w{weight}': float(
                    mean_response_times[weight]
                )
                for weight in (1, 2, 3)
            },
            **{
                f'median_response_time_w{weight}': float(
                    median_response_times[weight]
                )
                for weight in (1, 2, 3)
            },
            **{
                f'p90_response_time_w{weight}': float(
                    p90_response_times[weight]
                )
                for weight in (1, 2, 3)
            },
            **{
                f'first_response_time_w{weight}': float(
                    first_response_times[weight]
                )
                for weight in (1, 2, 3)
            },
            **{
                f'response_count_w{weight}': int(len(response_times[weight]))
                for weight in (1, 2, 3)
            },
        }

    def rescue_outcome_reward(self):
        quality = self.mission_outcome_metrics()['rescue_quality']
        return (
            RESCUE_OUTCOME_LINEAR_REWARD * quality
            + RESCUE_OUTCOME_QUADRATIC_REWARD * quality * quality
        )

    def fairness_outcome_reward(self):

        fairness = self.mission_outcome_metrics()[
            'priority_fairness_attainment'
        ]
        return FAIRNESS_OUTCOME_REWARD * fairness

    def triage_ordering_outcome_reward(self):

        ordering = self.mission_outcome_metrics()[
            'triage_delivery_ordering_score'
        ]
        return TRIAGE_ORDERING_OUTCOME_REWARD * ordering

    def triage_response_outcome_reward(self):

        response_ordering = self.mission_outcome_metrics()[
            'triage_response_time_ordering_score'
        ]
        return TRIAGE_RESPONSE_OUTCOME_REWARD * (
            2.0 * response_ordering - 1.0
        )

    def safe_return_reward(self):
        quality = self.mission_outcome_metrics()['rescue_quality']
        return SAFE_RETURN_BASE_REWARD + SAFE_RETURN_QUALITY_REWARD * quality

    def mission_phase(self):
        if self.all_patients_resolved():
            return "landing"
        return "rescue"

    def fleet_potential_components(self):

        operational_agents = [
            i for i in range(NUM_AGENTS)
            if not self.landed[i] and not self.battery_depleted[i]
        ]
        energy_return_agents = [
            agent_idx for agent_idx in operational_agents
            if self.energy_return_required(agent_idx)
        ]
        service_agents = [
            agent_idx for agent_idx in operational_agents
            if agent_idx not in energy_return_agents
        ]
        pending_patients = [
            p for p in range(MAX_PATIENTS)
            if self.patient_active[p] and not self.patients_delivered[p]
        ]
        max_grid_distance = max(1.0, 2.0 * (self.grid_size - 1))
        outcome = self.mission_outcome_metrics()
        coverage_cost = 0.0
        landing_progress = 0.0
        coverage_assignment = []
        navigation_targets = []
        agent_navigation_potential = [0.0] * NUM_AGENTS

        if service_agents and pending_patients:
            distances = []
            candidate_pairs = sorted(
                (
                    self.manhattan_distance(
                        self.agents[agent_idx],
                        self.patient_positions[patient_idx],
                    ),
                    self.patient_timers[patient_idx],
                    agent_idx,
                    patient_idx,
                )
                for agent_idx in service_agents
                for patient_idx in pending_patients
            )
            assigned_agents = set()
            assigned_patients = set()
            for _, _, agent_idx, patient_idx in candidate_pairs:
                if (agent_idx in assigned_agents
                        or patient_idx in assigned_patients):
                    continue
                assigned_agents.add(agent_idx)
                assigned_patients.add(patient_idx)
                distance_cells = self.manhattan_distance(
                    self.agents[agent_idx], self.patient_positions[patient_idx]
                )
                distances.append(distance_cells)
                agent_navigation_potential[agent_idx] = -float(distance_cells)
                coverage_assignment.append({
                    'agent': int(agent_idx),
                    'patient': int(patient_idx),
                    'distance_cells': int(distance_cells),
                    'normalized_cost': float(distance_cells / max_grid_distance),
                })
                navigation_targets.append({
                    'agent': int(agent_idx),
                    'kind': 'patient',
                    'index': int(patient_idx),
                    'position': list(self.patient_positions[patient_idx]),
                })
                if (len(assigned_agents) == len(service_agents)
                        or len(assigned_patients) == len(pending_patients)):
                    break
            coverage_cost = float(np.mean(distances)) / max_grid_distance

        landing_navigation_agents = (
            operational_agents
            if self.all_patients_resolved() else energy_return_agents
        )
        if landing_navigation_agents:
            landing_distances = []
            for agent_idx in landing_navigation_agents:
                distance_cells = self.landing_shortest_distance(
                    agent_idx, self.agents[agent_idx]
                )
                landing_distances.append(distance_cells)
                agent_navigation_potential[agent_idx] = -float(distance_cells)
                navigation_targets.append({
                    'agent': int(agent_idx),
                    'kind': (
                        'landing_zone' if self.all_patients_resolved()
                        else 'energy_return'
                    ),
                    'index': int(agent_idx),
                    'position': list(self.landing_zones[agent_idx]),
                })
            if self.all_patients_resolved():
                landed_fraction = sum(self.landed) / NUM_AGENTS
                approach_fraction = 1.0 - min(
                    1.0, float(np.mean(landing_distances)) / max_grid_distance
                )
                landing_progress = (
                    0.5 * landed_fraction + 0.5 * approach_fraction
                )

        total = (
            outcome['rescue_quality'] - outcome['death_fraction'] - coverage_cost
            + landing_progress
        )
        return {
            'delivered': float(outcome['delivery_utilization']),
            'triage': float(outcome['triage_efficiency']),
            'rescue_quality': float(outcome['rescue_quality']),
            'dead': float(outcome['death_fraction']),
            'coverage_cost': float(coverage_cost),
            'coverage_assignment': coverage_assignment,
            'navigation_targets': navigation_targets,
            'energy_return_agents': [
                int(agent_idx) for agent_idx in energy_return_agents
            ],
            'landing_progress': float(landing_progress),
            'agent_navigation': [
                float(value) for value in agent_navigation_potential
            ],
            'total': float(total),
        }

    def fleet_potential(self):
        return self.fleet_potential_components()['total']

    def step(self, actions):
        
        step_start_ns = time.perf_counter_ns()
        if len(actions) != NUM_AGENTS:
            raise ValueError(
                f'Expected {NUM_AGENTS} actions, received {len(actions)}'
            )
        actions = [int(action) for action in actions]
        if any(action < 0 or action >= ACTION_DIM for action in actions):
            raise ValueError(
                f'Every action must be an integer from 0 through {ACTION_DIM - 1}'
            )

        self.episode_step += 1
        old_phase = self.mission_phase()
        landing_ready_at_start = [
            self.landing_permitted(agent_idx)
            for agent_idx in range(NUM_AGENTS)
        ]
        for agent_idx, landing_ready in enumerate(landing_ready_at_start):
            if (landing_ready
                    and self.landing_zone_arrival_steps[agent_idx] < 0):
                self.landing_zone_arrival_steps[agent_idx] = max(
                    0, self.episode_step - 1
                )
        old_potential = self.fleet_potential_components()
        energy_return_agents_at_start = set(
            old_potential['energy_return_agents']
        )
        if old_phase == 'landing':
            energy_return_agents_at_start.update(
                agent_idx for agent_idx in range(NUM_AGENTS)
                if not self.landed[agent_idx]
                and not self.battery_depleted[agent_idx]
            )

        step_data = {
            'wind_entries': [0] * NUM_AGENTS,
            'wind_exposure_steps': [0] * NUM_AGENTS,
            'wind_exits': [0] * NUM_AGENTS,
            'wind_failures': [0] * NUM_AGENTS,
            'low_signal_entries': [0] * NUM_AGENTS,
            'low_signal_exposure_steps': [0] * NUM_AGENTS,
            'low_signal_exits': [0] * NUM_AGENTS,
            'low_signal_failures': [0] * NUM_AGENTS,
            'operational_agent_steps': 0,
            'movement_actions': 0,
            'wind_command_attempts': 0,
            'low_signal_command_attempts': 0,
            'wind_avoidance_opportunities': 0,
            'wind_hazard_selections': 0,
            'wind_dominated_hazard_selections': 0,
            'wind_shortcut_hazard_selections': 0,
            'low_signal_avoidance_opportunities': 0,
            'low_signal_hazard_selections': 0,
            'low_signal_dominated_hazard_selections': 0,
            'low_signal_shortcut_hazard_selections': 0,
            'wind_entry_progress_cells': 0.0,
            'low_signal_entry_progress_cells': 0.0,
            'wind_zone_refreshed': 0,
            'low_signal_zone_refreshed': 0,
            'wind_refresh_onset_agents': [0] * NUM_AGENTS,
            'low_signal_refresh_onset_agents': [0] * NUM_AGENTS,
            'battery_depletion_events': [],
            'death_penalty_applications': [0] * NUM_AGENTS,
            'death_reminder_penalty_applications': [0] * NUM_AGENTS,
            'dead_landing_events': [],
            'battery_drain_by_agent': [0.0] * NUM_AGENTS,
            'wind_battery_drain_by_agent': [0.0] * NUM_AGENTS,
            'landing_standby_steps': [0] * NUM_AGENTS,
            'energy_return_mode_flags': [
                int(agent_idx in energy_return_agents_at_start)
                for agent_idx in range(NUM_AGENTS)
            ],
            'energy_return_progress_flags': [0] * NUM_AGENTS,
            'energy_return_regress_flags': [0] * NUM_AGENTS,
            'energy_margin_delta_by_agent': [0.0] * NUM_AGENTS,
            'safe_return_margin_before': [
                float(self.safe_return_battery_margin(agent_idx))
                if not self.landed[agent_idx]
                and not self.battery_depleted[agent_idx] else 0.0
                for agent_idx in range(NUM_AGENTS)
            ],
            'safe_return_margin_after': [0.0] * NUM_AGENTS,
            'reserve_violation_flags': [0] * NUM_AGENTS,
            'obstacle_collisions': 0,
            'obstacle_collision_flags': [0] * NUM_AGENTS,
            'obstacle_collision_events': [],
            'obstacle_action_opportunities': [0] * NUM_AGENTS,
            'obstacle_action_selected': [0] * NUM_AGENTS,
            'dominated_obstacle_selections': [0] * NUM_AGENTS,
            'dominated_agent_conflict_selections': [0] * NUM_AGENTS,
            'agent_collisions': 0,
            'closeness_pairs': 0,
            'same_destination_collisions': 0,
            'head_on_collisions': 0,
            'collision_pairs': [],
            'agent_collision_flags': [0] * NUM_AGENTS,
            'patient_spawn_events': [],
            'patient_weight_escalation_events': [],
            'patient_delivery_events': [],
            'patient_death_events': [],
            'landing_events': [],
            'landing_zone_arrival_events': [],
            'landing_zone_departure_events': [],
            'landing_only_action_flags': [
                int(value) for value in landing_ready_at_start
            ],
            'forced_terminal_landing_actions': [0] * NUM_AGENTS,
            'hover_actions': sum(action == ACTION_HOVER for action in actions),
            'land_actions': sum(action == ACTION_LAND for action in actions),
            'landing_progress_actions_available': [0] * NUM_AGENTS,
            'landing_distance_reduced': [0] * NUM_AGENTS,
            'landing_distance_increased': [0] * NUM_AGENTS,
            'landing_distance_unchanged': [0] * NUM_AGENTS,
            'landing_hover_with_progress_available': [0] * NUM_AGENTS,
            'phase_before': old_phase,
            'death_positive_shaping_prevented': 0,
            'hazard_penalty_scale': float(self.hazard_penalty_scale),
            'response_wait_cost': 0.0,
        }
        reward_components = {
            'step': STEP_PENALTY,
            'clean': 0.0,
            'delivery': 0.0,
            'patient_death': 0.0,
            'obstacle_collision': 0.0,
            'agent_collision': 0.0,
            'battery': 0.0,
            'wind': 0.0,
            'low_signal': 0.0,
            'wind_routing': 0.0,
            'low_signal_routing': 0.0,
            'low_battery': 0.0,
            'energy_navigation': 0.0,
            'response_time': 0.0,
            'landing': 0.0,
            'hover': 0.0,
            'closeness': 0.0,
            'potential': 0.0,
            'rescue_outcome': 0.0,
            'fairness': 0.0,
            'triage_ordering': 0.0,
            'triage_response': 0.0,
            'safe_return': 0.0,
            'mission': 0.0,
        }
        
        
        
        local_rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        
        
        
        for agent_idx in range(NUM_AGENTS):
            if self.death_reminder_steps_remaining[agent_idx] <= 0:
                continue
            reward_components['battery'] += POST_DEPLETION_REMINDER_PENALTY
            local_rewards[agent_idx] += POST_DEPLETION_REMINDER_PENALTY
            step_data['death_penalty_applications'][agent_idx] = 1
            step_data[
                'death_reminder_penalty_applications'
            ][agent_idx] = 1
            self.death_reminder_steps_remaining[agent_idx] -= 1

        old_positions = [tuple(pos) for pos in self.agents]
        operational_at_start = [
            i for i in range(NUM_AGENTS)
            if not self.landed[i] and not self.battery_depleted[i]
        ]
        step_data['operational_agent_steps'] = len(operational_at_start)
        clean_step = [False] * NUM_AGENTS
        obstacle_agents = []
        new_positions = []

        occupied_at_start = {
            agent_idx: {
                tuple(self.agents[other_idx])
                for other_idx in range(NUM_AGENTS) if other_idx != agent_idx
            }
            for agent_idx in operational_at_start
        }
        wind_avoidance_agents = set()
        low_signal_avoidance_agents = set()
        wind_best_safe_progress = {}
        low_signal_best_safe_progress = {}
        best_collision_free_progress = {}
        chosen_progress = {}
        navigation_targets = {
            int(target['agent']): tuple(target['position'])
            for target in old_potential['navigation_targets']
        }
        for agent_idx in operational_at_start:
            x, y = old_positions[agent_idx]
            current_landing_distance = self.landing_shortest_distance(
                agent_idx, old_positions[agent_idx]
            )
            valid_destinations = []
            for dx, dy in MOVEMENT_OFFSETS:
                candidate = (x + dx, y + dy)
                if (candidate in self.obstacles
                        or not (0 <= candidate[0] < self.grid_size
                                and 0 <= candidate[1] < self.grid_size)):
                    step_data['obstacle_action_opportunities'][agent_idx] += 1
                if (0 <= candidate[0] < self.grid_size
                        and 0 <= candidate[1] < self.grid_size
                        and candidate not in self.obstacles
                        and candidate not in occupied_at_start[agent_idx]):
                    valid_destinations.append(candidate)
                if (old_phase == 'landing'
                        and 0 <= candidate[0] < self.grid_size
                        and 0 <= candidate[1] < self.grid_size
                        and candidate not in occupied_at_start[agent_idx]
                        and candidate not in self.obstacles
                        and self.landing_shortest_distance(
                            agent_idx, candidate
                        ) < current_landing_distance):
                    step_data[
                        'landing_progress_actions_available'
                    ][agent_idx] += 1
            target = navigation_targets.get(agent_idx)
            best_collision_free_progress[agent_idx] = max(
                [0] + [
                    (
                        self.manhattan_distance((x, y), target)
                        - self.manhattan_distance(cell, target)
                        if target is not None else 0
                    )
                    for cell in valid_destinations
                ]
            )
            if (any(cell in self.wind_zones for cell in valid_destinations)
                    and any(cell not in self.wind_zones
                            for cell in valid_destinations)):
                wind_avoidance_agents.add(agent_idx)
                target = navigation_targets.get(agent_idx)
                wind_best_safe_progress[agent_idx] = max(
                    (
                        self.manhattan_distance((x, y), target)
                        - self.manhattan_distance(cell, target)
                        if target is not None else 0
                    )
                    for cell in valid_destinations
                    if cell not in self.wind_zones
                )
            if (any(cell in self.low_signal_zones
                    for cell in valid_destinations)
                    and any(cell not in self.low_signal_zones
                            for cell in valid_destinations)):
                low_signal_avoidance_agents.add(agent_idx)
                target = navigation_targets.get(agent_idx)
                low_signal_best_safe_progress[agent_idx] = max(
                    (
                        self.manhattan_distance((x, y), target)
                        - self.manhattan_distance(cell, target)
                        if target is not None else 0
                    )
                    for cell in valid_destinations
                    if cell not in self.low_signal_zones
                )
        step_data['wind_avoidance_opportunities'] = len(
            wind_avoidance_agents
        )
        step_data['low_signal_avoidance_opportunities'] = len(
            low_signal_avoidance_agents
        )

        for agent_idx, action in enumerate(actions):
            x, y = self.agents[agent_idx]
            if self.landed[agent_idx]:
                new_positions.append((x, y))
                continue

            if self.drone_died[agent_idx]:
                
                
                
                if (action == ACTION_LAND
                        and not self.dead_landing_penalized[agent_idx]):
                    reward_components['landing'] += DEAD_LANDING_PENALTY
                    local_rewards[agent_idx] += DEAD_LANDING_PENALTY
                    self.dead_landing_penalized[agent_idx] = True
                    dead_event = {
                        'agent': agent_idx,
                        'step': self.episode_step,
                        'position': [x, y],
                        'at_assigned_zone': bool(
                            (x, y) == tuple(self.landing_zones[agent_idx])
                        ),
                        'penalty': float(DEAD_LANDING_PENALTY),
                        'reason': 'land_action_after_drone_death',
                    }
                    step_data['dead_landing_events'].append(dead_event)
                    step_data['landing_events'].append({
                        'agent': agent_idx,
                        'successful': False,
                        'reward': float(DEAD_LANDING_PENALTY),
                        'forced_by_terminal_action_mask': False,
                        'post_death': True,
                        'energy_return_landing': False,
                        'zone_arrival_step': int(
                            self.landing_zone_arrival_steps[agent_idx]
                        ),
                        'steps_from_zone_arrival': -1,
                    })
                new_positions.append((x, y))
                continue

            if action == ACTION_LAND:
                forced_terminal_landing = bool(
                    landing_ready_at_start[agent_idx]
                )
                if forced_terminal_landing:
                    step_data['forced_terminal_landing_actions'][agent_idx] = 1
                energy_return_landing = bool(
                    not self.all_patients_resolved()
                    and (x, y) == tuple(self.landing_zones[agent_idx])
                    and self.energy_return_required(agent_idx, (x, y))
                )
                if self.landing_permitted(agent_idx):
                    self.landed[agent_idx] = True
                    landing_reward = LANDING_REWARD
                elif (x, y) == self.landing_zones[agent_idx]:
                    
                    landing_reward = EARLY_LANDING_PENALTY
                else:
                    landing_reward = LAND_WRONG_PENALTY
                reward_components['landing'] += landing_reward
                local_rewards[agent_idx] += landing_reward
                step_data['landing_events'].append({
                    'agent': agent_idx,
                    'successful': bool(self.landed[agent_idx]),
                    'reward': float(landing_reward),
                    'forced_by_terminal_action_mask': (
                        forced_terminal_landing
                    ),
                    'post_death': False,
                    'energy_return_landing': energy_return_landing,
                    'zone_arrival_step': int(
                        self.landing_zone_arrival_steps[agent_idx]
                    ),
                    'steps_from_zone_arrival': int(
                        self.episode_step
                        - self.landing_zone_arrival_steps[agent_idx]
                        if self.landing_zone_arrival_steps[agent_idx] >= 0
                        else -1
                    ),
                })
                new_positions.append((x, y))
                continue

            clean_step[agent_idx] = True
            if action == ACTION_HOVER:
                new_x, new_y = x, y
                energy_standby = bool(
                    old_phase == 'rescue'
                    and agent_idx in energy_return_agents_at_start
                    and (x, y) == tuple(self.landing_zones[agent_idx])
                )
                if energy_standby:
                    hover_penalty = ENERGY_STANDBY_HOVER_PENALTY
                else:
                    hover_penalty = (
                        LANDING_HOVER_PENALTY
                        if old_phase == 'landing' else HOVER_PENALTY
                    )
                hover_cost = hover_penalty
                reward_components['hover'] += hover_cost
                local_rewards[agent_idx] += hover_cost
                if (old_phase == 'landing'
                        and step_data['landing_progress_actions_available'][
                            agent_idx
                        ] > 0):
                    step_data[
                        'landing_hover_with_progress_available'
                    ][agent_idx] = 1
            else:
                intended_dx, intended_dy = MOVEMENT_OFFSETS[action]
                intended_position = (x + intended_dx, y + intended_dy)
                step_data['movement_actions'] += 1
                intended_destination_valid = bool(
                    0 <= intended_position[0] < self.grid_size
                    and 0 <= intended_position[1] < self.grid_size
                    and intended_position not in self.obstacles
                    and intended_position
                    not in occupied_at_start.get(agent_idx, set())
                )
                target = navigation_targets.get(agent_idx)
                intended_progress = (
                    self.manhattan_distance((x, y), target)
                    - self.manhattan_distance(intended_position, target)
                    if target is not None else 0
                )
                chosen_progress[agent_idx] = intended_progress
                if (agent_idx in wind_avoidance_agents
                        and intended_destination_valid
                        and intended_position in self.wind_zones):
                    step_data['wind_hazard_selections'] += 1
                    wind_shortcut = bool(
                        intended_progress
                        > wind_best_safe_progress[agent_idx]
                    )
                    selection_penalty = (
                        WIND_SHORTCUT_SELECTION_PENALTY
                        if wind_shortcut
                        else WIND_DOMINATED_SELECTION_PENALTY
                    ) * self.hazard_penalty_scale
                    selection_name = (
                        'wind_shortcut_hazard_selections'
                        if wind_shortcut
                        else 'wind_dominated_hazard_selections'
                    )
                    step_data[selection_name] += 1
                    reward_components['wind_routing'] += selection_penalty
                    local_rewards[agent_idx] += selection_penalty
                if (agent_idx in low_signal_avoidance_agents
                        and intended_destination_valid
                        and intended_position in self.low_signal_zones):
                    step_data['low_signal_hazard_selections'] += 1
                    low_signal_shortcut = bool(
                        intended_progress
                        > low_signal_best_safe_progress[agent_idx]
                    )
                    selection_penalty = (
                        LOW_SIGNAL_SHORTCUT_SELECTION_PENALTY
                        if low_signal_shortcut
                        else LOW_SIGNAL_DOMINATED_SELECTION_PENALTY
                    ) * self.hazard_penalty_scale
                    selection_name = (
                        'low_signal_shortcut_hazard_selections'
                        if low_signal_shortcut
                        else 'low_signal_dominated_hazard_selections'
                    )
                    step_data[selection_name] += 1
                    reward_components[
                        'low_signal_routing'
                    ] += selection_penalty
                    local_rewards[agent_idx] += selection_penalty
                if (intended_position in self.obstacles
                        or not (0 <= intended_position[0] < self.grid_size
                                and 0 <= intended_position[1] < self.grid_size)):
                    step_data['obstacle_action_selected'][agent_idx] = 1
                    if (
                        best_collision_free_progress.get(agent_idx, 0)
                        >= intended_progress
                    ):
                        step_data[
                            'dominated_obstacle_selections'
                        ][agent_idx] = 1
                        reward_components[
                            'obstacle_collision'
                        ] += DOMINATED_OBSTACLE_SELECTION_PENALTY
                        local_rewards[agent_idx] += (
                            DOMINATED_OBSTACLE_SELECTION_PENALTY
                        )
                in_low_signal = (x, y) in self.low_signal_zones
                in_wind = (x, y) in self.wind_zones
                step_data['low_signal_command_attempts'] += int(in_low_signal)
                step_data['wind_command_attempts'] += int(in_wind)
                low_signal_failure = (
                    in_low_signal
                    and random.random() < LOW_SIGNAL_FAILURE_PROB
                )
                wind_failure = (
                    in_wind
                    and random.random() < WIND_MOVEMENT_FAILURE_PROB
                )
                if low_signal_failure or wind_failure:
                    new_x, new_y = x, y
                    step_data['low_signal_failures'][agent_idx] = int(
                        low_signal_failure
                    )
                    step_data['wind_failures'][agent_idx] = int(wind_failure)
                    clean_step[agent_idx] = False
                else:
                    dx, dy = MOVEMENT_OFFSETS[action]
                    new_x, new_y = x + dx, y + dy

                if (new_x < 0 or new_x >= self.grid_size
                        or new_y < 0 or new_y >= self.grid_size
                        or (new_x, new_y) in self.obstacles):
                    step_data['obstacle_collisions'] += 1
                    obstacle_agents.append(agent_idx)
                    step_data['obstacle_collision_flags'][agent_idx] = 1
                    obstacle_streak = (
                        self.obstacle_collision_streaks[agent_idx] + 1
                    )
                    step_data['obstacle_collision_events'].append({
                        'agent': agent_idx,
                        'action': action,
                        'action_name': ACTION_NAMES[action],
                        'position': [x, y],
                        'attempted_position': [new_x, new_y],
                        'boundary': bool(
                            new_x < 0 or new_x >= self.grid_size
                            or new_y < 0 or new_y >= self.grid_size
                        ),
                        'streak': obstacle_streak,
                        'repeated': bool(obstacle_streak > 1),
                    })
                    clean_step[agent_idx] = False
                    new_x, new_y = x, y
            new_positions.append((new_x, new_y))

        active = [
            i for i in range(NUM_AGENTS)
            if not self.landed[i] and not self.battery_depleted[i]
        ]
        collision_groups = {}
        for agent_idx in active:
            collision_groups.setdefault(new_positions[agent_idx], []).append(
                agent_idx
            )

        same_destination_pairs = set()
        for agents_at_position in collision_groups.values():
            for left in range(len(agents_at_position)):
                for right in range(left + 1, len(agents_at_position)):
                    same_destination_pairs.add(tuple(sorted((
                        agents_at_position[left], agents_at_position[right]
                    ))))

        head_on_pairs = set()
        for left in range(len(active)):
            for right in range(left + 1, len(active)):
                agent_a = active[left]
                agent_b = active[right]
                if (new_positions[agent_a] == old_positions[agent_b]
                        and new_positions[agent_b] == old_positions[agent_a]
                        and old_positions[agent_a] != old_positions[agent_b]):
                    head_on_pairs.add((agent_a, agent_b))

        collision_pairs = same_destination_pairs | head_on_pairs
        colliding_agents = {
            agent_idx for pair in collision_pairs for agent_idx in pair
        }
        for agent_idx in colliding_agents:
            new_positions[agent_idx] = old_positions[agent_idx]
            clean_step[agent_idx] = False

        step_data['agent_collisions'] = len(collision_pairs)
        step_data['same_destination_collisions'] = len(same_destination_pairs)
        step_data['head_on_collisions'] = len(head_on_pairs)
        step_data['collision_pairs'] = [list(pair) for pair in sorted(collision_pairs)]
        step_data['agent_collision_flags'] = [
            int(agent_idx in colliding_agents) for agent_idx in range(NUM_AGENTS)
        ]

        new_pair_streaks = {}
        for pair in collision_pairs:
            streak = self.collision_pair_streaks.get(pair, 0) + 1
            new_pair_streaks[pair] = streak
        self.collision_pair_streaks = new_pair_streaks

        for pair in collision_pairs:
            pair_streak = new_pair_streaks[pair]
            
            
            
            
            multiplier = 1.0 + COLLISION_STREAK_PENALTY_GROWTH * min(
                pair_streak - 1, COLLISION_STREAK_CAP
            )
            pair_penalty = AGENT_COLLISION_PENALTY * multiplier
            reward_components['agent_collision'] += pair_penalty
            for agent_idx in pair:
                local_rewards[agent_idx] += pair_penalty / 2.0
                if (
                    best_collision_free_progress.get(agent_idx, 0)
                    >= chosen_progress.get(agent_idx, 0)
                ):
                    step_data[
                        'dominated_agent_conflict_selections'
                    ][agent_idx] = 1
                    reward_components['agent_collision'] += (
                        DOMINATED_AGENT_CONFLICT_PENALTY
                    )
                    local_rewards[agent_idx] += (
                        DOMINATED_AGENT_CONFLICT_PENALTY
                    )

        obstacle_agent_set = set(obstacle_agents)
        for agent_idx in range(NUM_AGENTS):
            self.obstacle_collision_streaks[agent_idx] = (
                self.obstacle_collision_streaks[agent_idx] + 1
                if agent_idx in obstacle_agent_set else 0
            )
        if operational_at_start and obstacle_agents:
            for agent_idx in obstacle_agents:
                obstacle_multiplier = (
                    1.0 + COLLISION_STREAK_PENALTY_GROWTH * min(
                        self.obstacle_collision_streaks[agent_idx] - 1,
                        COLLISION_STREAK_CAP
                    )
                )
                obstacle_penalty = COLLISION_PENALTY * obstacle_multiplier
                reward_components['obstacle_collision'] += obstacle_penalty
                local_rewards[agent_idx] += obstacle_penalty

        for agent_idx, landing_ready in enumerate(landing_ready_at_start):
            if (landing_ready
                    and not self.landed[agent_idx]
                    and tuple(new_positions[agent_idx])
                    != tuple(self.landing_zones[agent_idx])):
                step_data['landing_zone_departure_events'].append({
                    'agent': agent_idx,
                    'step': self.episode_step,
                    'action': actions[agent_idx],
                    'action_name': ACTION_NAMES[actions[agent_idx]],
                    'from': list(old_positions[agent_idx]),
                    'to': list(new_positions[agent_idx]),
                })
                self.landing_zone_arrival_steps[agent_idx] = -1

        self.agents = new_positions
        for agent_idx in range(NUM_AGENTS):
            displacement = (
                self.agents[agent_idx][0] - old_positions[agent_idx][0],
                self.agents[agent_idx][1] - old_positions[agent_idx][1],
            )
            self.previous_displacements[agent_idx] = displacement
            self.previous_actions[agent_idx] = actions[agent_idx]
            had_collision = (
                agent_idx in colliding_agents
                or agent_idx in obstacle_agent_set
            )
            self.previous_collision_flags[agent_idx] = had_collision
            self.collision_streaks[agent_idx] = (
                self.collision_streaks[agent_idx] + 1
                if had_collision else 0
            )
            self.agent_path_lengths[agent_idx] += (
                abs(displacement[0]) + abs(displacement[1])
            )
            self.agent_unique_positions[agent_idx].add(self.agents[agent_idx])

        for agent_idx in range(NUM_AGENTS):
            if self.landed[agent_idx] or self.battery_depleted[agent_idx]:
                continue

            was_in_wind = old_positions[agent_idx] in self.wind_zones
            in_wind = self.agents[agent_idx] in self.wind_zones
            step_data['wind_entries'][agent_idx] = int(
                in_wind and not was_in_wind
            )
            step_data['wind_exits'][agent_idx] = int(
                was_in_wind and not in_wind
            )
            landing_standby = bool(
                old_phase == 'rescue'
                and actions[agent_idx] == ACTION_HOVER
                and agent_idx in energy_return_agents_at_start
                and tuple(self.agents[agent_idx])
                == tuple(self.landing_zones[agent_idx])
            )
            base_battery_drain = (
                BATTERY_DRAIN_AT_LANDING_ZONE
                if landing_standby else BATTERY_DRAIN_PER_STEP
            )
            wind_battery_drain = BATTERY_DRAIN_IN_WIND if in_wind else 0.0
            total_battery_drain = base_battery_drain + wind_battery_drain
            self.batteries[agent_idx] -= total_battery_drain
            step_data['battery_drain_by_agent'][agent_idx] = float(
                total_battery_drain
            )
            step_data['wind_battery_drain_by_agent'][agent_idx] = float(
                wind_battery_drain
            )
            step_data['landing_standby_steps'][agent_idx] = int(
                landing_standby
            )
            energy_usage_penalty = (
                ENERGY_USAGE_PENALTY_PER_UNIT * total_battery_drain
            )
            reward_components['battery'] += energy_usage_penalty
            local_rewards[agent_idx] += energy_usage_penalty

            if in_wind:
                penalty = WIND_PENALTY * self.hazard_penalty_scale
                if step_data['wind_entries'][agent_idx]:
                    penalty += (
                        WIND_ENTRY_PENALTY * self.hazard_penalty_scale
                    )
                reward_components['wind'] += penalty
                local_rewards[agent_idx] += penalty
                step_data['wind_exposure_steps'][agent_idx] = 1
                clean_step[agent_idx] = False

            was_in_low_signal = (
                old_positions[agent_idx] in self.low_signal_zones
            )
            in_low_signal = self.agents[agent_idx] in self.low_signal_zones
            step_data['low_signal_entries'][agent_idx] = int(
                in_low_signal and not was_in_low_signal
            )
            step_data['low_signal_exits'][agent_idx] = int(
                was_in_low_signal and not in_low_signal
            )
            if in_low_signal:
                penalty = LOW_SIGNAL_PENALTY * self.hazard_penalty_scale
                if step_data['low_signal_entries'][agent_idx]:
                    penalty += (
                        LOW_SIGNAL_ENTRY_PENALTY
                        * self.hazard_penalty_scale
                    )
                reward_components['low_signal'] += penalty
                local_rewards[agent_idx] += penalty
                step_data['low_signal_exposure_steps'][agent_idx] = 1
                clean_step[agent_idx] = False

            if 0 < self.batteries[agent_idx] < LOW_BATTERY_THRESHOLD:
                penalty = LOW_BATTERY_PENALTY
                reward_components['low_battery'] += penalty
                local_rewards[agent_idx] += penalty

            safe_return_margin = self.safe_return_battery_margin(agent_idx)
            step_data['safe_return_margin_after'][agent_idx] = float(
                safe_return_margin
            )
            energy_margin_delta = (
                safe_return_margin
                - step_data['safe_return_margin_before'][agent_idx]
            )
            step_data['energy_margin_delta_by_agent'][agent_idx] = float(
                energy_margin_delta
            )
            if agent_idx in energy_return_agents_at_start:
                energy_navigation_reward = (
                    ENERGY_MARGIN_SHAPING_FACTOR
                    * float(np.clip(energy_margin_delta, -1.0, 1.0))
                )
                reward_components[
                    'energy_navigation'
                ] += energy_navigation_reward
                local_rewards[agent_idx] += energy_navigation_reward
                step_data['energy_return_progress_flags'][agent_idx] = int(
                    energy_margin_delta > 1e-8
                )
                step_data['energy_return_regress_flags'][agent_idx] = int(
                    energy_margin_delta < -1e-8
                )
            if safe_return_margin < 0.0:
                reward_components[
                    'low_battery'
                ] += SAFE_RETURN_RESERVE_PENALTY
                local_rewards[agent_idx] += SAFE_RETURN_RESERVE_PENALTY
                step_data['reserve_violation_flags'][agent_idx] = 1

            if self.batteries[agent_idx] <= 0:
                reward_components['battery'] += BATTERY_DEPLETION_PENALTY
                local_rewards[agent_idx] += BATTERY_DEPLETION_PENALTY
                self.batteries[agent_idx] = 0
                self.battery_depleted[agent_idx] = True
                self.drone_died[agent_idx] = True
                self.death_reminder_steps_remaining[
                    agent_idx
                ] = POST_DEPLETION_REMINDER_STEPS
                step_data['death_penalty_applications'][agent_idx] = 1
                died_at_assigned_zone = bool(
                    tuple(self.agents[agent_idx])
                    == tuple(self.landing_zones[agent_idx])
                )
                step_data['battery_depletion_events'].append({
                    'agent': agent_idx,
                    'step': self.episode_step,
                    'position': list(self.agents[agent_idx]),
                    'battery_drain': float(total_battery_drain),
                    'wind_battery_drain': float(wind_battery_drain),
                    'safe_return_margin': float(safe_return_margin),
                    'drone_died': True,
                    'died_at_assigned_landing_zone': died_at_assigned_zone,
                    'full_death_penalty': float(BATTERY_DEPLETION_PENALTY),
                    'scheduled_reminder_penalties': int(
                        POST_DEPLETION_REMINDER_STEPS
                    ),
                })
                if (died_at_assigned_zone
                        and not self.dead_landing_penalized[agent_idx]):
                    
                    
                    
                    reward_components['landing'] += DEAD_LANDING_PENALTY
                    local_rewards[agent_idx] += DEAD_LANDING_PENALTY
                    self.dead_landing_penalized[agent_idx] = True
                    dead_event = {
                        'agent': agent_idx,
                        'step': self.episode_step,
                        'position': list(self.agents[agent_idx]),
                        'at_assigned_zone': True,
                        'penalty': float(DEAD_LANDING_PENALTY),
                        'reason': 'arrived_at_pad_on_depletion_transition',
                    }
                    step_data['dead_landing_events'].append(dead_event)
                    step_data['landing_events'].append({
                        'agent': agent_idx,
                        'successful': False,
                        'reward': float(DEAD_LANDING_PENALTY),
                        'forced_by_terminal_action_mask': False,
                        'post_death': True,
                        'energy_return_landing': False,
                        'zone_arrival_step': -1,
                        'steps_from_zone_arrival': -1,
                    })
                clean_step[agent_idx] = False

        if operational_at_start:
            clean_unit_reward = CLEAN_STEP_BONUS / len(operational_at_start)
            for agent_idx in operational_at_start:
                if clean_step[agent_idx]:
                    reward_components['clean'] += clean_unit_reward
                    local_rewards[agent_idx] += clean_unit_reward

        if self.dynamic_spawning and not self.all_patients_spawned():
            self.new_patient_timer -= 1
        if (self.dynamic_spawning
                and not self.all_patients_spawned()
                and self.new_patient_timer <= 0):
            remaining_patients = sum(
                not self.patient_active[patient_idx]
                for patient_idx in range(self.episode_max_patients)
            )
            maximum_interval = (
                self.patient_spawn_interval + self.patient_spawn_jitter
            )
            future_spawn_events = max(0, (
                self.final_patient_spawn_step - self.episode_step
            ) // max(1, maximum_interval))
            if self.episode_step >= self.final_patient_spawn_step:
                
                
                minimum_spawn = remaining_patients
                maximum_spawn = remaining_patients
            else:
                maximum_spawn = min(
                    self.maximum_patient_spawn_batch, remaining_patients
                )
                minimum_spawn = min(
                    maximum_spawn,
                    max(
                        self.minimum_patient_spawn_batch,
                        remaining_patients
                        - self.maximum_patient_spawn_batch
                        * future_spawn_events,
                    ),
                ) if maximum_spawn else 0
            spawn_count = (
                random.randint(minimum_spawn, maximum_spawn)
                if maximum_spawn else 0
            )
            for _ in range(spawn_count):
                for patient_idx in range(self.episode_max_patients):
                    if not self.patient_active[patient_idx]:
                        self.patient_active[patient_idx] = True
                        self.patient_timers[
                            patient_idx
                        ] = self.episode_patient_timer
                        self.patient_initial_timers[
                            patient_idx
                        ] = self.episode_patient_timer
                        
                        
                        
                        
                        self.patient_weights[patient_idx] = (
                            self.initial_patient_weights[patient_idx]
                        )
                        self.updated_patient_weights[patient_idx] = (
                            float(self.patient_weights[patient_idx])
                        )
                        self.sample_patient_survival_profile(patient_idx)
                        self.patients_died[patient_idx] = False
                        self.patient_spawn_steps[patient_idx] = self.episode_step
                        step_data['patient_spawn_events'].append(patient_idx)
                        break
            self.new_patient_timer = (
                self.sample_spawn_interval()
                if not self.all_patients_spawned() else 0
            )

        for patient_idx in range(self.episode_max_patients):
            if (not self.patient_active[patient_idx]
                    or self.patients_delivered[patient_idx]):
                continue
            self.patient_timers[patient_idx] -= 1
            if self.patient_timers[patient_idx] <= 0:
                
                
                
                
                initial_weight = self.initial_patient_weights[patient_idx]
                death_penalty = PATIENT_DEATH_PENALTY * (
                    1.0
                    + PRIORITY_DEATH_PENALTY_GROWTH * (initial_weight - 1)
                )
                reward_components['patient_death'] += death_penalty
                self.patients_delivered[patient_idx] = True
                self.patients_died[patient_idx] = True
                self.patient_resolution_steps[patient_idx] = self.episode_step
                step_data['patient_death_events'].append({
                    'patient': patient_idx,
                    'initial_weight': self.initial_patient_weights[patient_idx],
                    'final_weight': self.patient_weights[patient_idx],
                    'initial_timer': self.patient_initial_timers[patient_idx],
                    'response_time_ratio': 1.0,
                    'response_time_steps': int(
                        self.episode_step
                        - self.patient_spawn_steps[patient_idx]
                    ),
                    'survival_probability': float(
                        self.patient_survival_probabilities[patient_idx]
                    ),
                    'death_penalty': float(death_penalty),
                    'step': self.episode_step,
                })
                continue

            previous_weight = int(self.patient_weights[patient_idx])
            self.update_patient_deterioration(patient_idx)
            if self.patient_weights[patient_idx] > previous_weight:
                step_data['patient_weight_escalation_events'].append({
                    'patient': patient_idx,
                    'step': self.episode_step,
                    'elapsed_steps': int(
                        self.episode_step
                        - self.patient_spawn_steps[patient_idx]
                    ),
                    'timer_remaining': int(
                        self.patient_timers[patient_idx]
                    ),
                    'from_weight': previous_weight,
                    'to_weight': int(self.patient_weights[patient_idx]),
                    'survival_probability': float(
                        self.patient_survival_probabilities[patient_idx]
                    ),
                })

        
        
        
        
        
        
        
        response_wait_cost = 0.0
        for patient_idx in range(self.episode_max_patients):
            if (not self.patient_active[patient_idx]
                    or self.patients_delivered[patient_idx]):
                continue
            response_fraction = float(np.clip(
                (
                    self.episode_step
                    - self.patient_spawn_steps[patient_idx]
                ) / max(1, self.patient_initial_timers[patient_idx]),
                0.0,
                1.0,
            ))
            priority_fraction = (
                self.initial_patient_weights[patient_idx]
                / MAX_PATIENT_WEIGHT
            )
            response_wait_cost += (
                RESPONSE_WAIT_PENALTY_PER_PATIENT
                * priority_fraction
                * response_fraction
            )
        reward_components['response_time'] += response_wait_cost
        step_data['response_wait_cost'] = float(response_wait_cost)

        for agent_idx in range(NUM_AGENTS):
            if self.landed[agent_idx]:
                continue
            for patient_idx in range(self.episode_max_patients):
                if (not self.patient_active[patient_idx]
                        or self.patients_delivered[patient_idx]):
                    continue
                if self.agents[agent_idx] != self.patient_positions[patient_idx]:
                    continue
                timer_ratio = (
                    self.patient_timers[patient_idx]
                    / max(1, self.patient_initial_timers[patient_idx])
                )
                response_time_ratio = float(np.clip(
                    1.0 - timer_ratio, 0.0, 1.0
                ))
                initial_priority_fraction = (
                    self.initial_patient_weights[patient_idx]
                    / MAX_PATIENT_WEIGHT
                )
                delivery_reward = (
                    DELIVERY_COMPLETION_BONUS
                    + GOAL_REWARD * timer_ratio
                    * self.initial_patient_weights[patient_idx]
                )
                response_reward = (
                    RESPONSE_TIME_DELIVERY_REWARD
                    * initial_priority_fraction
                    * (1.0 - response_time_ratio) ** 2
                )
                service_potential_before = self.priority_service_potential()
                reward_components['delivery'] += delivery_reward
                reward_components['response_time'] += response_reward
                local_rewards[agent_idx] += delivery_reward
                local_rewards[agent_idx] += response_reward
                self.patients_delivered[patient_idx] = True
                self.patients_actually_delivered[patient_idx] = True
                self.patient_resolution_steps[patient_idx] = self.episode_step
                self.patient_delivery_agents[patient_idx] = agent_idx
                if self.first_delivery_step < 0:
                    self.first_delivery_step = self.episode_step
                self.last_delivery_step = self.episode_step
                service_potential_after = self.priority_service_potential()
                priority_service_reward = (
                    PRIORITY_SERVICE_POTENTIAL_SCALE
                    * max(
                        0.0,
                        service_potential_after - service_potential_before,
                    )
                )
                reward_components['fairness'] += priority_service_reward
                local_rewards[agent_idx] += priority_service_reward
                step_data['patient_delivery_events'].append({
                    'patient': patient_idx,
                    'agent': agent_idx,
                    'step': self.episode_step,
                    'timer_remaining': self.patient_timers[patient_idx],
                    'initial_timer': self.patient_initial_timers[patient_idx],
                    'response_time_ratio': float(
                        response_time_ratio
                    ),
                    'response_time_steps': int(
                        self.episode_step
                        - self.patient_spawn_steps[patient_idx]
                    ),
                    'initial_weight': self.initial_patient_weights[patient_idx],
                    'current_weight': self.patient_weights[patient_idx],
                    'survival_probability': float(
                        self.patient_survival_probabilities[patient_idx]
                    ),
                    'delivery_reward': float(delivery_reward),
                    'response_reward': float(response_reward),
                    'priority_service_reward': float(
                        priority_service_reward
                    ),
                    'priority_service_potential_before': float(
                        service_potential_before
                    ),
                    'priority_service_potential_after': float(
                        service_potential_after
                    ),
                    'reward': float(
                        delivery_reward
                        + response_reward
                        + priority_service_reward
                    ),
                })

        
        
        
        for agent_idx in range(NUM_AGENTS):
            landing_ready_now = self.landing_permitted(agent_idx)
            if (landing_ready_now
                    and self.landing_zone_arrival_steps[agent_idx] < 0):
                self.landing_zone_arrival_steps[agent_idx] = self.episode_step
                step_data['landing_zone_arrival_events'].append({
                    'agent': agent_idx,
                    'step': self.episode_step,
                    'position': list(self.agents[agent_idx]),
                    'energy_return_arrival': bool(
                        not self.all_patients_resolved()
                    ),
                })

        available = [
            i for i in range(NUM_AGENTS)
            if not self.landed[i] and not self.battery_depleted[i]
        ]
        minimum_agent_distance = 2 * self.grid_size
        for left in range(len(available)):
            for right in range(left + 1, len(available)):
                agent_a = available[left]
                agent_b = available[right]
                distance = self.manhattan_distance(
                    self.agents[agent_a], self.agents[agent_b]
                )
                minimum_agent_distance = min(minimum_agent_distance, distance)
                if 0 < distance <= CLOSENESS_RADIUS:
                    pair_penalty = CLOSENESS_PENALTY
                    reward_components['closeness'] += pair_penalty
                    local_rewards[agent_a] += pair_penalty / 2.0
                    local_rewards[agent_b] += pair_penalty / 2.0
                    step_data['closeness_pairs'] += 1
        step_data['minimum_agent_distance'] = (
            minimum_agent_distance if len(available) > 1 else -1
        )

        resolution_transition = (
            self.all_patients_resolved()
            and self.all_patients_resolved_step < 0
        )
        if resolution_transition:
            self.all_patients_resolved_step = self.episode_step
            self.landing_deadline = (
                self.all_patients_resolved_step + self.landing_grace_steps
            )
            
            
            reward_components['rescue_outcome'] += self.rescue_outcome_reward()
            reward_components['fairness'] += self.fairness_outcome_reward()
            reward_components[
                'triage_ordering'
            ] += self.triage_ordering_outcome_reward()
            reward_components[
                'triage_response'
            ] += self.triage_response_outcome_reward()
        failed_patient_outcome = (
            self.all_patients_resolved()
            and not self.rescue_success()
        )
        if failed_patient_outcome and self.irrecoverable_step < 0:
            self.irrecoverable_step = self.episode_step

        rescue_success = self.rescue_success()
        safe_return = self.safe_return_complete()
        success = self.mission_success()
        if safe_return and self.landing_completion_step < 0:
            self.landing_completion_step = self.episode_step
        fleet_inactive = all(
            self.landed[i] or self.battery_depleted[i]
            for i in range(NUM_AGENTS)
        )
        fleet_inactive_with_losses = bool(
            fleet_inactive
            and self.all_patients_resolved()
            and any(self.drone_died)
        )
        rescue_timeout = (
            not self.all_patients_resolved()
            and self.episode_step >= self.episode_max_steps
        )
        landing_timeout = (
            self.all_patients_resolved()
            and self.episode_step >= self.current_episode_deadline()
        )
        
        
        
        
        
        done = (
            safe_return
            or fleet_inactive_with_losses
            or rescue_timeout
            or landing_timeout
        )
        if success:
            self.termination_reason = "mission_success"
        elif safe_return:
            self.termination_reason = (
                "safe_return_below_application_thresholds"
                if self.perfect_rescue()
                else "safe_return_with_patient_losses"
            )
        elif fleet_inactive_with_losses:
            self.termination_reason = "fleet_exhausted_after_outcome"
        elif rescue_timeout:
            self.termination_reason = "rescue_timeout"
        elif landing_timeout:
            self.termination_reason = "landing_timeout"
        else:
            self.termination_reason = "in_progress"

        post_step_potential = self.fleet_potential_components()

        local_navigation_delta = np.zeros(NUM_AGENTS, dtype=np.float32)
        navigation_events = []
        for target in old_potential['navigation_targets']:
            agent_idx = int(target['agent'])
            target_position = tuple(target['position'])
            old_distance = self.manhattan_distance(
                old_positions[agent_idx], target_position
            )
            new_distance = self.manhattan_distance(
                self.agents[agent_idx], target_position
            )
            distance_delta = float(old_distance - new_distance)
            if distance_delta < -1.0 or distance_delta > 1.0:
                raise RuntimeError(
                    'A one-step navigation distance changed by more than one'
                )
            local_navigation_delta[agent_idx] = distance_delta
            if old_phase == 'landing':
                if distance_delta > 0.0:
                    step_data['landing_distance_reduced'][agent_idx] = 1
                elif distance_delta < 0.0:
                    step_data['landing_distance_increased'][agent_idx] = 1
                else:
                    step_data['landing_distance_unchanged'][agent_idx] = 1
            navigation_events.append({
                'agent': agent_idx,
                'kind': str(target['kind']),
                'index': int(target['index']),
                'old_distance': int(old_distance),
                'new_distance': int(new_distance),
                'distance_delta': float(distance_delta),
            })
        local_potential_rewards = SHAPING_FACTOR * local_navigation_delta
        step_data['wind_entry_progress_cells'] = float(sum(
            max(0.0, float(local_navigation_delta[agent_idx]))
            * step_data['wind_entries'][agent_idx]
            for agent_idx in range(NUM_AGENTS)
        ))
        step_data['low_signal_entry_progress_cells'] = float(sum(
            max(0.0, float(local_navigation_delta[agent_idx]))
            * step_data['low_signal_entries'][agent_idx]
            for agent_idx in range(NUM_AGENTS)
        ))
        local_rewards += local_potential_rewards
        reward_components['potential'] = float(local_potential_rewards.sum())
        potential_delta = float(local_navigation_delta.sum())
        raw_potential_change = potential_delta
        new_potential = post_step_potential
        step_data['navigation_events'] = navigation_events

        if safe_return:
            reward_components['safe_return'] += self.safe_return_reward()
            if success:
                reward_components['mission'] += MISSION_SUCCESS_REWARD
        elif done:
            delivered_fraction = (
                sum(
                    self.patients_actually_delivered[:self.episode_max_patients]
                ) / self.episode_max_patients
            )
            landed_fraction = (
                sum(self.landed) / NUM_AGENTS
                if self.all_patients_resolved() else 0.0
            )
            remaining_failure = max(
                0.0,
                1.0 - 0.75 * delivered_fraction - 0.25 * landed_fraction
            )
            reward_components['mission'] += (
                MISSION_FAILURE_PENALTY * remaining_failure
            )

        team_reward = float(sum(reward_components.values()))
        unattributed_reward = team_reward - float(local_rewards.sum())
        local_rewards += unattributed_reward / NUM_AGENTS
        if not np.isclose(local_rewards.sum(), team_reward, atol=1e-5):
            raise RuntimeError('Per-agent rewards do not sum to team reward')

        event_flags = 0
        if step_data['patient_delivery_events']:
            event_flags |= EVENT_DELIVERY
        if collision_pairs:
            event_flags |= EVENT_COLLISION
        if raw_potential_change > 1e-8:
            event_flags |= EVENT_PROGRESS
        if any(event['successful'] for event in step_data['landing_events']):
            event_flags |= EVENT_LANDING
        if done:
            event_flags |= EVENT_TERMINAL
        if old_phase == 'landing':
            event_flags |= EVENT_LANDING_PHASE
        if obstacle_agents:
            event_flags |= EVENT_OBSTACLE_COLLISION
        if (
                step_data['wind_hazard_selections']
                or step_data['low_signal_hazard_selections']
                or any(step_data['wind_entries'])
                or any(step_data['low_signal_entries'])
                or any(step_data['wind_exposure_steps'])
                or any(step_data['low_signal_exposure_steps'])):
            event_flags |= EVENT_HAZARD
        
        
        previous_wind_zones = set(self.wind_zones)
        previous_low_signal_zones = set(self.low_signal_zones)
        wind_refresh_due = self.wind_timer <= 1
        low_signal_refresh_due = self.low_signal_timer <= 1
        self.update_wind_zones()
        self.update_low_signal_zones()
        if wind_refresh_due or low_signal_refresh_due:
            self.refresh_return_energy_maps()
        step_data['wind_zone_refreshed'] = int(wind_refresh_due)
        step_data['low_signal_zone_refreshed'] = int(low_signal_refresh_due)
        if wind_refresh_due:
            step_data['wind_refresh_onset_agents'] = [
                int(
                    not self.landed[agent_idx]
                    and not self.battery_depleted[agent_idx]
                    and
                    tuple(self.agents[agent_idx]) not in previous_wind_zones
                    and tuple(self.agents[agent_idx]) in self.wind_zones
                )
                for agent_idx in range(NUM_AGENTS)
            ]
        if low_signal_refresh_due:
            step_data['low_signal_refresh_onset_agents'] = [
                int(
                    not self.landed[agent_idx]
                    and not self.battery_depleted[agent_idx]
                    and tuple(self.agents[agent_idx])
                    not in previous_low_signal_zones
                    and tuple(self.agents[agent_idx])
                    in self.low_signal_zones
                )
                for agent_idx in range(NUM_AGENTS)
            ]
        next_states = self.get_state()
        step_data.update({
            'team_reward': team_reward,
            'local_rewards': local_rewards.tolist(),
            'reward_components': reward_components,
            'potential_before': old_potential,
            'potential_after': new_potential,
            'potential_delta': float(potential_delta),
            'raw_potential_change': float(raw_potential_change),
            'local_potential_rewards': local_potential_rewards.tolist(),
            'phase_after': self.mission_phase(),
            'termination_reason': self.termination_reason,
            'event_flags': int(event_flags),
            'max_collision_streak': max(self.collision_streaks, default=0),
            'max_agent_collision_streak': max(
                new_pair_streaks.values(), default=0
            ),
            'max_obstacle_collision_streak': max(
                self.obstacle_collision_streaks, default=0
            ),
            'environment_step_latency_ms': (
                time.perf_counter_ns() - step_start_ns
            ) / 1_000_000.0,
        })
        rewards = [team_reward / NUM_AGENTS] * NUM_AGENTS
        return next_states, rewards, done, step_data

  
    def generate_random_positions(
        self, maximum_path_distance=0, active_patient_count=MAX_PATIENTS,
        maximum_landing_path_distance=0,
    ):

        total_needed = NUM_AGENTS + MAX_PATIENTS + NUM_AGENTS
        main_component = self.largest_traversable_component()
        eligible_positions = [
            position for position in main_component
            if 1 <= position[0] < self.grid_size - 1
            and 1 <= position[1] < self.grid_size - 1
        ]
        if len(eligible_positions) < total_needed:
            raise RuntimeError(
                'Largest traversable component cannot hold every mission entity'
            )

        def choose(candidates, count, selected, preferred_spacing=5):
            for minimum_distance in range(preferred_spacing, 0, -1):
                for _ in range(30):
                    shuffled = list(candidates)
                    random.shuffle(shuffled)
                    result = []
                    for position in shuffled:
                        if position in selected or position in result:
                            continue
                        if any(
                            self.manhattan_distance(position, existing)
                            < minimum_distance
                            for existing in selected + result
                        ):
                            continue
                        result.append(position)
                        if len(result) == count:
                            return result
            raise RuntimeError(
                f'Unable to place {count} curriculum entities in connected space'
            )

        starts = choose(
            eligible_positions, NUM_AGENTS, [], preferred_spacing=10
        )
        selected = list(starts)
        maximum_path_distance = max(0, int(maximum_path_distance))
        maximum_landing_path_distance = max(
            0, int(maximum_landing_path_distance)
        )
        active_patient_count = max(
            1, min(MAX_PATIENTS, int(active_patient_count))
        )
        start_distance_maps = [
            self.shortest_path_distance_map(start) for start in starts
        ]

        if maximum_path_distance > 0:
            bounded_patient_candidates = [
                position for position in eligible_positions
                if 1 <= min(
                    int(distance_map[position[0], position[1]])
                    for distance_map in start_distance_maps
                ) <= maximum_path_distance
            ]
            active_patients = choose(
                bounded_patient_candidates,
                active_patient_count,
                selected,
                preferred_spacing=4,
            )
        else:
            active_patients = choose(
                eligible_positions, active_patient_count,
                selected,
                preferred_spacing=8,
            )
        selected.extend(active_patients)

        landing_zones = []
        for agent_idx, distance_map in enumerate(start_distance_maps):
            candidates = eligible_positions
            if maximum_landing_path_distance > 0:
                candidates = [
                    position for position in eligible_positions
                    if 1 <= int(distance_map[position[0], position[1]])
                    <= maximum_landing_path_distance
                ]
            landing_zone = choose(
                candidates, 1, selected + landing_zones,
                preferred_spacing=4,
            )[0]
            landing_zones.append(landing_zone)
        selected.extend(landing_zones)

        inactive_count = MAX_PATIENTS - active_patient_count
        inactive_patients = choose(
            eligible_positions,
            inactive_count,
            selected,
            preferred_spacing=4,
        ) if inactive_count else []
        patients = active_patients + inactive_patients

        self.start_positions = starts
        self.patient_positions = patients
        self.landing_zones = landing_zones

    def reset(self, curriculum_stage=None):
        stage_index = (
            len(CURRICULUM_STAGES) - 1
            if curriculum_stage is None else int(curriculum_stage)
        )
        self.configure_curriculum_stage(stage_index)
        if not self.fixed_layout:
            self.generate_random_positions(
                maximum_path_distance=self.curriculum_max_distance,
                active_patient_count=self.episode_max_patients,
                maximum_landing_path_distance=(
                    self.curriculum_max_landing_distance
                ),
            )

        self.refresh_landing_distance_maps()

        self.agents = list(self.start_positions)
        self.batteries = [MAX_BATTERY] * NUM_AGENTS 
        self.landed = [False] * NUM_AGENTS
        self.battery_depleted = [False] * NUM_AGENTS
        self.drone_died = [False] * NUM_AGENTS
        self.death_reminder_steps_remaining = [0] * NUM_AGENTS
        self.dead_landing_penalized = [False] * NUM_AGENTS
        self.previous_actions = [ACTION_HOVER] * NUM_AGENTS
        self.previous_displacements = [(0, 0)] * NUM_AGENTS
        self.previous_collision_flags = [False] * NUM_AGENTS
        self.collision_streaks = [0] * NUM_AGENTS
        self.obstacle_collision_streaks = [0] * NUM_AGENTS
        self.collision_pair_streaks = {}
        self.landing_zone_arrival_steps = [-1] * NUM_AGENTS
        
        self.curriculum_initial_distances = [
            self.landing_shortest_distance(agent_index, position)
            for agent_index, position in enumerate(self.start_positions)
        ]
        self.curriculum_start_step = 0
        self.agent_path_lengths = [0] * NUM_AGENTS
        self.agent_unique_positions = [
            set([self.agents[i]]) for i in range(NUM_AGENTS)
        ]

        self.patients_delivered = [False] * MAX_PATIENTS
        self.patients_actually_delivered = [False] * MAX_PATIENTS
        self.patients_died = [False] * MAX_PATIENTS
        self.patient_timers = [self.episode_patient_timer] * MAX_PATIENTS
        self.patient_initial_timers = (
            [self.episode_patient_timer] * MAX_PATIENTS
        )
        self.patient_weights = [self.random_weight() for _ in range(MAX_PATIENTS)]
        self.initial_patient_weights = self.patient_weights.copy()
        self.updated_patient_weights = [float(value) for value in self.patient_weights]
        self.patient_decay_rates = [0.0] * MAX_PATIENTS
        self.patient_survival_offsets = [0.0] * MAX_PATIENTS
        self.patient_serious_thresholds = [0.0] * MAX_PATIENTS
        self.patient_critical_thresholds = [0.0] * MAX_PATIENTS
        self.patient_survival_probabilities = [1.0] * MAX_PATIENTS
        for patient_idx in range(MAX_PATIENTS):
            self.sample_patient_survival_profile(patient_idx)
        self.patient_active = [
            i < self.episode_initial_patients
            for i in range(MAX_PATIENTS)
        ]
        self.patient_spawn_steps = [
            0 if i < self.episode_initial_patients else -1
            for i in range(MAX_PATIENTS)
        ]
        self.patient_resolution_steps = [-1] * MAX_PATIENTS
        self.patient_delivery_agents = [-1] * MAX_PATIENTS
        self.episode_step = 0
        self.new_patient_timer = (
            self.sample_spawn_interval()
            if self.dynamic_spawning and not self.all_patients_spawned()
            else 0
        )
        self.first_delivery_step = -1
        self.last_delivery_step = -1
        self.all_patients_resolved_step = -1
        self.irrecoverable_step = -1
        self.landing_deadline = -1
        self.landing_completion_step = -1
        self.termination_reason = "in_progress"

        self.wind_zones = set()
        self.wind_rectangles = []
        self.wind_raster = self.cells_to_raster(self.wind_zones)
        self.wind_timer = WIND_APPEAR_INTERVAL
        self.low_signal_zones = set()
        self.low_signal_rectangles = []
        self.low_signal_raster = self.cells_to_raster(self.low_signal_zones)
        self.low_signal_timer = LOW_SIGNAL_APPEAR_INTERVAL

        self.astar_paths = self.compute_astar_paths()
        self.hazard_candidates = list({
            position for path in self.astar_paths for position in path
        })
        
        
        self.update_wind_zones(force=True)
        self.update_low_signal_zones(force=True)
        self.refresh_return_energy_maps()

        return self.get_state()

    
    
    

    
    
    
    
    

    
    
    
    
    

    
    
    
    
    

    
    
    
    
    

    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
    
    
    
    
            
    
    
    
    

    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    

    

        
    def render(self, screen):
        screen.fill((255, 255, 255))
        font_small = pygame.font.SysFont("arial", 7)

        def indexed_color(index, saturation=80, value=80):

            color = pygame.Color(0)
            color.hsva = (
                (index * 137.508) % 360,
                saturation,
                value,
                100
            )
            return color.r, color.g, color.b

        
        for x in range(0, WINDOW_SIZE, self.cell_size):
            pygame.draw.line(screen, (200,200,200), (x,0), (x,WINDOW_SIZE))
        for y in range(0, WINDOW_SIZE, self.cell_size):
            pygame.draw.line(screen, (200,200,200), (0,y), (WINDOW_SIZE,y))

        
        for obs in self.obstacles:
            pygame.draw.rect(screen, (0,0,0),
                pygame.Rect(obs[0]*self.cell_size, obs[1]*self.cell_size,
                            self.cell_size, self.cell_size))

        
        for wz in self.wind_zones:
            pygame.draw.rect(screen, (255,165,0),
                pygame.Rect(wz[0]*self.cell_size, wz[1]*self.cell_size,
                            self.cell_size, self.cell_size))

        
        for lsz in self.low_signal_zones:
            pygame.draw.rect(screen, (138,43,226),
                pygame.Rect(lsz[0]*self.cell_size, lsz[1]*self.cell_size,
                            self.cell_size, self.cell_size))

        
        
        for index, landing_zone in enumerate(self.landing_zones):
            zone_rect = pygame.Rect(
                landing_zone[0] * self.cell_size,
                landing_zone[1] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            zone_fill = indexed_color(index, saturation=25, value=100)
            zone_border = indexed_color(index, saturation=80, value=80)

            pygame.draw.rect(screen, zone_fill, zone_rect)
            pygame.draw.rect(screen, zone_border, zone_rect, 2)

            zone_label = font_small.render(
                f"LZ{index + 1}",
                True,
                (0, 0, 0)
            )
            screen.blit(
                zone_label,
                (zone_rect.x + 1, zone_rect.y + 1)
            )

        
        for p in range(MAX_PATIENTS):
            if not self.patient_active[p]:
                continue
            pp = self.patient_positions[p]
            if self.patients_delivered[p]:
                color = (180, 180, 180)   
            else:
                ratio = (
                    self.patient_timers[p]
                    / max(1, self.patient_initial_timers[p])
                )
                if ratio > 0.6:
                    color = (255, 120, 120)   
                elif ratio > 0.3:
                    color = (255, 200, 50)    
                else:
                    color = (220, 0, 0)      
            pygame.draw.rect(screen, color,
                pygame.Rect(pp[0]*self.cell_size, pp[1]*self.cell_size,
                            self.cell_size, self.cell_size))
            
            
            weight_text = f"W:{self.patient_weights[p]}"
            surf_weight = font_small.render(weight_text, True, (50, 50, 50))
            screen.blit(surf_weight, (pp[0]*self.cell_size, pp[1]*self.cell_size))
            
            if not self.patients_delivered[p]:
                timer_text = f"T:{self.patient_timers[p]}"
                surf_timer = font_small.render(timer_text, True, (0,0,0))
                screen.blit(surf_timer, (pp[0]*self.cell_size, pp[1]*self.cell_size + 8))

        
        
        for agent_index, position in enumerate(self.agents):
            landed = (
                agent_index < len(self.landed)
                and self.landed[agent_index]
            )

            battery = (
                self.batteries[agent_index]
                if agent_index < len(self.batteries)
                else 0
            )

            agent_color = indexed_color(
                agent_index,
                saturation=85,
                value=35 if landed else 80
            )

            center = (
                position[0] * self.cell_size + self.cell_size // 2,
                position[1] * self.cell_size + self.cell_size // 2
            )
            radius = max(2, self.cell_size // 3)

            pygame.draw.circle(
                screen,
                agent_color,
                center,
                radius
            )
            pygame.draw.circle(
                screen,
                (255, 255, 255),
                center,
                radius,
                2
            )

            
            agent_label = font_small.render(
                str(agent_index + 1),
                True,
                (255, 255, 255)
            )
            label_rect = agent_label.get_rect(center=center)
            screen.blit(agent_label, label_rect)

            
            battery_percentage = max(
                0.0,
                min(1.0, battery / MAX_BATTERY)
            )

            bar_width = self.cell_size - 2
            bar_height = 3
            bar_x = position[0] * self.cell_size + 1
            bar_y = max(0, position[1] * self.cell_size - 5)

            pygame.draw.rect(
                screen,
                (100, 100, 100),
                (bar_x, bar_y, bar_width, bar_height)
            )

            if battery_percentage > 0:
                if battery_percentage > 0.5:
                    battery_color = (0, 255, 0)
                elif battery_percentage > 0.2:
                    battery_color = (255, 255, 0)
                else:
                    battery_color = (255, 0, 0)

                pygame.draw.rect(
                    screen,
                    battery_color,
                    (
                        bar_x,
                        bar_y,
                        int(bar_width * battery_percentage),
                        bar_height
                    )
                )

        pygame.display.flip()

# Entity encoders and shared local DQN
class SetAttentionBlock(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(SetAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, entities, valid_mask):
        attention_output, _ = self.attention(
            entities,
            entities,
            entities,
            key_padding_mask=~valid_mask,
            need_weights=False
        )
        entities = self.norm1(entities + attention_output)
        entities = self.norm2(entities + self.feed_forward(entities))
        return entities * valid_mask.unsqueeze(-1).float()

class EntitySetTransformer(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, num_blocks):
        super(EntitySetTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.blocks = nn.ModuleList([
            SetAttentionBlock(embed_dim, num_heads)
            for _ in range(num_blocks)
        ])
        self.pool_seed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.pool_seed, std=0.02)
        self.pool_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.pool_norm1 = nn.LayerNorm(embed_dim)
        self.pool_feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.pool_norm2 = nn.LayerNorm(embed_dim)
        self.null_entity = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, entities, valid_mask):
        leading_shape = entities.shape[:-2]
        set_size = entities.shape[-2]
        entities = entities.reshape(-1, set_size, entities.shape[-1])
        valid_mask = valid_mask.reshape(-1, set_size).bool()

        encoded = self.input_projection(entities)
        empty_sets = ~valid_mask.any(dim=1, keepdim=True)
        null_entities = self.null_entity.expand(encoded.shape[0], -1, -1)
        encoded = torch.cat([encoded, null_entities], dim=1)
        valid_mask = torch.cat([valid_mask, empty_sets], dim=1)

        for block in self.blocks:
            encoded = block(encoded, valid_mask)

        seed = self.pool_seed.expand(encoded.shape[0], -1, -1)
        pooled, _ = self.pool_attention(
            seed,
            encoded,
            encoded,
            key_padding_mask=~valid_mask,
            need_weights=False
        )
        pooled = self.pool_norm1(seed + pooled)
        pooled = self.pool_norm2(pooled + self.pool_feed_forward(pooled))
        pooled = pooled.squeeze(1)

        
        
        pooled = pooled * (~empty_sets).float()
        return pooled.reshape(*leading_shape, self.embed_dim)

class EntityTokenTransformer(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, num_blocks):
        super(EntityTokenTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.blocks = nn.ModuleList([
            SetAttentionBlock(embed_dim, num_heads)
            for _ in range(num_blocks)
        ])
        self.null_entity = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, entities, valid_mask):
        leading_shape = entities.shape[:-2]
        set_size = entities.shape[-2]
        entities = entities.reshape(-1, set_size, entities.shape[-1])
        valid_mask = valid_mask.reshape(-1, set_size).bool()
        encoded = self.input_projection(entities)
        empty_sets = ~valid_mask.any(dim=1, keepdim=True)
        encoded = torch.cat([
            encoded,
            self.null_entity.expand(encoded.shape[0], -1, -1)
        ], dim=1)
        safe_mask = torch.cat([valid_mask, empty_sets], dim=1)
        for block in self.blocks:
            encoded = block(encoded, safe_mask)
        return (
            encoded.reshape(*leading_shape, set_size + 1, self.embed_dim),
            safe_mask.reshape(*leading_shape, set_size + 1),
        )

class ActionEntityCrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, edge_dim):
        super(ActionEntityCrossAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_heads)
        )
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, queries, entities, valid_mask, edge_features,
                return_attention=False):
        batch_size, action_count, _ = queries.shape
        entity_count = entities.shape[1]
        query = self.query_projection(queries).reshape(
            batch_size, action_count, self.num_heads, self.head_dim
        )
        key = self.key_projection(entities).reshape(
            batch_size, entity_count, self.num_heads, self.head_dim
        )
        value = self.value_projection(entities).reshape(
            batch_size, entity_count, self.num_heads, self.head_dim
        )
        scores = torch.einsum('bahd,bphd->bhap', query, key)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(edge_features).permute(0, 3, 1, 2)
        scores = scores.masked_fill(
            ~valid_mask.unsqueeze(1).unsqueeze(1),
            torch.finfo(scores.dtype).min
        )
        attention = torch.softmax(scores, dim=-1)
        context = torch.einsum('bhap,bphd->bahd', attention, value)
        context = context.reshape(batch_size, action_count, self.embed_dim)
        context = self.norm(
            queries + self.output_projection(context)
        )
        if return_attention:
            return context, attention.mean(dim=1)
        return context, None

class SharedLocalQNetwork(nn.Module):

    def __init__(self, action_dim):
        super(SharedLocalQNetwork, self).__init__()
        if action_dim != ACTION_DIM:
            raise ValueError(f'Expected action_dim={ACTION_DIM}, got {action_dim}')
        self.action_dim = action_dim
        self.agent_id_embedding = nn.Embedding(
            NUM_AGENTS, AGENT_ID_EMBED_DIM
        )
        self.action_embedding = nn.Embedding(action_dim, ENTITY_EMBED_DIM)
        self.register_buffer('action_offsets', torch.tensor([
            [0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0],
            [0.0, 0.0], [0.0, 0.0],
        ]) / GRID_SIZE)
        grid_center = LOCAL_GRID_RADIUS
        self.register_buffer('candidate_rows', torch.tensor([
            grid_center - 1, grid_center + 1, grid_center, grid_center,
            grid_center, grid_center,
        ]))
        self.register_buffer('candidate_cols', torch.tensor([
            grid_center, grid_center, grid_center - 1, grid_center + 1,
            grid_center, grid_center,
        ]))

        
        
        
        
        
        
        
        
        corridor_masks = torch.zeros(
            (ACTION_DIM, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE),
            dtype=torch.float32,
        )
        for action, (dx, dy) in enumerate(MOVEMENT_OFFSETS):
            for distance in range(1, LOCAL_GRID_RADIUS + 1):
                center_row = grid_center + dy * distance
                center_col = grid_center + dx * distance
                for lateral in (-1, 0, 1):
                    row = center_row + (lateral if dx else 0)
                    col = center_col + (lateral if dy else 0)
                    if (0 <= row < LOCAL_GRID_SIZE
                            and 0 <= col < LOCAL_GRID_SIZE):
                        corridor_masks[action, row, col] = 1.0
        corridor_masks[ACTION_HOVER, grid_center, grid_center] = 1.0
        corridor_masks[ACTION_LAND, grid_center, grid_center] = 1.0
        self.register_buffer('corridor_masks', corridor_masks)
        self.register_buffer(
            'corridor_counts',
            corridor_masks.sum(dim=(1, 2)).clamp(min=1.0),
        )

        self.self_encoder = nn.Sequential(
            nn.Linear(DRONE_STATE_DIM + 5 + MISSION_STATE_DIM, SELF_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(SELF_EMBED_DIM, SELF_EMBED_DIM),
            nn.ReLU(),
            nn.LayerNorm(SELF_EMBED_DIM)
        )
        self.patient_transformer = EntityTokenTransformer(
            13, ENTITY_EMBED_DIM, ATTENTION_HEADS, SET_ATTENTION_BLOCKS
        )
        self.drone_transformer = EntityTokenTransformer(
            24, ENTITY_EMBED_DIM, ATTENTION_HEADS, SET_ATTENTION_BLOCKS
        )
        self.patient_action_attention = ActionEntityCrossAttention(
            ENTITY_EMBED_DIM, ATTENTION_HEADS, 14
        )
        self.drone_action_attention = ActionEntityCrossAttention(
            ENTITY_EMBED_DIM, ATTENTION_HEADS, 9
        )
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                32 * LOCAL_GRID_SIZE * LOCAL_GRID_SIZE,
                GRID_EMBED_DIM,
            ),
            nn.ReLU()
        )
        base_dim = SELF_EMBED_DIM + GRID_EMBED_DIM + AGENT_ID_EMBED_DIM
        self.query_state_projection = nn.Linear(base_dim, ENTITY_EMBED_DIM)
        self.action_feature_encoder = nn.Sequential(
            nn.Linear(12, ENTITY_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(ENTITY_EMBED_DIM, ENTITY_EMBED_DIM)
        )
        
        
        
        
        self.navigation_feature_encoder = nn.Sequential(
            nn.Linear(4, ENTITY_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(ENTITY_EMBED_DIM, ENTITY_EMBED_DIM),
        )
        self.coordination_feature_encoder = nn.Sequential(
            nn.Linear(6, ENTITY_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(ENTITY_EMBED_DIM, ENTITY_EMBED_DIM),
        )
        self.rescue_action_fusion = nn.Sequential(
            nn.Linear(3 * ENTITY_EMBED_DIM, 192),
            nn.ReLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.ReLU()
        )
        self.landing_action_fusion = nn.Sequential(
            nn.Linear(2 * ENTITY_EMBED_DIM, 192),
            nn.ReLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.ReLU()
        )
        self.rescue_value_head = nn.Sequential(
            nn.Linear(base_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.rescue_advantage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.landing_value_head = nn.Sequential(
            nn.Linear(base_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.landing_advantage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, observation, return_diagnostics=False):
        drones = observation['drones'].float()
        patients = observation['patients'].float()
        pending_patient_masks = observation['pending_patient_masks'].bool()
        local_grids = observation['local_grids'].float()
        mission = observation['mission'].float()
        action_masks = observation['action_masks'].bool()

        batch_size, num_agents, _ = drones.shape
        num_patients = patients.shape[1]
        drone_x = drones[:, :, 0]
        drone_y = drones[:, :, 1]
        landing_dx = drones[:, :, 5] - drone_x
        landing_dy = drones[:, :, 6] - drone_y
        landing_euclidean = torch.sqrt(
            landing_dx.square() + landing_dy.square() + 1e-8
        )
        landing_manhattan = landing_dx.abs() + landing_dy.abs()
        
        
        
        battery_reserve = drones[:, :, 18]
        mission_per_agent = mission.unsqueeze(1).expand(-1, num_agents, -1)
        self_features = torch.cat([
            drones,
            landing_dx.unsqueeze(-1),
            landing_dy.unsqueeze(-1),
            landing_euclidean.unsqueeze(-1),
            landing_manhattan.unsqueeze(-1),
            battery_reserve.unsqueeze(-1),
            mission_per_agent,
        ], dim=-1)
        self_embedding = self.self_encoder(self_features)

        grid_embedding = self.grid_encoder(
            local_grids.reshape(
                batch_size * num_agents,
                3,
                LOCAL_GRID_SIZE,
                LOCAL_GRID_SIZE,
            )
        ).reshape(batch_size, num_agents, GRID_EMBED_DIM)
        agent_ids = torch.arange(
            num_agents, device=drones.device
        ).unsqueeze(0).expand(batch_size, -1)
        agent_id_embedding = self.agent_id_embedding(agent_ids)
        base_embedding = torch.cat([
            self_embedding, grid_embedding, agent_id_embedding
        ], dim=-1)

        action_ids = torch.arange(
            self.action_dim, device=drones.device
        ).view(1, 1, self.action_dim).expand(
            batch_size, num_agents, -1
        )
        action_offsets = self.action_offsets.view(
            1, 1, self.action_dim, 2
        ).expand(batch_size, num_agents, -1, -1)
        candidate_grid = local_grids[
            :, :, :, self.candidate_rows, self.candidate_cols
        ].permute(0, 1, 3, 2)
        directional_grid = torch.einsum(
            'bnchw,ahw->bnac', local_grids, self.corridor_masks
        ) / self.corridor_counts.view(1, 1, self.action_dim, 1)
        landing_after_dx = landing_dx.unsqueeze(-1) - action_offsets[..., 0]
        landing_after_dy = landing_dy.unsqueeze(-1) - action_offsets[..., 1]
        landing_after_manhattan = (
            landing_after_dx.abs() + landing_after_dy.abs()
        )
        is_hover = (action_ids == ACTION_HOVER).float()
        is_land = (action_ids == ACTION_LAND).float()
        action_features = torch.cat([
            action_offsets,
            is_hover.unsqueeze(-1),
            is_land.unsqueeze(-1),
            landing_after_manhattan.unsqueeze(-1),
            action_masks.float().unsqueeze(-1),
            candidate_grid,
            directional_grid,
        ], dim=-1)
        action_queries = (
            self.query_state_projection(base_embedding).unsqueeze(2)
            + self.action_embedding(action_ids)
            + self.action_feature_encoder(action_features)
        )

        patient_x = patients[:, :, 0].unsqueeze(1).expand(-1, num_agents, -1)
        patient_y = patients[:, :, 1].unsqueeze(1).expand(-1, num_agents, -1)
        patient_dx = patient_x - drone_x.unsqueeze(-1)
        patient_dy = patient_y - drone_y.unsqueeze(-1)
        patient_euclidean = torch.sqrt(
            patient_dx.square() + patient_dy.square() + 1e-8
        )
        patient_manhattan = patient_dx.abs() + patient_dy.abs()
        timer = patients[:, :, 2].unsqueeze(1).expand(-1, num_agents, -1)
        current_weight = patients[:, :, 3].unsqueeze(1).expand(
            -1, num_agents, -1
        )
        initial_weight = patients[:, :, 4].unsqueeze(1).expand(
            -1, num_agents, -1
        )
        response_age = patients[:, :, 9].unsqueeze(1).expand(
            -1, num_agents, -1
        )
        travel_fraction = patient_manhattan * GRID_SIZE / MAX_PATIENT_TIMER
        slack = timer - travel_fraction
        reachable = (slack >= 0.0).float()
        urgency = current_weight * (1.0 + response_age)
        patient_entities = torch.stack([
            patient_dx, patient_dy, patient_euclidean, patient_manhattan,
            timer, current_weight, initial_weight, 1.0 - timer,
            response_age, travel_fraction, slack, reachable, urgency,
        ], dim=-1)
        patient_mask_per_agent = pending_patient_masks.unsqueeze(1).expand(
            -1, num_agents, num_patients
        )
        patient_tokens, safe_patient_mask = self.patient_transformer(
            patient_entities, patient_mask_per_agent
        )

        patient_dx_after = (
            patient_dx.unsqueeze(2) - action_offsets[..., 0].unsqueeze(-1)
        )
        patient_dy_after = (
            patient_dy.unsqueeze(2) - action_offsets[..., 1].unsqueeze(-1)
        )
        patient_euclidean_after = torch.sqrt(
            patient_dx_after.square() + patient_dy_after.square() + 1e-8
        )
        patient_manhattan_after = (
            patient_dx_after.abs() + patient_dy_after.abs()
        )
        slack_after = (
            timer.unsqueeze(2)
            - patient_manhattan_after * GRID_SIZE / MAX_PATIENT_TIMER
        )
        patient_to_landing = (
            (
                drones[:, :, 5].unsqueeze(-1) - patient_x
            ).abs()
            + (
                drones[:, :, 6].unsqueeze(-1) - patient_y
            ).abs()
        )
        rescue_to_pad_distance = (
            patient_manhattan_after
            + patient_to_landing.unsqueeze(2)
        )
        rescue_energy_required = (
            rescue_to_pad_distance
            * GRID_SIZE
            * BATTERY_DRAIN_PER_STEP
            * RETURN_ENERGY_RISK_MULTIPLIER
            + SAFE_RETURN_BATTERY_BUFFER
        ) / MAX_BATTERY
        rescue_energy_margin = torch.clamp(
            drones[:, :, 2].unsqueeze(-1).unsqueeze(-1)
            - rescue_energy_required,
            min=-1.0,
            max=1.0,
        )
        patient_edges = torch.stack([
            patient_dx_after,
            patient_dy_after,
            patient_euclidean_after,
            patient_manhattan_after,
            patient_manhattan.unsqueeze(2) - patient_manhattan_after,
            timer.unsqueeze(2).expand(-1, -1, self.action_dim, -1),
            current_weight.unsqueeze(2).expand(-1, -1, self.action_dim, -1),
            urgency.unsqueeze(2).expand(-1, -1, self.action_dim, -1),
            slack_after,
            (slack_after >= 0.0).float(),
            response_age.unsqueeze(2).expand(-1, -1, self.action_dim, -1),
            rescue_energy_required,
            rescue_energy_margin,
            (rescue_energy_margin >= 0.0).float(),
        ], dim=-1)
        patient_edges = torch.cat([
            patient_edges,
            torch.zeros_like(patient_edges[..., :1, :])
        ], dim=-2)

        pending_per_agent = pending_patient_masks.unsqueeze(1).expand(
            -1, num_agents, -1
        )
        has_pending = pending_per_agent.any(dim=-1)
        masked_patient_distance = patient_manhattan.masked_fill(
            ~pending_per_agent, 2.0
        )
        nearest_patient_distance = masked_patient_distance.min(dim=-1).values
        pending_per_action = pending_per_agent.unsqueeze(2).expand(
            -1, -1, self.action_dim, -1
        )
        nearest_patient_after = patient_manhattan_after.masked_fill(
            ~pending_per_action, 2.0
        ).min(dim=-1).values
        nearest_patient_distance = torch.where(
            has_pending,
            nearest_patient_distance,
            torch.zeros_like(nearest_patient_distance),
        )
        nearest_patient_after = torch.where(
            has_pending.unsqueeze(-1),
            nearest_patient_after,
            torch.zeros_like(nearest_patient_after),
        )
        nearest_patient_progress_cells = (
            nearest_patient_distance.unsqueeze(-1) - nearest_patient_after
        ) * GRID_SIZE
        landing_progress_cells = (
            landing_manhattan.unsqueeze(-1) - landing_after_manhattan
        ) * GRID_SIZE
        navigation_features = torch.stack([
            nearest_patient_after,
            nearest_patient_progress_cells,
            landing_after_manhattan,
            landing_progress_cells,
        ], dim=-1)
        action_queries = (
            action_queries
            + self.navigation_feature_encoder(navigation_features)
        )

        other_x = drone_x.unsqueeze(1).expand(-1, num_agents, -1)
        other_y = drone_y.unsqueeze(1).expand(-1, num_agents, -1)
        other_dx = other_x - drone_x.unsqueeze(2)
        other_dy = other_y - drone_y.unsqueeze(2)
        other_distance = torch.sqrt(
            other_dx.square() + other_dy.square() + 1e-8
        )
        normalized_closeness = CLOSENESS_RADIUS / GRID_SIZE
        other_indices = torch.arange(
            num_agents, device=drones.device
        ).view(1, 1, num_agents).expand(batch_size, num_agents, -1)
        ego_indices = torch.arange(
            num_agents, device=drones.device
        ).view(1, num_agents, 1).expand(batch_size, -1, num_agents)
        other_role = other_indices.float() / max(1, num_agents - 1)
        role_delta = (
            (other_indices - ego_indices).float() / max(1, num_agents - 1)
        )
        other_previous_actions = drones[:, :, 11:11 + ACTION_DIM].unsqueeze(1).expand(
            -1, num_agents, -1, -1
        )
        other_cross_layer = drones[:, :, 17:22].unsqueeze(1).expand(
            -1, num_agents, -1, -1
        )
        other_entities = torch.cat([
            torch.stack([
                other_dx,
                other_dy,
                other_distance,
                drones[:, :, 2].unsqueeze(1).expand(-1, num_agents, -1),
                drones[:, :, 3].unsqueeze(1).expand(-1, num_agents, -1),
                drones[:, :, 4].unsqueeze(1).expand(-1, num_agents, -1),
                drones[:, :, 7].unsqueeze(1).expand(-1, num_agents, -1),
                drones[:, :, 8].unsqueeze(1).expand(-1, num_agents, -1),
                drones[:, :, 9].unsqueeze(1).expand(-1, num_agents, -1),
                drones[:, :, 10].unsqueeze(1).expand(-1, num_agents, -1),
                (other_distance <= normalized_closeness).float(),
                other_role,
                role_delta,
            ], dim=-1),
            other_previous_actions,
            other_cross_layer,
        ], dim=-1)
        self_mask = torch.eye(
            num_agents, device=drones.device, dtype=torch.bool
        ).unsqueeze(0)
        other_mask = (~self_mask).expand(batch_size, -1, -1)
        drone_tokens, safe_drone_mask = self.drone_transformer(
            other_entities, other_mask
        )

        candidate_x = drone_x.unsqueeze(-1) + action_offsets[..., 0]
        candidate_y = drone_y.unsqueeze(-1) + action_offsets[..., 1]
        other_dx_after = (
            other_x.unsqueeze(2) - candidate_x.unsqueeze(-1)
        )
        other_dy_after = (
            other_y.unsqueeze(2) - candidate_y.unsqueeze(-1)
        )
        other_distance_after = torch.sqrt(
            other_dx_after.square() + other_dy_after.square() + 1e-8
        )
        other_manhattan_after = other_dx_after.abs() + other_dy_after.abs()
        predicted_other_x = (
            other_x + drones[:, :, 7].unsqueeze(1).expand(-1, num_agents, -1)
            / GRID_SIZE
        )
        predicted_other_y = (
            other_y + drones[:, :, 8].unsqueeze(1).expand(-1, num_agents, -1)
            / GRID_SIZE
        )
        predicted_dx = predicted_other_x.unsqueeze(2) - candidate_x.unsqueeze(-1)
        predicted_dy = predicted_other_y.unsqueeze(2) - candidate_y.unsqueeze(-1)
        predicted_distance = torch.sqrt(
            predicted_dx.square() + predicted_dy.square() + 1e-8
        )
        same_predicted_destination = (
            (predicted_dx.abs() < 0.5 / GRID_SIZE)
            & (predicted_dy.abs() < 0.5 / GRID_SIZE)
        ).float()
        head_on_prediction = (
            (other_dx_after.abs() < 0.5 / GRID_SIZE)
            & (other_dy_after.abs() < 0.5 / GRID_SIZE)
            & ((predicted_other_x.unsqueeze(2) - drone_x.unsqueeze(-1).unsqueeze(-1)).abs()
               < 0.5 / GRID_SIZE)
            & ((predicted_other_y.unsqueeze(2) - drone_y.unsqueeze(-1).unsqueeze(-1)).abs()
               < 0.5 / GRID_SIZE)
        ).float()
        active_other_mask = (
            other_mask
            & ~drones[:, :, 3].bool().unsqueeze(1)
            & ~drones[:, :, 4].bool().unsqueeze(1)
        )
        active_other_per_action = active_other_mask.unsqueeze(2).expand(
            -1, -1, self.action_dim, -1
        )
        active_other_count = active_other_per_action.float().sum(
            dim=-1
        ).clamp(min=1.0)
        has_active_other = active_other_per_action.any(dim=-1)
        minimum_other_distance_after = other_distance_after.masked_fill(
            ~active_other_per_action, 2.0
        ).min(dim=-1).values
        maximum_approach_cells = (
            other_distance.unsqueeze(2) - other_distance_after
        ).masked_fill(~active_other_per_action, -2.0).max(dim=-1).values * GRID_SIZE
        minimum_other_distance_after = torch.where(
            has_active_other,
            minimum_other_distance_after,
            torch.zeros_like(minimum_other_distance_after),
        )
        maximum_approach_cells = torch.where(
            has_active_other,
            maximum_approach_cells,
            torch.zeros_like(maximum_approach_cells),
        )
        candidate_occupied = (
            (other_dx_after.abs() < 0.5 / GRID_SIZE)
            & (other_dy_after.abs() < 0.5 / GRID_SIZE)
        ).float()
        ego_yields_to_other = (
            ego_indices > other_indices
        ).float().unsqueeze(2).expand(-1, -1, self.action_dim, -1)
        coordination_features = torch.stack([
            minimum_other_distance_after,
            maximum_approach_cells,
            (candidate_occupied * active_other_per_action.float()).sum(dim=-1)
            / active_other_count,
            (same_predicted_destination * active_other_per_action.float()).sum(
                dim=-1
            ) / active_other_count,
            (head_on_prediction * active_other_per_action.float()).sum(dim=-1)
            / active_other_count,
            (
                same_predicted_destination * ego_yields_to_other
                * active_other_per_action.float()
            ).sum(dim=-1) / active_other_count,
        ], dim=-1)
        action_queries = (
            action_queries
            + self.coordination_feature_encoder(coordination_features)
        )
        drone_edges = torch.stack([
            other_dx_after,
            other_dy_after,
            other_distance_after,
            other_manhattan_after,
            other_distance.unsqueeze(2) - other_distance_after,
            predicted_distance,
            same_predicted_destination,
            head_on_prediction,
            role_delta.unsqueeze(2).expand(-1, -1, self.action_dim, -1),
        ], dim=-1)
        drone_edges = torch.cat([
            drone_edges,
            torch.zeros_like(drone_edges[..., :1, :])
        ], dim=-2)

        flat_count = batch_size * num_agents
        flat_queries = action_queries.reshape(
            flat_count, self.action_dim, ENTITY_EMBED_DIM
        )
        patient_context, patient_attention = self.patient_action_attention(
            flat_queries,
            patient_tokens.reshape(flat_count, num_patients + 1, ENTITY_EMBED_DIM),
            safe_patient_mask.reshape(flat_count, num_patients + 1),
            patient_edges.reshape(
                flat_count, self.action_dim, num_patients + 1, 14
            ),
            return_attention=return_diagnostics
        )
        drone_context, drone_attention = self.drone_action_attention(
            flat_queries,
            drone_tokens.reshape(flat_count, num_agents + 1, ENTITY_EMBED_DIM),
            safe_drone_mask.reshape(flat_count, num_agents + 1),
            drone_edges.reshape(
                flat_count, self.action_dim, num_agents + 1, 9
            ),
            return_attention=return_diagnostics
        )
        fp32_head_context = (
            torch.autocast(device_type='cuda', enabled=False)
            if drones.device.type == 'cuda' else nullcontext()
        )
        with fp32_head_context:
            rescue_action_latent = self.rescue_action_fusion(torch.cat([
                flat_queries, patient_context, drone_context
            ], dim=-1).float()).reshape(
                batch_size, num_agents, self.action_dim, 128
            )
            landing_action_latent = self.landing_action_fusion(torch.cat([
                flat_queries, drone_context
            ], dim=-1).float()).reshape(
                batch_size, num_agents, self.action_dim, 128
            )

            
            
            
            valid_action_count = action_masks.sum(
                dim=-1, keepdim=True
            ).clamp(min=1).float()
            float_action_masks = action_masks.float()

            def phase_q_values(value_head, advantage_head, phase_latent):
                advantage = advantage_head(phase_latent).squeeze(-1)
                valid_advantage_mean = (
                    (advantage * float_action_masks).sum(
                        dim=-1, keepdim=True
                    ) / valid_action_count
                )
                return (
                    value_head(base_embedding.float())
                    + advantage - valid_advantage_mean
                )

            rescue_q_values = phase_q_values(
                self.rescue_value_head,
                self.rescue_advantage_head,
                rescue_action_latent,
            )
            landing_q_values = phase_q_values(
                self.landing_value_head,
                self.landing_advantage_head,
                landing_action_latent,
            )
            
            
            
            
            
            
            global_landing_phase = mission[:, 8].reshape(
                batch_size, 1
            ).float()
            agent_return_phase = drones[:, :, -1].float()
            landing_phase = torch.maximum(
                global_landing_phase, agent_return_phase
            ).unsqueeze(-1)
            q_values = (
                (1.0 - landing_phase) * rescue_q_values
                + landing_phase * landing_q_values
            )
            action_latent = (
                (1.0 - landing_phase.unsqueeze(-1)) * rescue_action_latent
                + landing_phase.unsqueeze(-1) * landing_action_latent
            )

        if not return_diagnostics:
            return q_values
        normalized_queries = torch.nn.functional.normalize(
            action_latent, dim=-1
        )
        query_similarity = torch.matmul(
            normalized_queries, normalized_queries.transpose(-1, -2)
        )
        off_diagonal = ~torch.eye(
            self.action_dim, device=drones.device, dtype=torch.bool
        )
        diagnostics = {
            'patient_attention': patient_attention.reshape(
                batch_size, num_agents, self.action_dim, num_patients + 1
            ),
            'drone_attention': drone_attention.reshape(
                batch_size, num_agents, self.action_dim, num_agents + 1
            ),
            'patient_attention_mask': safe_patient_mask,
            'drone_attention_mask': safe_drone_mask,
            'action_query_similarity': query_similarity[..., off_diagonal].mean(),
            'action_latent_norm': action_latent.norm(dim=-1).mean(),
            'rescue_q_values': rescue_q_values,
            'landing_q_values': landing_q_values,
            'q_value_dtype': str(q_values.dtype),
        }
        return q_values, diagnostics

class CentralStateEncoder(nn.Module):

    def __init__(self):
        super(CentralStateEncoder, self).__init__()
        self.agent_id_embedding = nn.Embedding(
            NUM_AGENTS, AGENT_ID_EMBED_DIM
        )
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                8 * LOCAL_GRID_SIZE * LOCAL_GRID_SIZE,
                CENTRAL_GRID_EMBED_DIM,
            ),
            nn.ReLU()
        )
        agent_entity_dim = (
            DRONE_STATE_DIM + CENTRAL_GRID_EMBED_DIM + AGENT_ID_EMBED_DIM
        )
        self.drone_transformer = EntitySetTransformer(
            agent_entity_dim,
            ENTITY_EMBED_DIM,
            ATTENTION_HEADS,
            SET_ATTENTION_BLOCKS
        )
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_entity_dim, ENTITY_EMBED_DIM),
            nn.ReLU(),
            nn.LayerNorm(ENTITY_EMBED_DIM)
        )
        self.patient_transformer = EntitySetTransformer(
            PATIENT_STATE_DIM,
            ENTITY_EMBED_DIM,
            ATTENTION_HEADS,
            SET_ATTENTION_BLOCKS
        )
        self.mission_encoder = nn.Sequential(
            nn.Linear(MISSION_STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(2 * ENTITY_EMBED_DIM + 64, GLOBAL_EMBED_DIM),
            nn.ReLU(),
            nn.LayerNorm(GLOBAL_EMBED_DIM)
        )
        self.agent_output = nn.Sequential(
            nn.Linear(
                ENTITY_EMBED_DIM + GLOBAL_EMBED_DIM,
                AGENT_MIX_CONTEXT_DIM
            ),
            nn.ReLU(),
            nn.LayerNorm(AGENT_MIX_CONTEXT_DIM)
        )

    def forward(self, observation):
        drones = observation['drones'].float()
        patients = observation['patients'].float()
        patient_masks = observation['patient_masks'].bool()
        local_grids = observation['local_grids'].float()
        mission = observation['mission'].float()

        batch_size, num_agents, _ = drones.shape
        grid_embedding = self.grid_encoder(
            local_grids.reshape(
                batch_size * num_agents,
                3,
                LOCAL_GRID_SIZE,
                LOCAL_GRID_SIZE,
            )
        ).reshape(batch_size, num_agents, CENTRAL_GRID_EMBED_DIM)
        agent_ids = torch.arange(
            num_agents, device=drones.device
        ).unsqueeze(0).expand(batch_size, -1)
        agent_id_embedding = self.agent_id_embedding(agent_ids)
        agent_entities = torch.cat([
            drones, grid_embedding, agent_id_embedding
        ], dim=-1)

        drone_mask = torch.ones(
            (batch_size, num_agents), device=drones.device, dtype=torch.bool
        )
        drone_context = self.drone_transformer(agent_entities, drone_mask)
        individual_agent_context = self.agent_encoder(agent_entities)
        patient_context = self.patient_transformer(patients, patient_masks)
        mission_context = self.mission_encoder(mission)
        global_context = self.output(torch.cat([
            drone_context,
            patient_context,
            mission_context,
        ], dim=-1))
        agent_context = self.agent_output(torch.cat([
            individual_agent_context,
            global_context.unsqueeze(1).expand(-1, num_agents, -1),
        ], dim=-1))
        return global_context, agent_context

class QMixer(nn.Module):

    def __init__(self):
        super(QMixer, self).__init__()
        self.state_encoder = CentralStateEncoder()
        self.hyper_w1 = nn.Sequential(
            nn.Linear(AGENT_MIX_CONTEXT_DIM, AGENT_MIX_CONTEXT_DIM),
            nn.ReLU(),
            nn.Linear(AGENT_MIX_CONTEXT_DIM, MIXER_EMBED_DIM)
        )
        self.hyper_b1 = nn.Linear(GLOBAL_EMBED_DIM, MIXER_EMBED_DIM)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(GLOBAL_EMBED_DIM, GLOBAL_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(GLOBAL_EMBED_DIM, MIXER_EMBED_DIM)
        )
        self.state_value = nn.Sequential(
            nn.Linear(GLOBAL_EMBED_DIM, MIXER_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(MIXER_EMBED_DIM, 1)
        )

    def forward(self, agent_utilities, observation, agent_mask,
                return_diagnostics=False):
        batch_size = agent_utilities.shape[0]
        global_embedding, agent_context = self.state_encoder(observation)
        utilities = (agent_utilities * agent_mask).view(
            batch_size, 1, NUM_AGENTS
        )

        active_count = agent_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        weights1 = (
            MIXER_MIN_RAW_WEIGHT
            + (2.0 - MIXER_MIN_RAW_WEIGHT)
            * torch.sigmoid(self.hyper_w1(agent_context))
        ) / active_count.unsqueeze(-1)
        bias1 = self.hyper_b1(global_embedding).view(
            batch_size, 1, MIXER_EMBED_DIM
        )
        pre_hidden = torch.bmm(utilities, weights1) + bias1
        hidden = torch.nn.functional.elu(pre_hidden)
        weights2 = (
            MIXER_MIN_RAW_WEIGHT
            + (2.0 - MIXER_MIN_RAW_WEIGHT)
            * torch.sigmoid(self.hyper_w2(global_embedding))
        ) / MIXER_EMBED_DIM
        weights2 = weights2.view(batch_size, MIXER_EMBED_DIM, 1)
        state_value = self.state_value(global_embedding).view(batch_size, 1, 1)
        mixed_utility = torch.bmm(hidden, weights2)
        q_total = (mixed_utility + state_value).view(batch_size, 1)
        if not return_diagnostics:
            return q_total

        elu_derivative = torch.where(
            pre_hidden > 0.0, torch.ones_like(pre_hidden), torch.exp(pre_hidden)
        )
        effective_sensitivity = torch.bmm(
            weights1,
            elu_derivative.transpose(1, 2) * weights2
        ).squeeze(-1)
        return q_total, {
            'state_value': state_value.view(batch_size, 1),
            'mixed_utility': mixed_utility.view(batch_size, 1),
            'mixer_weight_mean': torch.cat([
                weights1.reshape(batch_size, -1),
                weights2.reshape(batch_size, -1),
            ], dim=1).mean(dim=1),
            'mixer_weight_min': torch.minimum(
                weights1.amin(dim=(1, 2)), weights2.amin(dim=(1, 2))
            ),
            'mixer_weight_max': torch.maximum(
                weights1.amax(dim=(1, 2)), weights2.amax(dim=(1, 2))
            ),
            'utility_sensitivity': effective_sensitivity,
        }

# Prioritized joint replay buffer
class PrioritySumTree:

    def __init__(self, capacity):
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self.capacity = capacity
        self.tree_capacity = tree_capacity
        self.tree = np.zeros(2 * tree_capacity, dtype=np.float64)

    def update(self, data_index, priority):
        tree_index = self.tree_capacity + data_index
        change = priority - self.tree[tree_index]
        while tree_index >= 1:
            self.tree[tree_index] += change
            tree_index //= 2

    def total(self):
        return self.tree[1]

    def get(self, value):
        tree_index = 1
        while tree_index < self.tree_capacity:
            left = tree_index * 2
            
            
            
            if value < self.tree[left]:
                tree_index = left
            else:
                value -= self.tree[left]
                tree_index = left + 1
        data_index = tree_index - self.tree_capacity
        return data_index, self.tree[tree_index]

    def get_many(self, values):

        values = np.asarray(values, dtype=np.float64).copy()
        tree_indices = np.ones(values.shape, dtype=np.int64)
        while np.any(tree_indices < self.tree_capacity):
            left = tree_indices * 2
            left_totals = self.tree[left]
            choose_left = values < left_totals
            values = np.where(choose_left, values, values - left_totals)
            tree_indices = np.where(choose_left, left, left + 1)
        data_indices = tree_indices - self.tree_capacity
        return data_indices, self.tree[tree_indices]

    def update_many(self, data_indices, priorities):

        data_indices = np.asarray(data_indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float64)
        if data_indices.size == 0:
            return
        
        
        unique_indices, inverse = np.unique(data_indices, return_inverse=True)
        unique_priorities = np.zeros(unique_indices.shape, dtype=np.float64)
        np.maximum.at(unique_priorities, inverse, priorities)
        tree_indices = self.tree_capacity + unique_indices
        self.tree[tree_indices] = unique_priorities
        while tree_indices.size and tree_indices[0] > 1:
            parents = np.unique(tree_indices // 2)
            self.tree[parents] = (
                self.tree[parents * 2] + self.tree[parents * 2 + 1]
            )
            tree_indices = parents

class PrioritizedJointReplayBuffer:

    def __init__(self, capacity, gamma, n_step):
        replay_fraction_total = (
            REPLAY_UNIFORM_FRACTION
            + REPLAY_RESCUE_FRACTION
            + REPLAY_LANDING_FRACTION
        )
        if not math.isclose(replay_fraction_total, 1.0, abs_tol=1e-8):
            raise ValueError(
                'Replay source fractions must sum to one'
            )
        self.capacity = capacity
        self.gamma = gamma
        self.n_step = n_step
        self.size = 0
        self.position = 0
        self.max_priority = 1.0
        self.stage_trees = {
            stage_index: {
                'uniform': PrioritySumTree(capacity),
                'rescue': PrioritySumTree(capacity),
                'landing': PrioritySumTree(capacity),
            }
            for stage_index in range(len(CURRICULUM_STAGES))
        }
        self.n_step_queues = {}
        self.landing_phase_count = 0
        self.event_counts = {
            event_flag: 0 for event_flag in EVENT_PRIORITY_MULTIPLIERS
        }

        self.drones = np.empty(
            (capacity, NUM_AGENTS, DRONE_STATE_DIM), dtype=np.float16
        )
        self.patients = np.empty(
            (capacity, MAX_PATIENTS, PATIENT_STATE_DIM), dtype=np.float16
        )
        self.patient_masks = np.empty(
            (capacity, MAX_PATIENTS), dtype=np.uint8
        )
        self.pending_patient_masks = np.empty_like(self.patient_masks)
        self.local_grids = np.empty(
            (
                capacity,
                NUM_AGENTS,
                3,
                LOCAL_GRID_SIZE,
                LOCAL_GRID_SIZE,
            ),
            dtype=np.uint8,
        )
        self.missions = np.empty(
            (capacity, MISSION_STATE_DIM), dtype=np.float16
        )
        self.action_masks = np.empty(
            (capacity, NUM_AGENTS, ACTION_DIM), dtype=np.uint8
        )

        self.next_drones = np.empty_like(self.drones)
        self.next_patients = np.empty_like(self.patients)
        self.next_patient_masks = np.empty_like(self.patient_masks)
        self.next_pending_patient_masks = np.empty_like(
            self.pending_patient_masks
        )
        self.next_local_grids = np.empty_like(self.local_grids)
        self.next_missions = np.empty_like(self.missions)
        self.next_action_masks = np.empty_like(self.action_masks)

        self.actions = np.empty((capacity, NUM_AGENTS), dtype=np.int8)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.local_rewards = np.empty(
            (capacity, NUM_AGENTS), dtype=np.float32
        )
        self.dones = np.empty(capacity, dtype=np.uint8)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.event_flags = np.zeros(capacity, dtype=np.uint8)
        self.landing_phase_flags = np.zeros(capacity, dtype=np.uint8)
        self.curriculum_stages = np.full(capacity, -1, dtype=np.int8)
        self.scaled_priorities = np.zeros(capacity, dtype=np.float64)
        self.transition_ids = np.full(capacity, -1, dtype=np.int64)
        self.next_transition_id = 0

    def allocated_mb(self):
        arrays = [
            self.drones, self.patients, self.patient_masks,
            self.pending_patient_masks,
            self.local_grids, self.missions, self.action_masks,
            self.next_drones, self.next_patients, self.next_patient_masks,
            self.next_pending_patient_masks,
            self.next_local_grids, self.next_missions,
            self.next_action_masks, self.actions, self.rewards,
            self.local_rewards, self.dones, self.discounts,
            self.event_flags, self.landing_phase_flags,
            self.curriculum_stages, self.scaled_priorities,
            self.transition_ids,
        ]
        arrays.extend(
            tree.tree
            for stage_trees in self.stage_trees.values()
            for tree in stage_trees.values()
        )
        return sum(array.nbytes for array in arrays) / (1024.0 ** 2)

    def copy_observation(self, observation):
        return {
            key: np.array(value, copy=True)
            for key, value in observation.items()
        }

    def push(self, state, actions, reward, local_rewards, event_flags,
             next_state, done, curriculum_stage=None, stream_id=0):
        if curriculum_stage is None:
            curriculum_stage = len(CURRICULUM_STAGES) - 1
        curriculum_stage = int(curriculum_stage)
        if not 0 <= curriculum_stage < len(CURRICULUM_STAGES):
            raise ValueError(f'Invalid replay curriculum stage {curriculum_stage}')
        transition = (
            self.copy_observation(state),
            np.asarray(actions, dtype=np.int8).copy(),
            float(reward),
            np.asarray(local_rewards, dtype=np.float32).copy(),
            int(event_flags),
            self.copy_observation(next_state),
            bool(done),
            curriculum_stage,
        )
        queue = self.n_step_queues.setdefault(int(stream_id), deque())
        queue.append(transition)

        if done:
            while queue:
                self._store_from_n_step_queue(queue)
                queue.popleft()
            self.n_step_queues.pop(int(stream_id), None)
        elif len(queue) >= self.n_step:
            self._store_from_n_step_queue(queue)
            queue.popleft()

    def _store_from_n_step_queue(self, queue):
        state, actions, _, _, _, _, _, curriculum_stage = queue[0]
        n_step_reward = 0.0
        n_step_local_rewards = np.zeros(NUM_AGENTS, dtype=np.float32)
        n_step_event_flags = 0
        final_next_state = queue[0][5]
        final_done = False
        steps_used = 0

        for step_index, transition in enumerate(queue):
            (_, _, reward, local_rewards, event_flags,
             next_state, done, transition_stage) = transition
            if transition_stage != curriculum_stage:
                raise RuntimeError('N-step replay stream crossed curriculum stages')
            n_step_reward += (self.gamma ** step_index) * reward
            n_step_local_rewards += (
                (self.gamma ** step_index) * local_rewards
            )
            n_step_event_flags |= event_flags
            final_next_state = next_state
            final_done = done
            steps_used += 1
            if done or steps_used >= self.n_step:
                break

        index = self.position
        self._write_observation(index, state, next_state=False)
        self._write_observation(index, final_next_state, next_state=True)
        self.actions[index] = actions
        self.rewards[index] = n_step_reward
        self.local_rewards[index] = n_step_local_rewards
        self.dones[index] = final_done
        self.discounts[index] = self.gamma ** steps_used
        if self.size == self.capacity:
            overwritten_flags = int(self.event_flags[index])
            for event_flag in self.event_counts:
                if overwritten_flags & event_flag:
                    self.event_counts[event_flag] -= 1
            self.landing_phase_count -= int(
                self.landing_phase_flags[index]
            )
            overwritten_stage = int(self.curriculum_stages[index])
            if overwritten_stage >= 0:
                for tree in self.stage_trees[overwritten_stage].values():
                    tree.update(index, 0.0)
        self.event_flags[index] = n_step_event_flags
        for event_flag in self.event_counts:
            if n_step_event_flags & event_flag:
                self.event_counts[event_flag] += 1
        
        
        
        
        is_landing_phase = int(
            float(state['mission'][8]) >= 0.5
            or np.any(state['drones'][:, -1] >= 0.5)
            or bool(n_step_event_flags & EVENT_LANDING)
        )
        self.landing_phase_flags[index] = is_landing_phase
        self.landing_phase_count += is_landing_phase
        self.curriculum_stages[index] = curriculum_stage
        self.transition_ids[index] = self.next_transition_id
        self.next_transition_id += 1

        event_multiplier = self.event_priority_multiplier(n_step_event_flags)
        initial_priority = min(
            PER_PRIORITY_MAX, max(1.0, self.max_priority) * event_multiplier
        )
        scaled_priority = initial_priority ** PER_ALPHA
        self.scaled_priorities[index] = scaled_priority
        stage_trees = self.stage_trees[curriculum_stage]
        stage_trees['uniform'].update(index, 1.0)
        stage_trees['rescue'].update(
            index, scaled_priority if not is_landing_phase else 0.0
        )
        stage_trees['landing'].update(
            index, scaled_priority if is_landing_phase else 0.0
        )
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _write_observation(self, index, observation, next_state):
        prefix = 'next_' if next_state else ''
        getattr(self, f'{prefix}drones')[index] = observation['drones']
        getattr(self, f'{prefix}patients')[index] = observation['patients']
        getattr(self, f'{prefix}patient_masks')[index] = observation['patient_masks']
        getattr(self, f'{prefix}pending_patient_masks')[index] = (
            observation['pending_patient_masks']
        )
        getattr(self, f'{prefix}local_grids')[index] = observation['local_grids']
        getattr(self, f'{prefix}missions')[index] = observation['mission']
        getattr(self, f'{prefix}action_masks')[index] = observation['action_masks']

    def sample(self, batch_size, beta, device, current_stage=None):
        if self.size < batch_size:
            raise RuntimeError('Replay does not contain a complete minibatch')
        if current_stage is None:
            current_stage = len(CURRICULUM_STAGES) - 1
        current_stage = int(current_stage)
        if current_stage == len(CURRICULUM_STAGES) - 1:
            desired_stage_weights = {current_stage: 1.0}
        elif current_stage == 0:
            desired_stage_weights = {
                0: CURRICULUM_CURRENT_PROBABILITY +
                   CURRICULUM_PREVIOUS_PROBABILITY,
                len(CURRICULUM_STAGES) - 1: CURRICULUM_FULL_PROBABILITY,
            }
        else:
            desired_stage_weights = {
                current_stage: CURRICULUM_CURRENT_PROBABILITY,
                current_stage - 1: CURRICULUM_PREVIOUS_PROBABILITY,
                len(CURRICULUM_STAGES) - 1: CURRICULUM_FULL_PROBABILITY,
            }
        stage_weights = {
            stage: weight
            for stage, weight in desired_stage_weights.items()
            if weight > 0.0 and self.stage_trees[stage]['uniform'].total() > 0.0
        }
        if not stage_weights:
            stage_weights = {
                stage: trees['uniform'].total()
                for stage, trees in self.stage_trees.items()
                if trees['uniform'].total() > 0.0
            }
        weight_total = sum(stage_weights.values())
        stage_weights = {
            stage: weight / weight_total
            for stage, weight in stage_weights.items()
        }

        def allocate_counts(total_count, fractions):
            keys = list(fractions)
            exact = np.asarray(
                [total_count * fractions[key] for key in keys],
                dtype=np.float64
            )
            counts = np.floor(exact).astype(np.int64)
            remainder = int(total_count - counts.sum())
            if remainder:
                order = np.argsort(-(exact - counts), kind='stable')
                counts[order[:remainder]] += 1
            return {key: int(count) for key, count in zip(keys, counts)}

        stage_counts = allocate_counts(batch_size, stage_weights)
        sampled_indices = []
        sampled_sources = []
        sampled_stages = []
        proposal_counts = {}
        source_weights = {
            'uniform': REPLAY_UNIFORM_FRACTION,
            'rescue': REPLAY_RESCUE_FRACTION,
            'landing': REPLAY_LANDING_FRACTION,
        }
        source_ids = {'uniform': 0, 'rescue': 1, 'landing': 2}
        for stage, stage_count in stage_counts.items():
            if stage_count <= 0:
                continue
            trees = self.stage_trees[stage]
            available_sources = {
                source: weight
                for source, weight in source_weights.items()
                if weight > 0.0 and trees[source].total() > 0.0
            }
            available_total = sum(available_sources.values())
            available_sources = {
                source: weight / available_total
                for source, weight in available_sources.items()
            }
            source_counts = allocate_counts(stage_count, available_sources)
            for source, source_count in source_counts.items():
                if source_count <= 0:
                    continue
                tree = trees[source]
                segment = tree.total() / source_count
                values = segment * (
                    np.arange(source_count, dtype=np.float64)
                    + np.random.random(source_count)
                )
                source_indices, _ = tree.get_many(values)
                sampled_indices.append(source_indices)
                sampled_sources.append(np.full(
                    source_count, source_ids[source], dtype=np.int8
                ))
                sampled_stages.append(np.full(
                    source_count, stage, dtype=np.int8
                ))
                proposal_counts[(stage, source)] = source_count

        indices = np.concatenate(sampled_indices)
        sampling_sources = np.concatenate(sampled_sources)
        sampled_stage_ids = np.concatenate(sampled_stages)
        if indices.size != batch_size:
            raise RuntimeError(
                f'Stage replay sampled {indices.size}/{batch_size} transitions'
            )
        shuffle_order = np.random.permutation(batch_size)
        indices = indices[shuffle_order]
        sampling_sources = sampling_sources[shuffle_order]
        sampled_stage_ids = sampled_stage_ids[shuffle_order]

        probabilities = np.zeros(batch_size, dtype=np.float64)
        for stage in np.unique(sampled_stage_ids):
            stage_mask = sampled_stage_ids == stage
            stage_indices = indices[stage_mask]
            trees = self.stage_trees[int(stage)]
            stage_probabilities = np.zeros(stage_indices.size, dtype=np.float64)
            for source in source_weights:
                source_count = proposal_counts.get((int(stage), source), 0)
                if source_count <= 0:
                    continue
                tree = trees[source]
                leaves = tree.tree[tree.tree_capacity + stage_indices]
                stage_probabilities += (
                    source_count / batch_size
                ) * leaves / tree.total()
            probabilities[stage_mask] = stage_probabilities
        if np.any(probabilities <= 0.0):
            raise RuntimeError('Replay produced a zero-probability sample')
        priorities = self.scaled_priorities[indices]
        weights = (self.size * probabilities) ** (-beta)
        weights /= max(weights.max(), 1e-8)

        def tensor(array, dtype):
            
            
            sample = array[indices]
            return torch.as_tensor(sample, dtype=dtype, device=device)

        observations = {
            'drones': tensor(self.drones, torch.float32),
            'patients': tensor(self.patients, torch.float32),
            'patient_masks': tensor(self.patient_masks, torch.bool),
            'pending_patient_masks': tensor(
                self.pending_patient_masks, torch.bool
            ),
            'local_grids': tensor(self.local_grids, torch.float32),
            'mission': tensor(self.missions, torch.float32),
            'action_masks': tensor(self.action_masks, torch.bool),
        }
        next_observations = {
            'drones': tensor(self.next_drones, torch.float32),
            'patients': tensor(self.next_patients, torch.float32),
            'patient_masks': tensor(self.next_patient_masks, torch.bool),
            'pending_patient_masks': tensor(
                self.next_pending_patient_masks, torch.bool
            ),
            'local_grids': tensor(self.next_local_grids, torch.float32),
            'mission': tensor(self.next_missions, torch.float32),
            'action_masks': tensor(self.next_action_masks, torch.bool),
        }
        return {
            'indices': indices,
            'observations': observations,
            'actions': tensor(self.actions, torch.long),
            'rewards': tensor(self.rewards, torch.float32).unsqueeze(1),
            'local_rewards': tensor(self.local_rewards, torch.float32),
            'next_observations': next_observations,
            'dones': tensor(self.dones, torch.float32).unsqueeze(1),
            'discounts': tensor(self.discounts, torch.float32).unsqueeze(1),
            'weights': torch.as_tensor(
                weights, dtype=torch.float32, device=device
            ).unsqueeze(1),
            'sampling_probabilities': torch.as_tensor(
                probabilities, dtype=torch.float32, device=device
            ).unsqueeze(1),
            'sample_priorities': torch.as_tensor(
                np.power(priorities, 1.0 / PER_ALPHA),
                dtype=torch.float32,
                device=device
            ).unsqueeze(1),
            'event_flags': tensor(self.event_flags, torch.long),
            'landing_phase_flags': tensor(
                self.landing_phase_flags, torch.bool
            ),
            'curriculum_stages': tensor(
                self.curriculum_stages, torch.long
            ),
            'sampling_sources': torch.as_tensor(
                sampling_sources, dtype=torch.long, device=device
            ),
            'sampling_uniform_fraction': float(
                np.mean(sampling_sources == 0)
            ),
            'sampling_rescue_fraction': float(
                np.mean(sampling_sources == 1)
            ),
            'sampling_landing_fraction': float(
                np.mean(sampling_sources == 2)
            ),
            'sample_ages': torch.as_tensor(
                (self.next_transition_id - 1) - self.transition_ids[indices],
                dtype=torch.float32,
                device=device
            ),
        }

    def event_priority_multiplier(self, event_flags):
        multiplier = 1.0
        for event_flag, event_multiplier in EVENT_PRIORITY_MULTIPLIERS.items():
            if event_flags & event_flag:
                multiplier = max(multiplier, event_multiplier)
        return multiplier

    def event_fraction(self, event_flag):
        return self.event_counts[event_flag] / max(1, self.size)

    def landing_phase_fraction(self):
        return self.landing_phase_count / max(1, self.size)

    def stage_fraction(self, stage_index):
        if self.size == 0:
            return 0.0
        return float(np.mean(
            self.curriculum_stages[:self.size] == int(stage_index)
        ))

    def update_priorities(self, indices, td_errors):
        indices = np.asarray(indices, dtype=np.int64)
        td_errors = np.asarray(td_errors, dtype=np.float32)
        event_multipliers = np.fromiter((
            self.event_priority_multiplier(int(event_flags))
            for event_flags in self.event_flags[indices]
        ), dtype=np.float64, count=indices.size)
        priorities = np.minimum(
            PER_PRIORITY_MAX,
            (np.abs(td_errors) + PER_EPSILON) * event_multipliers
        )
        if priorities.size:
            self.max_priority = max(
                self.max_priority, float(priorities.max())
            )
        scaled_priorities = priorities ** PER_ALPHA
        self.scaled_priorities[indices] = scaled_priorities
        for stage in np.unique(self.curriculum_stages[indices]):
            stage_indices = indices[self.curriculum_stages[indices] == stage]
            stage_priorities = scaled_priorities[
                self.curriculum_stages[indices] == stage
            ]
            landing_mask = self.landing_phase_flags[stage_indices].astype(bool)
            stage_trees = self.stage_trees[int(stage)]
            stage_trees['rescue'].update_many(
                stage_indices,
                np.where(landing_mask, 0.0, stage_priorities)
            )
            stage_trees['landing'].update_many(
                stage_indices,
                np.where(landing_mask, stage_priorities, 0.0)
            )

    def __len__(self):
        return self.size

# QMIX centralized training and decentralized execution
class CTDEAgent:

    def __init__(self, action_dim, lr, gamma, device, mixed_precision=True):
        if action_dim != ACTION_DIM:
            raise ValueError(f'Expected action_dim={ACTION_DIM}, got {action_dim}')
        self.device = device
        self.num_agents = NUM_AGENTS
        self.action_dim = action_dim
        self.gamma = gamma
        self.use_mixed_precision = (
            mixed_precision
            and device.type == 'cuda'
            and getattr(torch.cuda, 'is_bf16_supported', lambda: False)()
        )

        self.policy_net = SharedLocalQNetwork(action_dim).to(self.device)
        self.target_net = SharedLocalQNetwork(action_dim).to(self.device)
        self.mixer = QMixer().to(self.device)
        self.target_mixer = QMixer().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_net.eval()
        self.target_mixer.eval()

        self.online_parameters = (
            list(self.policy_net.parameters()) + list(self.mixer.parameters())
        )
        self.uses_fused_optimizer = False
        if device.type == 'cuda':
            try:
                self.optimizer = optim.Adam(
                    self.online_parameters, lr=lr, eps=1e-5, fused=True
                )
                self.uses_fused_optimizer = True
            except (TypeError, RuntimeError):
                self.optimizer = optim.Adam(
                    self.online_parameters, lr=lr, eps=1e-5
                )
        else:
            self.optimizer = optim.Adam(
                self.online_parameters, lr=lr, eps=1e-5
            )
        self.joint_buffer = PrioritizedJointReplayBuffer(
            BUFFER_CAPACITY, gamma, N_STEP
        )
        self.learner_steps = 0

        
        self.metric_updates = 0
        self.loss_sum = 0.0
        self.q_total_sum = 0.0
        self.local_q_sum = 0.0
        self.gradient_norm_sum = 0.0
        self.absolute_td_error_sum = 0.0
        self.importance_weight_sum = 0.0
        self.gradient_clipped_updates = 0
        self.extra_metric_sums = {}
        self.extra_metric_counts = {}

    def autocast_context(self):

        if self.use_mixed_precision:
            return torch.autocast(
                device_type='cuda', dtype=torch.bfloat16
            )
        return nullcontext()

    def _to_device(self, observation):
        return {
            key: torch.as_tensor(
                value,
                dtype=torch.bool if key in {
                    'patient_masks', 'pending_patient_masks', 'action_masks'
                }
                else torch.float32,
                device=self.device
            ).unsqueeze(0)
            for key, value in observation.items()
        }

    @staticmethod
    def attention_statistics(attention, valid_mask):
        valid = valid_mask.unsqueeze(2).expand_as(attention)
        safe_attention = attention.float() * valid.float()
        entropy = -(
            safe_attention
            * safe_attention.clamp(min=1e-8).log()
        ).sum(dim=-1)
        valid_count = valid_mask.sum(dim=-1).clamp(min=1).unsqueeze(-1)
        normalizer = valid_count.float().log().clamp(min=1.0)
        normalized_entropy = entropy / normalizer
        maximum, top_index = safe_attention.max(dim=-1)
        return normalized_entropy, maximum, top_index

    def compact_policy_diagnostics(self, q_values, network_diagnostics):
        masked_q = q_values.masked_fill(
            ~network_diagnostics['action_masks'], -1e9
        )
        top_values = masked_q.topk(k=2, dim=-1).values
        valid_count = network_diagnostics['action_masks'].sum(dim=-1)
        action_gap = torch.where(
            valid_count > 1,
            top_values[..., 0] - top_values[..., 1],
            torch.zeros_like(top_values[..., 0])
        )
        top_tie_count = (
            (q_values == top_values[..., :1])
            & network_diagnostics['action_masks']
        ).sum(dim=-1)
        exact_top_tie = top_tie_count > 1
        patient_entropy, patient_max, patient_top = self.attention_statistics(
            network_diagnostics['patient_attention'],
            network_diagnostics['patient_attention_mask']
        )
        drone_entropy, drone_max, drone_top = self.attention_statistics(
            network_diagnostics['drone_attention'],
            network_diagnostics['drone_attention_mask']
        )
        patient_attention = network_diagnostics[
            'patient_attention'
        ][..., :MAX_PATIENTS].float()
        patient_initial_weights = network_diagnostics[
            'patient_initial_weights'
        ].unsqueeze(1).unsqueeze(1)
        patient_response_ages = network_diagnostics[
            'patient_response_ages'
        ].unsqueeze(1).unsqueeze(1)
        pending_patient_attention_mask = network_diagnostics[
            'patient_attention_mask'
        ][..., :MAX_PATIENTS].unsqueeze(2)
        attention_mass_by_weight = {
            weight: float((
                patient_attention
                * (patient_initial_weights == weight).float()
            ).sum(dim=-1).mean().item())
            for weight in (1, 2, 3)
        }
        attention_enrichment_by_weight = {}
        for weight in (1, 2, 3):
            class_mask = (
                (patient_initial_weights == weight)
                & pending_patient_attention_mask
            )
            class_prevalence = (
                class_mask.float().sum(dim=-1)
                / pending_patient_attention_mask.float().sum(
                    dim=-1
                ).clamp(min=1.0)
            )
            class_mass = (
                patient_attention * class_mask.float()
            ).sum(dim=-1)
            class_prevalence = class_prevalence.expand_as(class_mass)
            class_present = class_prevalence > 0.0
            attention_enrichment_by_weight[weight] = float(
                (class_mass[class_present]
                 / class_prevalence[class_present]).mean().item()
                if class_present.any() else 0.0
            )
        return {
            'q_values': q_values[0].float().cpu().tolist(),
            'action_gap': action_gap[0].float().cpu().tolist(),
            'exact_top_tie': exact_top_tie[0].cpu().tolist(),
            'top_tie_count': top_tie_count[0].cpu().tolist(),
            'exact_q_tie_fraction': float(
                exact_top_tie.float().mean().item()
            ),
            'q_value_dtype': str(q_values.dtype),
            'patient_attention_entropy': float(patient_entropy.mean().item()),
            'patient_attention_max': float(patient_max.mean().item()),
            'patient_top_indices': patient_top[0].cpu().tolist(),
            'patient_top_weights': patient_max[0].float().cpu().tolist(),
            'patient_attention_mass_by_initial_weight': (
                attention_mass_by_weight
            ),
            'patient_attention_enrichment_by_initial_weight': (
                attention_enrichment_by_weight
            ),
            'patient_attention_weighted_response_age': float((
                patient_attention * patient_response_ages
            ).sum(dim=-1).mean().item()),
            'drone_attention_entropy': float(drone_entropy.mean().item()),
            'drone_attention_max': float(drone_max.mean().item()),
            'drone_top_indices': drone_top[0].cpu().tolist(),
            'drone_top_weights': drone_max[0].float().cpu().tolist(),
            'action_query_similarity': float(
                network_diagnostics['action_query_similarity'].item()
            ),
            'action_latent_norm': float(
                network_diagnostics['action_latent_norm'].item()
            ),
        }

    def select_actions(self, state, epsilon, return_diagnostics=False):

        if not np.all(state['action_masks'].any(axis=1)):
            raise ValueError('Every agent must have at least one valid action')
        observation = self._to_device(state)
        with torch.inference_mode():
            with self.autocast_context():
                if return_diagnostics:
                    q_values_batch, network_diagnostics = self.policy_net(
                        observation, return_diagnostics=True
                    )
                    network_diagnostics['action_masks'] = observation[
                        'action_masks'
                    ]
                    network_diagnostics['patient_initial_weights'] = (
                        observation['patients'][:, :, 4]
                        * MAX_PATIENT_WEIGHT
                    ).round().long()
                    network_diagnostics['patient_response_ages'] = (
                        observation['patients'][:, :, 9]
                    )
                else:
                    q_values_batch = self.policy_net(observation)

            
            
            
            q_values_batch = q_values_batch.float()
            q_values = q_values_batch[0]
            action_mask = observation['action_masks'][0]
            masked_q_values = q_values.masked_fill(~action_mask, -1e9)
            actions = masked_q_values.argmax(dim=-1).cpu().numpy()

        agent_epsilon = min(1.0, max(0.0, float(epsilon)))
        action_masks = state['action_masks'].astype(bool)
        for agent_index in range(self.num_agents):
            if random.random() >= agent_epsilon:
                continue
            valid_actions = np.flatnonzero(action_masks[agent_index])
            actions[agent_index] = int(random.choice(valid_actions.tolist()))
        action_list = actions.tolist()
        if return_diagnostics:
            return action_list, self.compact_policy_diagnostics(
                q_values_batch, network_diagnostics
            )
        return action_list

    def select_actions_batch(self, states, epsilon, return_diagnostics=False):

        if not states:
            return []
        observations = {
            key: torch.as_tensor(
                np.stack([state[key] for state in states]),
                dtype=torch.bool if key in {
                    'patient_masks', 'pending_patient_masks', 'action_masks'
                } else torch.float32,
                device=self.device,
            )
            for key in states[0]
        }
        with torch.no_grad(), self.autocast_context():
            q_values = self.policy_net(observations).float()
        masked_q_values = q_values.masked_fill(
            ~observations['action_masks'], -1e9
        )
        unconstrained_actions = q_values.argmax(dim=-1).cpu().numpy()
        actions = masked_q_values.argmax(dim=-1).cpu().numpy()
        agent_epsilon = min(1.0, max(0.0, float(epsilon)))
        for environment_index, state in enumerate(states):
            for agent_index in range(self.num_agents):
                if random.random() >= agent_epsilon:
                    continue
                valid_actions = np.flatnonzero(
                    state['action_masks'][agent_index].astype(bool)
                )
                actions[environment_index, agent_index] = int(
                    random.choice(valid_actions.tolist())
                )
        if return_diagnostics:
            return actions.tolist(), {
                'unconstrained_actions': unconstrained_actions.tolist(),
            }
        return actions.tolist()

    def push(self, state, actions, team_reward, local_rewards, event_flags,
             next_state, done, curriculum_stage=None, stream_id=0):
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.shape != (self.num_agents,):
            raise ValueError(
                f'actions must have shape {(self.num_agents,)}, '
                f'got {action_array.shape}'
            )
        self.joint_buffer.push(
            state,
            action_array,
            team_reward * TD_REWARD_SCALE,
            np.asarray(local_rewards, dtype=np.float32) * TD_REWARD_SCALE,
            event_flags,
            next_state,
            done,
            curriculum_stage=curriculum_stage,
            stream_id=stream_id,
        )

    def beta_at_step(self, global_step):
        progress = min(1.0, global_step / max(1, EPSILON_END_STEP))
        return PER_BETA_START + progress * (
            PER_BETA_END - PER_BETA_START
        )

    def train_step(self, batch_size, global_step, current_stage=None):
        if len(self.joint_buffer) < max(batch_size, REPLAY_WARMUP):
            return None

        beta = self.beta_at_step(global_step)
        batch = self.joint_buffer.sample(
            batch_size, beta, self.device, current_stage=current_stage
        )
        observations = batch['observations']
        next_observations = batch['next_observations']
        actions = batch['actions']

        collect_diagnostics = (
            (self.learner_steps + 1) % ATTENTION_DIAGNOSTIC_INTERVAL == 0
        )
        with self.autocast_context():
            if collect_diagnostics:
                all_utilities, network_diagnostics = self.policy_net(
                    observations, return_diagnostics=True
                )
            else:
                all_utilities = self.policy_net(observations)
                network_diagnostics = None
            chosen_utilities = all_utilities.gather(
                2, actions.unsqueeze(-1)
            ).squeeze(-1)
            agent_mask = (
                (1.0 - observations['drones'][:, :, 3])
                * (1.0 - observations['drones'][:, :, 4])
            )
            if collect_diagnostics:
                q_total, mixer_diagnostics = self.mixer(
                    chosen_utilities,
                    observations,
                    agent_mask,
                    return_diagnostics=True
                )
            else:
                q_total = self.mixer(
                    chosen_utilities, observations, agent_mask
                )
                mixer_diagnostics = None

            with torch.no_grad():
                
                
                next_online_utilities = self.policy_net(next_observations)
                next_online_utilities = next_online_utilities.masked_fill(
                    ~next_observations['action_masks'], -1e9
                )
                next_actions = next_online_utilities.argmax(
                    dim=-1, keepdim=True
                )
                next_target_utilities = self.target_net(next_observations)
                next_chosen_utilities = next_target_utilities.gather(
                    2, next_actions
                ).squeeze(-1)
                next_agent_mask = (
                    (1.0 - next_observations['drones'][:, :, 3])
                    * (1.0 - next_observations['drones'][:, :, 4])
                )
                next_q_total = self.target_mixer(
                    next_chosen_utilities,
                    next_observations,
                    next_agent_mask
                )
                target = (
                    batch['rewards']
                    + batch['discounts'] * (1.0 - batch['dones']) * next_q_total
                )
                local_target = (
                    batch['local_rewards']
                    + batch['discounts']
                    * (1.0 - batch['dones'])
                    * next_chosen_utilities
                    * next_agent_mask
                )

            td_error = target.float() - q_total.float()
            element_loss = torch.nn.functional.smooth_l1_loss(
                q_total.float(), target.float(), reduction='none'
            )
            team_loss = (batch['weights'] * element_loss).mean()
            local_td_error = local_target.float() - chosen_utilities.float()
            local_element_loss = torch.nn.functional.smooth_l1_loss(
                chosen_utilities.float(), local_target.float(), reduction='none'
            )
            per_sample_local_loss = (
                (local_element_loss * agent_mask).sum(dim=1)
                / agent_mask.sum(dim=1).clamp(min=1.0)
            )
            landing_sample_mask = observations['mission'][:, 8] >= 0.5
            local_loss = (
                batch['weights'].squeeze(1)
                * per_sample_local_loss
            ).mean()
            loss = team_loss + LOCAL_TD_LOSS_WEIGHT * local_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if collect_diagnostics:
            module_gradient_norms = {
                'patient_encoder_gradient_norm': self.module_gradient_norm([
                    self.policy_net.patient_transformer,
                    self.policy_net.patient_action_attention,
                ]),
                'drone_encoder_gradient_norm': self.module_gradient_norm([
                    self.policy_net.drone_transformer,
                    self.policy_net.drone_action_attention,
                ]),
                'action_head_gradient_norm': self.module_gradient_norm([
                    self.policy_net.rescue_action_fusion,
                    self.policy_net.landing_action_fusion,
                    self.policy_net.navigation_feature_encoder,
                    self.policy_net.rescue_advantage_head,
                    self.policy_net.rescue_value_head,
                    self.policy_net.landing_advantage_head,
                    self.policy_net.landing_value_head,
                ]),
                'rescue_head_gradient_norm': self.module_gradient_norm([
                    self.policy_net.rescue_advantage_head,
                    self.policy_net.rescue_value_head,
                ]),
                'landing_head_gradient_norm': self.module_gradient_norm([
                    self.policy_net.landing_advantage_head,
                    self.policy_net.landing_value_head,
                ]),
                'mixer_gradient_norm': self.module_gradient_norm([
                    self.mixer
                ]),
                'mixer_state_value_gradient_norm': self.module_gradient_norm([
                    self.mixer.state_value
                ]),
            }
        else:
            module_gradient_norms = {}
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online_parameters, GRADIENT_CLIP
        )
        self.optimizer.step()

        self.joint_buffer.update_priorities(
            batch['indices'], td_error.detach().squeeze(1).cpu().numpy()
        )
        self.learner_steps += 1
        if self.learner_steps % TARGET_UPDATE_STEPS == 0:
            self.update_target()

        loss_value = float(loss.item())
        self.metric_updates += 1
        self.loss_sum += loss_value
        self.q_total_sum += float(q_total.detach().mean().item())
        self.local_q_sum += float(chosen_utilities.detach().mean().item())
        self.gradient_norm_sum += float(gradient_norm.item())
        self.absolute_td_error_sum += float(
            td_error.detach().abs().mean().item()
        )
        self.importance_weight_sum += float(batch['weights'].mean().item())
        self.gradient_clipped_updates += int(
            gradient_norm.item() > GRADIENT_CLIP
        )
        valid_q = all_utilities.detach().masked_fill(
            ~observations['action_masks'], -1e9
        )
        top_q = valid_q.topk(k=2, dim=-1).values
        valid_action_count = observations['action_masks'].sum(dim=-1)
        active_action_gap = torch.where(
            valid_action_count > 1,
            top_q[..., 0] - top_q[..., 1],
            torch.zeros_like(top_q[..., 0])
        )
        greedy_q = top_q[..., 0]
        exact_top_tie = (
            (all_utilities.detach() == top_q[..., :1])
            & observations['action_masks']
        ).sum(dim=-1) > 1
        rescue_sample_mask = ~landing_sample_mask
        self.accumulate_metric('team_td_loss', team_loss.item())
        self.accumulate_metric('local_td_loss', local_loss.item())
        self.accumulate_metric(
            'unweighted_team_loss', element_loss.mean().item()
        )
        self.accumulate_metric('signed_td_error', td_error.mean().item())
        self.accumulate_metric('td_error_std', td_error.std().item())
        self.accumulate_metric('td_error_max', td_error.abs().max().item())
        self.accumulate_metric(
            'local_absolute_td_error', local_td_error.abs().mean().item()
        )
        self.accumulate_metric('target_q_total', target.mean().item())
        self.accumulate_metric(
            'q_action_gap', active_action_gap[agent_mask.bool()].mean().item()
        )
        self.accumulate_metric(
            'exact_q_tie_fraction',
            exact_top_tie[agent_mask.bool()].float().mean().item()
        )
        self.accumulate_metric(
            'chosen_greedy_q_gap',
            (greedy_q - chosen_utilities.detach())[agent_mask.bool()].mean().item()
        )
        weights = batch['weights'].detach().squeeze(1)
        self.accumulate_metric('importance_weight_std', weights.std().item())
        self.accumulate_metric('importance_weight_min', weights.min().item())
        self.accumulate_metric('importance_weight_max', weights.max().item())
        effective_sample_fraction = (
            weights.sum().square() / weights.square().sum().clamp(min=1e-8)
            / weights.numel()
        )
        self.accumulate_metric(
            'importance_effective_sample_fraction',
            effective_sample_fraction.item()
        )
        self.accumulate_metric('sample_age_mean', batch['sample_ages'].mean().item())
        self.accumulate_metric('sample_age_max', batch['sample_ages'].max().item())
        self.accumulate_metric(
            'sample_priority_mean', batch['sample_priorities'].mean().item()
        )
        self.accumulate_metric(
            'sample_priority_max', batch['sample_priorities'].max().item()
        )
        for metric_name, event_flag in {
            'sample_delivery_fraction': EVENT_DELIVERY,
            'sample_collision_fraction': EVENT_COLLISION,
            'sample_progress_fraction': EVENT_PROGRESS,
            'sample_landing_fraction': EVENT_LANDING,
            'sample_terminal_fraction': EVENT_TERMINAL,
            'sample_landing_phase_event_fraction': EVENT_LANDING_PHASE,
            'sample_obstacle_collision_fraction': EVENT_OBSTACLE_COLLISION,
            'sample_hazard_fraction': EVENT_HAZARD,
        }.items():
            self.accumulate_metric(
                metric_name,
                ((batch['event_flags'] & event_flag) != 0).float().mean().item()
            )
        self.accumulate_metric(
            'sample_landing_phase_fraction',
            batch['landing_phase_flags'].float().mean().item()
        )
        self.accumulate_metric(
            'sample_uniform_source_fraction',
            (batch['sampling_sources'] == 0).float().mean().item()
        )
        self.accumulate_metric(
            'sample_rescue_per_source_fraction',
            (batch['sampling_sources'] == 1).float().mean().item()
        )
        self.accumulate_metric(
            'sample_landing_per_source_fraction',
            (batch['sampling_sources'] == 2).float().mean().item()
        )
        for stage_index in range(len(CURRICULUM_STAGES)):
            self.accumulate_metric(
                f'sample_curriculum_stage_{stage_index}_fraction',
                (batch['curriculum_stages'] == stage_index).float().mean().item()
            )

        
        
        per_sample_team_loss = element_loss.detach().squeeze(1)
        per_sample_local_error = (
            (local_td_error.detach().abs() * agent_mask).sum(dim=1)
            / agent_mask.sum(dim=1).clamp(min=1.0)
        )
        for phase_name, sample_mask in (
                ('rescue', rescue_sample_mask),
                ('landing', landing_sample_mask)):
            if not sample_mask.any():
                continue
            phase_agent_mask = agent_mask.bool() & sample_mask.unsqueeze(1)
            self.accumulate_metric(
                f'{phase_name}_team_td_loss',
                per_sample_team_loss[sample_mask].mean().item()
            )
            self.accumulate_metric(
                f'{phase_name}_local_absolute_td_error',
                per_sample_local_error[sample_mask].mean().item()
            )
            if phase_agent_mask.any():
                self.accumulate_metric(
                    f'{phase_name}_q_action_gap',
                    active_action_gap[phase_agent_mask].mean().item()
                )
                self.accumulate_metric(
                    f'{phase_name}_exact_q_tie_fraction',
                    exact_top_tie[phase_agent_mask].float().mean().item()
                )
        self.accumulate_metric(
            'target_network_age', self.learner_steps % TARGET_UPDATE_STEPS
        )

        if collect_diagnostics:
            patient_entropy, patient_max, _ = self.attention_statistics(
                network_diagnostics['patient_attention'],
                network_diagnostics['patient_attention_mask']
            )
            drone_entropy, drone_max, _ = self.attention_statistics(
                network_diagnostics['drone_attention'],
                network_diagnostics['drone_attention_mask']
            )
            diagnostic_values = {
                'patient_attention_entropy': patient_entropy.mean().item(),
                'patient_attention_max': patient_max.mean().item(),
                'drone_attention_entropy': drone_entropy.mean().item(),
                'drone_attention_max': drone_max.mean().item(),
                'action_query_similarity': network_diagnostics[
                    'action_query_similarity'
                ].item(),
                'action_latent_norm': network_diagnostics[
                    'action_latent_norm'
                ].item(),
                'mixer_state_value': mixer_diagnostics[
                    'state_value'
                ].mean().item(),
                'mixer_utility_contribution': mixer_diagnostics[
                    'mixed_utility'
                ].mean().item(),
                'mixer_weight_mean': mixer_diagnostics[
                    'mixer_weight_mean'
                ].mean().item(),
                'mixer_weight_min': mixer_diagnostics[
                    'mixer_weight_min'
                ].mean().item(),
                'mixer_weight_max': mixer_diagnostics[
                    'mixer_weight_max'
                ].mean().item(),
                'utility_sensitivity_mean': mixer_diagnostics[
                    'utility_sensitivity'
                ].mean().item(),
                'utility_sensitivity_min': mixer_diagnostics[
                    'utility_sensitivity'
                ].min().item(),
                'utility_sensitivity_max': mixer_diagnostics[
                    'utility_sensitivity'
                ].max().item(),
                **module_gradient_norms,
            }
            for metric_name, metric_value in diagnostic_values.items():
                self.accumulate_metric(metric_name, metric_value)
        return loss_value

    @staticmethod
    def module_gradient_norm(modules):
        squared_norm = 0.0
        for module in modules:
            for parameter in module.parameters():
                if parameter.grad is not None:
                    squared_norm += float(
                        parameter.grad.detach().float().square().sum().item()
                    )
        return math.sqrt(squared_norm)

    def accumulate_metric(self, name, value):
        if not math.isfinite(float(value)):
            raise RuntimeError(f'Non-finite learner metric {name}: {value}')
        self.extra_metric_sums[name] = (
            self.extra_metric_sums.get(name, 0.0) + float(value)
        )
        self.extra_metric_counts[name] = (
            self.extra_metric_counts.get(name, 0) + 1
        )

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def get_metrics(self):
        count = max(1, self.metric_updates)
        metrics = {
            'loss': self.loss_sum / count if self.metric_updates else 0.0,
            'q_total': self.q_total_sum / count if self.metric_updates else 0.0,
            'local_q': self.local_q_sum / count if self.metric_updates else 0.0,
            'gradient_norm': (
                self.gradient_norm_sum / count if self.metric_updates else 0.0
            ),
            'absolute_td_error': (
                self.absolute_td_error_sum / count
                if self.metric_updates else 0.0
            ),
            'mean_importance_weight': (
                self.importance_weight_sum / count
                if self.metric_updates else 0.0
            ),
            'gradient_clip_fraction': (
                self.gradient_clipped_updates / count
                if self.metric_updates else 0.0
            ),
            'replay_size': len(self.joint_buffer),
            'replay_max_priority': self.joint_buffer.max_priority,
            'buffer_delivery_fraction': self.joint_buffer.event_fraction(
                EVENT_DELIVERY
            ),
            'buffer_collision_fraction': self.joint_buffer.event_fraction(
                EVENT_COLLISION
            ),
            'buffer_progress_fraction': self.joint_buffer.event_fraction(
                EVENT_PROGRESS
            ),
            'buffer_landing_fraction': self.joint_buffer.event_fraction(
                EVENT_LANDING
            ),
            'buffer_terminal_fraction': self.joint_buffer.event_fraction(
                EVENT_TERMINAL
            ),
            'buffer_landing_phase_fraction': (
                self.joint_buffer.landing_phase_fraction()
            ),
            'buffer_obstacle_collision_fraction': (
                self.joint_buffer.event_fraction(EVENT_OBSTACLE_COLLISION)
            ),
            'buffer_hazard_fraction': self.joint_buffer.event_fraction(
                EVENT_HAZARD
            ),
            **{
                f'buffer_curriculum_stage_{stage_index}_fraction':
                    self.joint_buffer.stage_fraction(stage_index)
                for stage_index in range(len(CURRICULUM_STAGES))
            },
            **{
                metric_name: self.extra_metric_sums[metric_name]
                / max(1, self.extra_metric_counts[metric_name])
                for metric_name in self.extra_metric_sums
            },
        }
        self.metric_updates = 0
        self.loss_sum = 0.0
        self.q_total_sum = 0.0
        self.local_q_sum = 0.0
        self.gradient_norm_sum = 0.0
        self.absolute_td_error_sum = 0.0
        self.importance_weight_sum = 0.0
        self.gradient_clipped_updates = 0
        self.extra_metric_sums = {}
        self.extra_metric_counts = {}
        return metrics

# Parallel episode collection
class TrainingEpisodeTracker:

    def __init__(self, env, curriculum_stage, selection_probability):
        self.curriculum_stage = int(curriculum_stage)
        self.selection_probability = float(selection_probability)
        self.obstacles = set(env.obstacles)
        self.total_reward = 0.0
        self.steps = 0
        self.local_rewards = np.zeros(NUM_AGENTS, dtype=np.float64)
        self.local_potential_rewards = np.zeros(NUM_AGENTS, dtype=np.float64)
        self.wind_entries = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.wind_exposure_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.wind_exits = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.wind_failures = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.low_signal_entries = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.low_signal_exposure_steps = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        self.low_signal_exits = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.low_signal_failures = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.operational_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.movement_actions = 0
        self.wind_command_attempts = 0
        self.low_signal_command_attempts = 0
        self.wind_avoidance_opportunities = 0
        self.wind_hazard_selections = 0
        self.wind_dominated_hazard_selections = 0
        self.wind_shortcut_hazard_selections = 0
        self.low_signal_avoidance_opportunities = 0
        self.low_signal_hazard_selections = 0
        self.low_signal_dominated_hazard_selections = 0
        self.low_signal_shortcut_hazard_selections = 0
        self.wind_entry_progress_cells = 0.0
        self.low_signal_entry_progress_cells = 0.0
        self.wind_zone_refreshes = 0
        self.low_signal_zone_refreshes = 0
        self.wind_refresh_onsets = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.low_signal_refresh_onsets = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        self.deliveries_by_agent = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.obstacle_collisions_by_agent = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.agent_collisions_by_agent = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.action_counts = np.zeros(
            (NUM_AGENTS, ACTION_DIM), dtype=np.int64
        )
        self.unconstrained_action_counts = np.zeros(
            (NUM_AGENTS, ACTION_DIM), dtype=np.int64
        )
        self.unconstrained_obstacle_preferences = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        self.unconstrained_boundary_preferences = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        self.unconstrained_occupied_cell_preferences = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        self.phase_action_counts = {
            phase: np.zeros((NUM_AGENTS, ACTION_DIM), dtype=np.int64)
            for phase in ('rescue', 'landing', 'irrecoverable')
        }
        self.phase_steps = {'rescue': 0, 'landing': 0, 'irrecoverable': 0}
        self.reward_components = {
            name: 0.0 for name in REWARD_COMPONENT_NAMES
        }
        self.phase_reward_components = {
            phase: {name: 0.0 for name in REWARD_COMPONENT_NAMES}
            for phase in ('rescue', 'landing', 'irrecoverable')
        }
        self.obstacle_collisions = 0
        self.agent_collisions = 0
        self.same_destination_collisions = 0
        self.head_on_collisions = 0
        self.collision_steps = 0
        self.obstacle_collision_steps = 0
        self.rescue_collisions = 0
        self.landing_collisions = 0
        self.obstacle_opportunities = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.obstacle_actions_selected = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.dominated_obstacle_selections = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        self.dominated_agent_conflict_selections = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        self.maximum_collision_streak = 0
        self.maximum_agent_collision_streak = 0
        self.maximum_obstacle_collision_streak = 0
        self.minimum_agent_distance = 2 * GRID_SIZE
        self.valid_action_counts = []
        self.patient_events = []
        self.landing_events = []
        self.landing_zone_arrivals = []
        self.landing_zone_departures = []
        self.obstacle_collision_events = []
        self.battery_depletion_events = []
        self.dead_landing_events = []
        self.death_penalty_applications = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        self.death_reminder_penalty_applications = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        self.battery_drain = np.zeros(NUM_AGENTS, dtype=np.float64)
        self.wind_battery_drain = np.zeros(NUM_AGENTS, dtype=np.float64)
        self.landing_standby_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.energy_return_mode_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.energy_return_progress_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.energy_return_regress_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.energy_return_activations = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.previous_energy_return_flags = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        self.energy_margin_delta = np.zeros(NUM_AGENTS, dtype=np.float64)
        self.reserve_violation_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
        self.minimum_safe_return_margin = np.full(
            NUM_AGENTS, np.inf, dtype=np.float64
        )
        self.potential_deltas = []
        self.raw_potential_changes = []
        self.inference_latency_ms = []
        self.environment_latency_ms = []
        self.training_update_latency_ms = []
        self.initial_distances = list(env.curriculum_initial_distances)

    def record_step(self, state, actions, step_data, inference_latency_ms,
                    unconstrained_actions=None):
        self.steps += 1
        self.total_reward += float(step_data['team_reward'])
        self.inference_latency_ms.append(float(inference_latency_ms))
        self.environment_latency_ms.append(float(
            step_data['environment_step_latency_ms']
        ))
        self.valid_action_counts.extend(
            state['action_masks'].sum(axis=1).tolist()
        )
        for agent_index, action in enumerate(actions):
            self.action_counts[agent_index, action] += 1
            self.phase_action_counts[
                step_data['phase_before']
            ][agent_index, action] += 1
            if not state['drones'][agent_index, 3] \
                    and not state['drones'][agent_index, 4]:
                self.operational_steps[agent_index] += 1
        if unconstrained_actions is not None:
            positions = [
                (
                    int(round(drone[0] * GRID_SIZE)),
                    int(round(drone[1] * GRID_SIZE)),
                )
                for drone in state['drones']
            ]
            for agent_index, raw_action in enumerate(unconstrained_actions):
                self.unconstrained_action_counts[
                    agent_index, raw_action
                ] += 1
                if raw_action >= len(MOVEMENT_OFFSETS):
                    continue
                dx, dy = MOVEMENT_OFFSETS[raw_action]
                candidate = (
                    positions[agent_index][0] + dx,
                    positions[agent_index][1] + dy,
                )
                if candidate in self.obstacles:
                    self.unconstrained_obstacle_preferences[agent_index] += 1
                elif not (
                    0 <= candidate[0] < GRID_SIZE
                    and 0 <= candidate[1] < GRID_SIZE
                ):
                    self.unconstrained_boundary_preferences[agent_index] += 1
                elif candidate in {
                    position for other_index, position in enumerate(positions)
                    if other_index != agent_index
                }:
                    self.unconstrained_occupied_cell_preferences[
                        agent_index
                    ] += 1
        self.phase_steps[step_data['phase_before']] += 1
        self.local_rewards += np.asarray(
            step_data['local_rewards'], dtype=np.float64
        )
        self.local_potential_rewards += np.asarray(
            step_data['local_potential_rewards'], dtype=np.float64
        )
        self.wind_entries += np.asarray(
            step_data['wind_entries'], dtype=np.int64
        )
        self.wind_exposure_steps += np.asarray(
            step_data['wind_exposure_steps'], dtype=np.int64
        )
        self.wind_exits += np.asarray(
            step_data['wind_exits'], dtype=np.int64
        )
        self.wind_failures += np.asarray(
            step_data['wind_failures'], dtype=np.int64
        )
        step_battery_drain = np.asarray(
            step_data['battery_drain_by_agent'], dtype=np.float64
        )
        self.battery_drain += step_battery_drain
        self.wind_battery_drain += np.asarray(
            step_data['wind_battery_drain_by_agent'], dtype=np.float64
        )
        self.landing_standby_steps += np.asarray(
            step_data['landing_standby_steps'], dtype=np.int64
        )
        energy_return_flags = np.asarray(
            step_data['energy_return_mode_flags'], dtype=np.int64
        )
        self.energy_return_mode_steps += energy_return_flags
        self.energy_return_progress_steps += np.asarray(
            step_data['energy_return_progress_flags'], dtype=np.int64
        )
        self.energy_return_regress_steps += np.asarray(
            step_data['energy_return_regress_flags'], dtype=np.int64
        )
        self.energy_return_activations += (
            (energy_return_flags > 0)
            & (self.previous_energy_return_flags == 0)
        ).astype(np.int64)
        self.previous_energy_return_flags = energy_return_flags
        self.energy_margin_delta += np.asarray(
            step_data['energy_margin_delta_by_agent'], dtype=np.float64
        )
        self.reserve_violation_steps += np.asarray(
            step_data['reserve_violation_flags'], dtype=np.int64
        )
        operational_mask = step_battery_drain > 0.0
        safe_return_margins = np.asarray(
            step_data['safe_return_margin_after'], dtype=np.float64
        )
        self.minimum_safe_return_margin[operational_mask] = np.minimum(
            self.minimum_safe_return_margin[operational_mask],
            safe_return_margins[operational_mask],
        )
        self.low_signal_entries += np.asarray(
            step_data['low_signal_entries'], dtype=np.int64
        )
        self.low_signal_exposure_steps += np.asarray(
            step_data['low_signal_exposure_steps'], dtype=np.int64
        )
        self.low_signal_exits += np.asarray(
            step_data['low_signal_exits'], dtype=np.int64
        )
        self.low_signal_failures += np.asarray(
            step_data['low_signal_failures'], dtype=np.int64
        )
        self.movement_actions += int(step_data['movement_actions'])
        self.wind_command_attempts += int(step_data['wind_command_attempts'])
        self.low_signal_command_attempts += int(
            step_data['low_signal_command_attempts']
        )
        self.wind_avoidance_opportunities += int(
            step_data['wind_avoidance_opportunities']
        )
        self.wind_hazard_selections += int(
            step_data['wind_hazard_selections']
        )
        self.wind_dominated_hazard_selections += int(
            step_data['wind_dominated_hazard_selections']
        )
        self.wind_shortcut_hazard_selections += int(
            step_data['wind_shortcut_hazard_selections']
        )
        self.low_signal_avoidance_opportunities += int(
            step_data['low_signal_avoidance_opportunities']
        )
        self.low_signal_hazard_selections += int(
            step_data['low_signal_hazard_selections']
        )
        self.low_signal_dominated_hazard_selections += int(
            step_data['low_signal_dominated_hazard_selections']
        )
        self.low_signal_shortcut_hazard_selections += int(
            step_data['low_signal_shortcut_hazard_selections']
        )
        self.wind_entry_progress_cells += float(
            step_data['wind_entry_progress_cells']
        )
        self.low_signal_entry_progress_cells += float(
            step_data['low_signal_entry_progress_cells']
        )
        self.wind_zone_refreshes += int(step_data['wind_zone_refreshed'])
        self.low_signal_zone_refreshes += int(
            step_data['low_signal_zone_refreshed']
        )
        self.wind_refresh_onsets += np.asarray(
            step_data['wind_refresh_onset_agents'], dtype=np.int64
        )
        self.low_signal_refresh_onsets += np.asarray(
            step_data['low_signal_refresh_onset_agents'], dtype=np.int64
        )
        self.obstacle_collisions += int(step_data['obstacle_collisions'])
        self.agent_collisions += int(step_data['agent_collisions'])
        self.obstacle_collision_steps += int(
            step_data['obstacle_collisions'] > 0
        )
        self.collision_steps += int(step_data['agent_collisions'] > 0)
        self.same_destination_collisions += int(
            step_data['same_destination_collisions']
        )
        self.head_on_collisions += int(step_data['head_on_collisions'])
        self.obstacle_opportunities += np.asarray(
            step_data['obstacle_action_opportunities'], dtype=np.int64
        )
        self.obstacle_actions_selected += np.asarray(
            step_data['obstacle_action_selected'], dtype=np.int64
        )
        self.dominated_obstacle_selections += np.asarray(
            step_data['dominated_obstacle_selections'], dtype=np.int64
        )
        self.dominated_agent_conflict_selections += np.asarray(
            step_data['dominated_agent_conflict_selections'], dtype=np.int64
        )
        self.obstacle_collisions_by_agent += np.asarray(
            step_data['obstacle_collision_flags'], dtype=np.int64
        )
        self.agent_collisions_by_agent += np.asarray(
            step_data['agent_collision_flags'], dtype=np.int64
        )
        if step_data['phase_before'] == 'rescue':
            self.rescue_collisions += int(step_data['agent_collisions'])
        elif step_data['phase_before'] == 'landing':
            self.landing_collisions += int(step_data['agent_collisions'])
        self.maximum_collision_streak = max(
            self.maximum_collision_streak, step_data['max_collision_streak']
        )
        self.maximum_agent_collision_streak = max(
            self.maximum_agent_collision_streak,
            step_data['max_agent_collision_streak']
        )
        self.maximum_obstacle_collision_streak = max(
            self.maximum_obstacle_collision_streak,
            step_data['max_obstacle_collision_streak']
        )
        if step_data['minimum_agent_distance'] >= 0:
            self.minimum_agent_distance = min(
                self.minimum_agent_distance, step_data['minimum_agent_distance']
            )
        for event in step_data['patient_delivery_events']:
            self.deliveries_by_agent[event['agent']] += 1
            self.patient_events.append({'type': 'delivery', **event})
        self.patient_events.extend(
            {'type': 'death', **event}
            for event in step_data['patient_death_events']
        )
        self.patient_events.extend(
            {'type': 'triage_escalation', **event}
            for event in step_data['patient_weight_escalation_events']
        )
        self.patient_events.extend({
            'type': 'spawn', 'patient': patient_index,
            'step': self.steps,
        } for patient_index in step_data['patient_spawn_events'])
        self.landing_events.extend(step_data['landing_events'])
        self.landing_zone_arrivals.extend(
            step_data['landing_zone_arrival_events']
        )
        self.landing_zone_departures.extend(
            step_data['landing_zone_departure_events']
        )
        remaining_obstacle_event_slots = max(
            0,
            MAX_RECORDED_OBSTACLE_EVENTS_PER_EPISODE
            - len(self.obstacle_collision_events),
        )
        if remaining_obstacle_event_slots:
            self.obstacle_collision_events.extend(
                step_data['obstacle_collision_events'][
                    :remaining_obstacle_event_slots
                ]
            )
        self.battery_depletion_events.extend(
            step_data['battery_depletion_events']
        )
        self.dead_landing_events.extend(step_data['dead_landing_events'])
        self.death_penalty_applications += np.asarray(
            step_data['death_penalty_applications'], dtype=np.int64
        )
        self.death_reminder_penalty_applications += np.asarray(
            step_data['death_reminder_penalty_applications'], dtype=np.int64
        )
        self.potential_deltas.append(float(step_data['potential_delta']))
        self.raw_potential_changes.append(float(
            step_data['raw_potential_change']
        ))
        for name, value in step_data['reward_components'].items():
            self.reward_components[name] += float(value)
            self.phase_reward_components[
                step_data['phase_before']
            ][name] += float(value)

    @staticmethod
    def triage_metrics(env):
        result = dict(env.mission_outcome_metrics())
        result['weighted_delivery_score'] = result['weighted_delivered']
        result['max_possible_weighted_score'] = result['weighted_spawned']
        return result

    def finish(self, env, global_step, learner_steps, device):
        outcome = env.mission_outcome_metrics()
        diagnostics = {
            'episode_mode': env.episode_mode,
            'curriculum_stage': self.curriculum_stage,
            'curriculum_stage_name': env.curriculum_stage_name,
            'curriculum_selection_probability': self.selection_probability,
            'curriculum_max_distance': int(env.curriculum_max_distance),
            'curriculum_max_landing_distance': int(
                env.curriculum_max_landing_distance
            ),
            'hazard_penalty_scale': float(env.hazard_penalty_scale),
            'episode_patient_timer': int(env.episode_patient_timer),
            'patient_spawn_interval': int(env.patient_spawn_interval),
            'patient_spawn_jitter': int(env.patient_spawn_jitter),
            'patient_spawn_batch_range': [
                int(env.minimum_patient_spawn_batch),
                int(env.maximum_patient_spawn_batch),
            ],
            'final_patient_spawn_step': int(
                env.final_patient_spawn_step
            ),
            'termination_reason': env.termination_reason,
            'mission_outcome': outcome,
            'rescue_success': bool(env.rescue_success()),
            'perfect_rescue': bool(env.perfect_rescue()),
            'safe_return_complete': bool(env.safe_return_complete()),
            'landing_deadline_step': env.landing_deadline,
            'landing_completion_step': env.landing_completion_step,
            'start_positions': [list(value) for value in env.start_positions],
            'patient_positions': [list(value) for value in env.patient_positions],
            'landing_zones': [list(value) for value in env.landing_zones],
            'initial_landing_distances': self.initial_distances,
            'deliveries_by_agent': self.deliveries_by_agent.tolist(),
            'local_reward_totals': self.local_rewards.tolist(),
            'local_potential_reward_totals': self.local_potential_rewards.tolist(),
            'local_reward_sum_error': float(
                self.local_rewards.sum() - self.total_reward
            ),
            'action_counts': self.action_counts.tolist(),
            'unconstrained_action_counts': (
                self.unconstrained_action_counts.tolist()
            ),
            'unconstrained_obstacle_preferences_by_agent': (
                self.unconstrained_obstacle_preferences.tolist()
            ),
            'unconstrained_boundary_preferences_by_agent': (
                self.unconstrained_boundary_preferences.tolist()
            ),
            'unconstrained_occupied_cell_preferences_by_agent': (
                self.unconstrained_occupied_cell_preferences.tolist()
            ),
            'phase_action_counts': {
                phase: values.tolist()
                for phase, values in self.phase_action_counts.items()
            },
            'phase_steps': dict(self.phase_steps),
            'reward_components_by_phase': self.phase_reward_components,
            'obstacle_action_opportunities_by_agent': (
                self.obstacle_opportunities.tolist()
            ),
            'obstacle_actions_selected_by_agent': (
                self.obstacle_actions_selected.tolist()
            ),
            'dominated_obstacle_selections_by_agent': (
                self.dominated_obstacle_selections.tolist()
            ),
            'dominated_agent_conflict_selections_by_agent': (
                self.dominated_agent_conflict_selections.tolist()
            ),
            'obstacle_collisions_by_agent': (
                self.obstacle_collisions_by_agent.tolist()
            ),
            'agent_collisions_by_agent': self.agent_collisions_by_agent.tolist(),
            'same_destination_collisions': self.same_destination_collisions,
            'head_on_collisions': self.head_on_collisions,
            'collision_steps': self.collision_steps,
            'obstacle_collision_steps': self.obstacle_collision_steps,
            'rescue_collisions': self.rescue_collisions,
            'landing_collisions': self.landing_collisions,
            'maximum_collision_streak': self.maximum_collision_streak,
            'maximum_agent_collision_streak': self.maximum_agent_collision_streak,
            'maximum_obstacle_collision_streak': (
                self.maximum_obstacle_collision_streak
            ),
            'minimum_agent_distance': (
                self.minimum_agent_distance
                if self.minimum_agent_distance < 2 * GRID_SIZE else -1
            ),
            'mean_valid_actions': float(np.mean(self.valid_action_counts)),
            'minimum_valid_actions': int(min(self.valid_action_counts)),
            'wind_entries_by_agent': self.wind_entries.tolist(),
            'wind_exposure_steps_by_agent': (
                self.wind_exposure_steps.tolist()
            ),
            'wind_exits_by_agent': self.wind_exits.tolist(),
            'wind_failures_by_agent': self.wind_failures.tolist(),
            'low_signal_entries_by_agent': self.low_signal_entries.tolist(),
            'low_signal_exposure_steps_by_agent': (
                self.low_signal_exposure_steps.tolist()
            ),
            'low_signal_exits_by_agent': self.low_signal_exits.tolist(),
            'low_signal_failures_by_agent': self.low_signal_failures.tolist(),
            'operational_steps_by_agent': self.operational_steps.tolist(),
            'operational_agent_steps': int(self.operational_steps.sum()),
            'movement_actions': int(self.movement_actions),
            'wind_command_attempts': int(self.wind_command_attempts),
            'low_signal_command_attempts': int(
                self.low_signal_command_attempts
            ),
            'obstacle_collision_rate_per_operational_step': float(
                self.obstacle_collisions / max(1, self.operational_steps.sum())
            ),
            'agent_collision_rate_per_operational_step': float(
                self.agent_collisions / max(1, self.operational_steps.sum())
            ),
            'wind_exposure_rate': float(
                self.wind_exposure_steps.sum()
                / max(1, self.operational_steps.sum())
            ),
            'wind_failure_rate': float(
                self.wind_failures.sum() / max(1, self.wind_command_attempts)
            ),
            'low_signal_exposure_rate': float(
                self.low_signal_exposure_steps.sum()
                / max(1, self.operational_steps.sum())
            ),
            'low_signal_failure_rate': float(
                self.low_signal_failures.sum()
                / max(1, self.low_signal_command_attempts)
            ),
            'wind_avoidance_opportunities': int(
                self.wind_avoidance_opportunities
            ),
            'wind_hazard_selections': int(self.wind_hazard_selections),
            'wind_dominated_hazard_selections': int(
                self.wind_dominated_hazard_selections
            ),
            'wind_shortcut_hazard_selections': int(
                self.wind_shortcut_hazard_selections
            ),
            'wind_avoidance_rate': float(
                1.0 - self.wind_hazard_selections
                / max(1, self.wind_avoidance_opportunities)
            ),
            'wind_dominated_avoidance_rate': float(
                1.0 - self.wind_dominated_hazard_selections
                / max(1, self.wind_avoidance_opportunities)
            ),
            'wind_rational_entry_fraction': float(
                self.wind_shortcut_hazard_selections
                / max(1, self.wind_hazard_selections)
            ),
            'low_signal_avoidance_opportunities': int(
                self.low_signal_avoidance_opportunities
            ),
            'low_signal_hazard_selections': int(
                self.low_signal_hazard_selections
            ),
            'low_signal_dominated_hazard_selections': int(
                self.low_signal_dominated_hazard_selections
            ),
            'low_signal_shortcut_hazard_selections': int(
                self.low_signal_shortcut_hazard_selections
            ),
            'low_signal_avoidance_rate': float(
                1.0 - self.low_signal_hazard_selections
                / max(1, self.low_signal_avoidance_opportunities)
            ),
            'low_signal_dominated_avoidance_rate': float(
                1.0 - self.low_signal_dominated_hazard_selections
                / max(1, self.low_signal_avoidance_opportunities)
            ),
            'low_signal_rational_entry_fraction': float(
                self.low_signal_shortcut_hazard_selections
                / max(1, self.low_signal_hazard_selections)
            ),
            'wind_entry_progress_cells': float(
                self.wind_entry_progress_cells
            ),
            'low_signal_entry_progress_cells': float(
                self.low_signal_entry_progress_cells
            ),
            'wind_progress_cells_per_entry': float(
                self.wind_entry_progress_cells
                / max(1, self.wind_entries.sum())
            ),
            'low_signal_progress_cells_per_entry': float(
                self.low_signal_entry_progress_cells
                / max(1, self.low_signal_entries.sum())
            ),
            'wind_zone_refreshes': int(self.wind_zone_refreshes),
            'low_signal_zone_refreshes': int(
                self.low_signal_zone_refreshes
            ),
            'wind_refresh_onsets_by_agent': (
                self.wind_refresh_onsets.tolist()
            ),
            'low_signal_refresh_onsets_by_agent': (
                self.low_signal_refresh_onsets.tolist()
            ),
            'patient_events': self.patient_events,
            'landing_events': self.landing_events,
            'landing_zone_arrival_events': self.landing_zone_arrivals,
            'landing_zone_departure_events': self.landing_zone_departures,
            'obstacle_collision_events': self.obstacle_collision_events,
            'battery_depletion_events': self.battery_depletion_events,
            'dead_landing_events': self.dead_landing_events,
            'death_penalty_applications_by_agent': (
                self.death_penalty_applications.tolist()
            ),
            'death_reminder_penalty_applications_by_agent': (
                self.death_reminder_penalty_applications.tolist()
            ),
            'battery_drain_by_agent': self.battery_drain.tolist(),
            'wind_battery_drain_by_agent': self.wind_battery_drain.tolist(),
            'landing_standby_steps_by_agent': (
                self.landing_standby_steps.tolist()
            ),
            'energy_return_mode_steps_by_agent': (
                self.energy_return_mode_steps.tolist()
            ),
            'energy_return_progress_steps_by_agent': (
                self.energy_return_progress_steps.tolist()
            ),
            'energy_return_regress_steps_by_agent': (
                self.energy_return_regress_steps.tolist()
            ),
            'energy_return_activations_by_agent': (
                self.energy_return_activations.tolist()
            ),
            'energy_margin_delta_by_agent': self.energy_margin_delta.tolist(),
            'reserve_violation_steps_by_agent': (
                self.reserve_violation_steps.tolist()
            ),
            'minimum_safe_return_margin_by_agent': [
                float(value) if math.isfinite(value) else 0.0
                for value in self.minimum_safe_return_margin
            ],
            'total_battery_drain': float(self.battery_drain.sum()),
            'total_wind_battery_drain': float(
                self.wind_battery_drain.sum()
            ),
            'wind_energy_fraction': float(
                self.wind_battery_drain.sum()
                / max(1e-8, self.battery_drain.sum())
            ),
            'battery_units_per_delivery': float(
                self.battery_drain.sum()
                / max(1, outcome['delivered_count'])
            ),
            'weighted_deliveries_per_battery_unit': float(
                outcome['weighted_delivered']
                / max(1e-8, self.battery_drain.sum())
            ),
            'energy_return_mode_fraction': float(
                self.energy_return_mode_steps.sum()
                / max(1, self.operational_steps.sum())
            ),
            'energy_return_progress_rate': float(
                self.energy_return_progress_steps.sum()
                / max(1, self.energy_return_mode_steps.sum())
            ),
            'energy_return_regress_rate': float(
                self.energy_return_regress_steps.sum()
                / max(1, self.energy_return_mode_steps.sum())
            ),
            'energy_return_success_rate': float(
                sum(
                    self.energy_return_activations[agent_index] > 0
                    and env.landed[agent_index]
                    for agent_index in range(NUM_AGENTS)
                ) / max(
                    1,
                    sum(self.energy_return_activations > 0),
                )
            ),
            'mean_landing_battery': float(np.mean([
                env.batteries[agent_index]
                for agent_index in range(NUM_AGENTS)
                if env.landed[agent_index]
            ])) if any(env.landed) else 0.0,
            'minimum_landing_battery': float(min([
                env.batteries[agent_index]
                for agent_index in range(NUM_AGENTS)
                if env.landed[agent_index]
            ])) if any(env.landed) else 0.0,
            'reserve_violation_rate': float(
                self.reserve_violation_steps.sum()
                / max(1, self.operational_steps.sum())
            ),
            'potential_delta_mean': float(np.mean(self.potential_deltas)),
            'potential_delta_min': float(np.min(self.potential_deltas)),
            'potential_delta_max': float(np.max(self.potential_deltas)),
            'raw_potential_change_mean': float(np.mean(
                self.raw_potential_changes
            )),
            'agent_path_lengths': list(env.agent_path_lengths),
            'agent_unique_cells': [
                len(positions) for positions in env.agent_unique_positions
            ],
            'patient_spawn_steps': list(env.patient_spawn_steps),
            'patient_resolution_steps': list(env.patient_resolution_steps),
            'patient_initial_timers': list(env.patient_initial_timers),
            'patient_time_to_resolution_ratio': [
                (
                    (env.patient_resolution_steps[patient_index]
                     - env.patient_spawn_steps[patient_index])
                    / max(1, env.patient_initial_timers[patient_index])
                    if env.patient_spawn_steps[patient_index] >= 0
                    and env.patient_resolution_steps[patient_index] >= 0
                    else -1.0
                )
                for patient_index in range(MAX_PATIENTS)
            ],
            'patient_delivery_agents': list(env.patient_delivery_agents),
            'initial_patient_weights': list(env.initial_patient_weights),
            'final_patient_weights': list(env.patient_weights),
            'patient_survival_probabilities': [
                float(value) for value in env.patient_survival_probabilities
            ],
            'patient_decay_rates': [
                float(value) for value in env.patient_decay_rates
            ],
            'patient_survival_offsets': [
                float(value) for value in env.patient_survival_offsets
            ],
            'patient_serious_thresholds': [
                float(value) for value in env.patient_serious_thresholds
            ],
            'patient_critical_thresholds': [
                float(value) for value in env.patient_critical_thresholds
            ],
            'wind_rectangles': [list(value) for value in env.wind_rectangles],
            'low_signal_rectangles': [
                list(value) for value in env.low_signal_rectangles
            ],
            'hazard_route_challenges': [
                dict(value) for value in env.hazard_route_challenges
            ],
            'triage_service_state': env.triage_service_state(),
            'final_batteries': [float(value) for value in env.batteries],
            'final_landed': [bool(value) for value in env.landed],
            'final_battery_depleted': [
                bool(value) for value in env.battery_depleted
            ],
            'final_drone_died': [
                bool(value) for value in env.drone_died
            ],
            'global_step': int(global_step),
            'learner_steps': int(learner_steps),
            'gpu_memory_allocated_mb': (
                float(torch.cuda.memory_allocated(device) / (1024.0 ** 2))
                if device.type == 'cuda' else 0.0
            ),
        }
        complexity = {
            'inference_latency_ms': float(np.mean(self.inference_latency_ms)),
            'environment_step_latency_ms': float(np.mean(
                self.environment_latency_ms
            )),
            'training_update_latency_ms': (
                float(np.mean(self.training_update_latency_ms))
                if self.training_update_latency_ms else 0.0
            ),
            'total_decision_latency_ms': float(
                np.mean(self.inference_latency_ms)
                + np.mean(self.environment_latency_ms)
            ),
            'process_rss_mb': float(get_peak_rss_mb()),
        }
        return {
            'total_reward': self.total_reward,
            'success': bool(
                env.episode_mode == 'full_mission' and env.mission_success()
            ),
            'curriculum_success': bool(
                env.episode_mode != 'full_mission' and env.mission_success()
            ),
            'agent_delivered': (self.deliveries_by_agent > 0).tolist(),
            'patients_delivered_count': int(sum(
                env.patients_actually_delivered[:env.episode_max_patients]
            )),
            'patients_died_count': int(sum(
                env.patients_died[:env.episode_max_patients]
            )),
            'patients_spawned_count': int(sum(
                env.patient_active[:env.episode_max_patients]
            )),
            'triage_data': self.triage_metrics(env),
            'complexity_data': complexity,
            'episode_diagnostics': diagnostics,
        }

# Training metrics
class Data_Collection:
    def __init__(self):
        self.episodes                  = []
        self.total_rewards             = []
        self.success_rate              = []          
        self.patients_delivered_counts = []         
        self.patients_died_counts      = []
        self.patients_spawned_counts   = []
        self.steps_per_episode         = []
        self.collisions_obstacles      = []
        self.collisions_agents         = []
        self.epsilon_values            = []
        self.episode_modes             = []
        self.curriculum_success_rate   = []
        self.curriculum_max_distances  = []
        self.curriculum_max_landing_distances = []
        self.curriculum_initial_distances = []
        self.curriculum_start_steps    = []
        self.qmix_losses               = []
        self.q_total_values            = []
        self.local_q_values            = []
        self.gradient_norms            = []
        self.absolute_td_errors         = []
        self.mean_importance_weights    = []
        self.gradient_clip_fractions    = []
        self.replay_sizes              = []
        self.replay_max_priorities      = []
        self.per_beta_values           = []
        self.reward_component_names = list(REWARD_COMPONENT_NAMES)
        self.reward_components = {
            name: [] for name in self.reward_component_names
        }

        
        
        for agent_index in range(NUM_AGENTS):
            setattr(self, f'agent_{agent_index}_delivered', [])
            setattr(self, f'agent_{agent_index}_landed', [])
            setattr(self, f'wind_entries_agent{agent_index}', [])
            setattr(self, f'low_signal_entries_agent{agent_index}', [])
            setattr(self, f'battery_remaining_agent{agent_index}', [])

        
        
        
        self.delivered_w1              = []   
        self.delivered_w2              = []   
        self.delivered_w3              = []   
        self.died_w1                   = []   
        self.died_w2                   = []
        self.died_w3                   = []
        
        
        
        self.weighted_delivery_score   = []
        self.max_possible_weighted_score = []
        self.triage_efficiency         = []
        self.lower_triage_delivery_floor = []
        self.acuity_priority_score     = []
        self.class_delivery_jain_fairness = []
        self.priority_normalized_jain_fairness = []
        self.priority_target_attainment = []
        self.priority_fairness_attainment = []
        self.triage_delivery_ordering_score = []
        self.triage_delivery_rate_ordering_score = []
        self.triage_response_time_ordering_score = []
        self.triage_response_tiebreak_pairs = []
        self.high_vs_low_response_advantage = []
        self.w3_vs_w1_response_advantage_steps = []
        self.w3_before_w1_response_fraction = []
        self.mean_delivered_response_time = []
        self.mean_response_ratio_w1 = []
        self.mean_response_ratio_w2 = []
        self.mean_response_ratio_w3 = []
        for weight in (1, 2, 3):
            setattr(self, f'mean_response_time_w{weight}', [])
            setattr(self, f'median_response_time_w{weight}', [])
            setattr(self, f'p90_response_time_w{weight}', [])
            setattr(self, f'first_response_time_w{weight}', [])

        
        
        
        self.inference_latency_ms       = []
        self.environment_step_latency_ms = []
        self.training_update_latency_ms = []
        self.total_decision_latency_ms  = []
        self.process_rss_mb             = []
        self.complexity_summary         = {}
        self.evaluation_history         = []
        self.episode_diagnostics        = []
        self.learning_diagnostics       = {}
        self.run_configuration          = {}

    def log_episode(self, episode, total_reward, success,
                    agent_delivered, patients_delivered_count, patients_died_count,
                    patients_spawned_count, landed, steps, collisions_obs, collisions_ag,
                    wind_entries, low_signal_entries, epsilon, batteries,
                    triage_data, complexity_data, learning_data,
                    reward_component_totals, episode_diagnostics,
                    episode_mode="full_mission", curriculum_max_distance=0,
                    curriculum_max_landing_distance=0,
                    curriculum_initial_distances=None,
                    curriculum_success=False, curriculum_start_step=0):
        self.episodes.append(episode)
        self.total_rewards.append(total_reward)
        self.success_rate.append(1 if success else 0)
        self.patients_delivered_counts.append(patients_delivered_count)
        self.patients_died_counts.append(patients_died_count)
        self.patients_spawned_counts.append(patients_spawned_count)
        self.steps_per_episode.append(steps)
        self.collisions_obstacles.append(collisions_obs)
        self.collisions_agents.append(collisions_ag)
        self.episode_modes.append(str(episode_mode))
        self.curriculum_success_rate.append(
            1 if str(episode_mode).startswith("curriculum_stage_")
            and curriculum_success else 0
        )
        self.curriculum_max_distances.append(int(curriculum_max_distance))
        self.curriculum_max_landing_distances.append(
            int(curriculum_max_landing_distance)
        )
        self.curriculum_initial_distances.append(list(
            curriculum_initial_distances or [0] * NUM_AGENTS
        ))
        self.curriculum_start_steps.append(int(curriculum_start_step))
        for agent_index in range(NUM_AGENTS):
            getattr(self, f'agent_{agent_index}_delivered').append(
                1 if agent_delivered[agent_index] else 0
            )
            getattr(self, f'agent_{agent_index}_landed').append(
                1 if landed[agent_index] else 0
            )
            getattr(self, f'wind_entries_agent{agent_index}').append(
                wind_entries[agent_index]
            )
            getattr(self, f'low_signal_entries_agent{agent_index}').append(
                low_signal_entries[agent_index]
            )
            getattr(self, f'battery_remaining_agent{agent_index}').append(
                batteries[agent_index]
            )
        self.epsilon_values.append(epsilon)
        self.delivered_w1.append(triage_data['delivered_w1'])
        self.delivered_w2.append(triage_data['delivered_w2'])
        self.delivered_w3.append(triage_data['delivered_w3'])
        self.died_w1.append(triage_data['died_w1'])
        self.died_w2.append(triage_data['died_w2'])
        self.died_w3.append(triage_data['died_w3'])
        self.weighted_delivery_score.append(triage_data['weighted_delivery_score'])
        self.max_possible_weighted_score.append(triage_data['max_possible_weighted_score'])
        self.triage_efficiency.append(triage_data['triage_efficiency'])
        self.lower_triage_delivery_floor.append(
            triage_data['lower_triage_delivery_floor']
        )
        self.acuity_priority_score.append(
            triage_data['acuity_priority_score']
        )
        self.class_delivery_jain_fairness.append(
            triage_data['class_delivery_jain_fairness']
        )
        self.priority_normalized_jain_fairness.append(
            triage_data['priority_normalized_jain_fairness']
        )
        self.priority_target_attainment.append(
            triage_data['priority_target_attainment']
        )
        self.priority_fairness_attainment.append(
            triage_data['priority_fairness_attainment']
        )
        self.triage_delivery_ordering_score.append(
            triage_data['triage_delivery_ordering_score']
        )
        self.triage_delivery_rate_ordering_score.append(
            triage_data['triage_delivery_rate_ordering_score']
        )
        self.triage_response_time_ordering_score.append(
            triage_data['triage_response_time_ordering_score']
        )
        self.triage_response_tiebreak_pairs.append(
            triage_data['triage_response_tiebreak_pairs']
        )
        self.high_vs_low_response_advantage.append(
            triage_data['high_vs_low_response_advantage']
        )
        self.w3_vs_w1_response_advantage_steps.append(
            triage_data['w3_vs_w1_response_advantage_steps']
        )
        self.w3_before_w1_response_fraction.append(
            triage_data['w3_before_w1_response_fraction']
        )
        self.mean_delivered_response_time.append(
            triage_data['mean_delivered_response_time']
        )
        for weight in (1, 2, 3):
            getattr(self, f'mean_response_ratio_w{weight}').append(
                triage_data[f'mean_response_ratio_w{weight}']
            )
            for statistic in (
                    'mean', 'median', 'p90', 'first'):
                getattr(
                    self, f'{statistic}_response_time_w{weight}'
                ).append(
                    triage_data[
                        f'{statistic}_response_time_w{weight}'
                    ]
                )
        self.qmix_losses.append(learning_data['loss'])
        self.q_total_values.append(learning_data['q_total'])
        self.local_q_values.append(learning_data['local_q'])
        self.gradient_norms.append(learning_data['gradient_norm'])
        self.absolute_td_errors.append(learning_data['absolute_td_error'])
        self.mean_importance_weights.append(
            learning_data['mean_importance_weight']
        )
        self.gradient_clip_fractions.append(
            learning_data['gradient_clip_fraction']
        )
        self.replay_sizes.append(learning_data['replay_size'])
        self.replay_max_priorities.append(
            learning_data['replay_max_priority']
        )
        self.per_beta_values.append(learning_data['per_beta'])
        legacy_learning_keys = {
            'loss', 'q_total', 'local_q', 'gradient_norm',
            'absolute_td_error', 'mean_importance_weight',
            'gradient_clip_fraction', 'replay_size',
            'replay_max_priority', 'per_beta'
        }
        existing_episode_count = len(self.episodes) - 1
        for metric_name, metric_values in self.learning_diagnostics.items():
            if metric_name not in learning_data:
                metric_values.append(None)
        for metric_name, metric_value in learning_data.items():
            if metric_name in legacy_learning_keys:
                continue
            if metric_name not in self.learning_diagnostics:
                self.learning_diagnostics[metric_name] = (
                    [None] * existing_episode_count
                )
            self.learning_diagnostics[metric_name].append(metric_value)
        for component_name in self.reward_component_names:
            self.reward_components[component_name].append(
                reward_component_totals[component_name]
            )
        
        
        self.inference_latency_ms.append(complexity_data['inference_latency_ms'])
        self.environment_step_latency_ms.append(complexity_data['environment_step_latency_ms'])
        self.training_update_latency_ms.append(complexity_data['training_update_latency_ms'])
        self.total_decision_latency_ms.append(complexity_data['total_decision_latency_ms'])
        self.process_rss_mb.append(complexity_data['process_rss_mb'])
        self.episode_diagnostics.append(episode_diagnostics)

    def save_to_json(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"marl_training_data_{timestamp}.json"

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        temporary_filename = filename.with_suffix(filename.suffix + ".tmp")

        source_path = Path(__file__).resolve()
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        data = {
            'metrics_schema_version': 18,
            'training_configuration': {
                'model_version': MODEL_VERSION,
                'source_file': str(source_path),
                'source_sha256': source_sha256,
                'python_version': platform.python_version(),
                'pytorch_version': str(torch.__version__),
                'cuda_version': torch.version.cuda,
                'grid_size': GRID_SIZE,
                'num_obstacles': NUM_OBSTACLES,
                'num_wind_zone_rectangles': NUM_WIND_ZONE_RECTANGLES,
                'num_low_signal_zone_rectangles': (
                    NUM_LOW_SIGNAL_ZONE_RECTANGLES
                ),
                'hazard_rectangle_width_range': [
                    HAZARD_RECTANGLE_MIN_WIDTH,
                    HAZARD_RECTANGLE_MAX_WIDTH,
                ],
                'hazard_rectangle_height_range': [
                    HAZARD_RECTANGLE_MIN_HEIGHT,
                    HAZARD_RECTANGLE_MAX_HEIGHT,
                ],
                'hazard_rectangles_may_overlap_obstacles': True,
                'wind_zone_refresh_interval_steps': WIND_APPEAR_INTERVAL,
                'low_signal_zone_refresh_interval_steps': (
                    LOW_SIGNAL_APPEAR_INTERVAL
                ),
                'hazard_metric_semantics': {
                    'entries': 'outside-to-inside transitions',
                    'exposure_steps': 'operational agent-steps inside zone',
                    'avoidance_rate': (
                        'one minus hazardous selections divided by '
                        'available hazardous-entry actions'
                    ),
                    'refresh_onsets': (
                        'operational agents newly covered when zones refresh'
                    ),
                    'dominated_selection': (
                        'hazard selected while a safe neighbor makes equal or '
                        'better target progress'
                    ),
                    'shortcut_selection': (
                        'hazard selected only when it improves one-step target '
                        'progress over every safe neighbor'
                    ),
                },
                'local_grid_radius': LOCAL_GRID_RADIUS,
                'local_grid_size': LOCAL_GRID_SIZE,
                'expanded_decentralized_hazard_observation': True,
                'directional_hazard_corridor_features': True,
                'decision_aware_hazard_routing_reward': True,
                'progressive_hazard_penalty_curriculum': True,
                'hazards_present_from_first_curriculum_stage': True,
                'hazard_actions_masked': False,
                'route_intercepting_hazard_rectangles': True,
                'hazard_route_source': 'nearest_drone_to_patient_shortest_path',
                'hazard_route_fraction_range': [
                    HAZARD_ROUTE_MIN_FRACTION,
                    HAZARD_ROUTE_MAX_FRACTION,
                ],
                'hazard_safe_detour_required': True,
                'hazard_max_safe_detour_ratio': (
                    HAZARD_MAX_SAFE_DETOUR_RATIO
                ),
                'num_agents': NUM_AGENTS,
                'num_initial_patients': NUM_INITIAL_PATIENTS,
                'max_patients': MAX_PATIENTS,
                'energy_budget': {
                    'battery_units': 'percentage_points',
                    'maximum_battery': MAX_BATTERY,
                    'clean_step_drain': BATTERY_DRAIN_PER_STEP,
                    'clean_step_full_charge_percent': (
                        100.0 * BATTERY_DRAIN_PER_STEP / MAX_BATTERY
                    ),
                    'wind_additional_drain': BATTERY_DRAIN_IN_WIND,
                    'wind_total_step_drain': (
                        BATTERY_DRAIN_PER_STEP + BATTERY_DRAIN_IN_WIND
                    ),
                    'landing_zone_standby_drain': (
                        BATTERY_DRAIN_AT_LANDING_ZONE
                    ),
                    'clean_endurance_steps': (
                        MAX_BATTERY / BATTERY_DRAIN_PER_STEP
                    ),
                    'wind_step_full_charge_fraction': (
                        (
                            BATTERY_DRAIN_PER_STEP
                            + BATTERY_DRAIN_IN_WIND
                        ) / MAX_BATTERY
                    ),
                    'low_battery_threshold': LOW_BATTERY_THRESHOLD,
                    'safe_return_buffer': SAFE_RETURN_BATTERY_BUFFER,
                    'energy_return_trigger_margin': (
                        ENERGY_RETURN_TRIGGER_MARGIN
                    ),
                    'return_energy_risk_multiplier': (
                        RETURN_ENERGY_RISK_MULTIPLIER
                    ),
                    'obstacle_aware_return_reserve': True,
                    'stochastic_risk_adjusted_return_reserve': True,
                    'wind_signal_expected_cost_return_map': True,
                    'single_return_safety_reserve': True,
                    'energy_return_navigation_shaping': True,
                    'energy_margin_shaping_factor': (
                        ENERGY_MARGIN_SHAPING_FACTOR
                    ),
                    'candidate_rescue_to_pad_feasibility_observation': True,
                    'episode_continues_after_drone_death': True,
                    'dead_drones_remain_physically_inactive': True,
                    'death_reminder_steps': POST_DEPLETION_REMINDER_STEPS,
                    'death_reminder_penalty': (
                        POST_DEPLETION_REMINDER_PENALTY
                    ),
                    'dead_landing_penalty': DEAD_LANDING_PENALTY,
                    'pre_resolution_energy_return_landing': True,
                },
                'patient_timer': MAX_PATIENT_TIMER,
                'patient_timer_normalization_max': MAX_PATIENT_TIMER,
                'stage_specific_patient_timers': True,
                'survival_decay_ranges': SURVIVAL_DECAY_RANGES,
                'survival_offset_ranges': SURVIVAL_OFFSET_RANGES,
                'serious_survival_threshold_range': (
                    SERIOUS_SURVIVAL_THRESHOLD_RANGE
                ),
                'critical_survival_threshold_range': (
                    CRITICAL_SURVIVAL_THRESHOLD_RANGE
                ),
                'patient_spawn_interval': NEW_PATIENT_SPAWN_INTERVAL,
                'patient_spawn_interval_jitter': (
                    PATIENT_SPAWN_INTERVAL_JITTER
                ),
                'final_patient_spawn_step': FINAL_PATIENT_SPAWN_STEP,
                'patient_spawn_batch_range': [
                    MIN_PATIENT_SPAWN_BATCH, MAX_PATIENT_SPAWN_BATCH
                ],
                'stage_specific_arrival_bursts': True,
                'scarcity_curriculum': True,
                'curriculum_stage_parameters_are_authoritative': True,
                'action_dim': ACTION_DIM,
                'action_names': list(ACTION_NAMES),
                'drone_state_dim': DRONE_STATE_DIM,
                'patient_state_dim': PATIENT_STATE_DIM,
                'batch_size': BATCH_SIZE,
                'buffer_capacity': BUFFER_CAPACITY,
                'replay_warmup': REPLAY_WARMUP,
                'n_step': N_STEP,
                'gamma': GAMMA,
                'learning_rate': LEARNING_RATE,
                'per_alpha': PER_ALPHA,
                'per_beta_start': PER_BETA_START,
                'per_beta_end': PER_BETA_END,
                'per_priority_max': PER_PRIORITY_MAX,
                'replay_uniform_fraction': REPLAY_UNIFORM_FRACTION,
                'replay_rescue_fraction': REPLAY_RESCUE_FRACTION,
                'replay_landing_fraction': REPLAY_LANDING_FRACTION,
                'td_reward_scale': TD_REWARD_SCALE,
                'local_td_loss_weight': LOCAL_TD_LOSS_WEIGHT,
                'gradient_clip': GRADIENT_CLIP,
                'train_every_steps': TRAIN_EVERY_STEPS,
                'updates_per_train': UPDATES_PER_TRAIN,
                'target_update_steps': TARGET_UPDATE_STEPS,
                'mission_state_dim': MISSION_STATE_DIM,
                'epsilon_is_per_agent': True,
                'epsilon_start': EPSILON_START,
                'epsilon_mid': EPSILON_MID,
                'epsilon_end': EPSILON_END,
                'epsilon_mid_step': EPSILON_MID_STEP,
                'epsilon_end_step': EPSILON_END_STEP,
                'curriculum_epsilon_reset': CURRICULUM_EPSILON_RESET,
                'curriculum_epsilon_floor': CURRICULUM_EPSILON_FLOOR,
                'curriculum_epsilon_decay_episodes': (
                    CURRICULUM_EPSILON_DECAY_EPISODES
                ),
                'static_invalid_action_masking': False,
                'boundary_invalid_action_masking': False,
                'obstacle_action_masking': False,
                'occupied_cell_action_masking': False,
                'learned_obstacle_and_agent_avoidance': True,
                'head_on_swap_collision_detection': True,
                'safe_patient_spawn_deadline': True,
                'agent_aligned_bounded_qmix': True,
                'mixer_min_raw_weight': MIXER_MIN_RAW_WEIGHT,
                'action_conditioned_entity_attention': True,
                'phase_specific_dueling_heads': True,
                'fp32_action_value_heads': True,
                'phase_balanced_replay': True,
                'stage_and_phase_balanced_replay': True,
                'obstacle_aware_landing_distance': True,
                'connected_random_mission_layout': True,
                'competence_gated_full_curriculum': True,
                'hard_stage_budget_curriculum_promotion': (
                    CURRICULUM_FORCE_PROMOTION_AT_MAXIMUM
                ),
                'guaranteed_final_stage_exposure': True,
                'curriculum_stages': [dict(stage) for stage in CURRICULUM_STAGES],
                'curriculum_required_consecutive_passes': (
                    CURRICULUM_REQUIRED_PASSES
                ),
                'pending_only_local_patient_attention': True,
                'explicit_hover_action': True,
                'terminal_land_only_action_masking': True,
                'energy_return_land_only_action_masking': True,
                'per_drone_died_observation': True,
                'post_death_motion_allowed': False,
                'fleet_death_immediate_termination': False,
                'landing_zone_relative_direction_features': True,
                'irrecoverable_failure_termination': False,
                'post_resolution_landing_phase': True,
                'landing_after_patient_deaths': True,
                'reward_derived_local_td_credit': True,
                'agent_attributed_navigation_potential': True,
                'priority_neutral_unique_coverage_shaping': True,
                'individual_logistic_survival_dynamics': True,
                'patient_response_age_observation': True,
                'priority_weighted_response_reward': True,
                'dense_priority_service_debt_reward': True,
                'priority_service_potential_scale': (
                    PRIORITY_SERVICE_POTENTIAL_SCALE
                ),
                'priority_death_penalty_growth': (
                    PRIORITY_DEATH_PENALTY_GROWTH
                ),
                'counterfactual_collision_selection_penalties': True,
                'response_ordering_curriculum_gate': True,
                'energy_safety_curriculum_gate': True,
                'wind_movement_failure_probability': (
                    WIND_MOVEMENT_FAILURE_PROB
                ),
                'application_success_thresholds': {
                    'delivery_utilization': (
                        APPLICATION_DELIVERY_SUCCESS_THRESHOLD
                    ),
                    'triage_efficiency': (
                        APPLICATION_TRIAGE_SUCCESS_THRESHOLD
                    ),
                    'lower_triage_delivery_floor': (
                        APPLICATION_LOWER_TRIAGE_FLOOR
                    ),
                    'acuity_priority_score': (
                        APPLICATION_ACUITY_PRIORITY_THRESHOLD
                    ),
                    'priority_fairness_attainment': (
                        APPLICATION_PRIORITY_FAIRNESS_THRESHOLD
                    ),
                    'triage_delivery_ordering_score': (
                        APPLICATION_TRIAGE_ORDERING_THRESHOLD
                    ),
                    'triage_response_time_ordering_score': (
                        APPLICATION_TRIAGE_RESPONSE_THRESHOLD
                    ),
                },
                'triage_class_delivery_targets': {
                    str(weight): float(target)
                    for weight, target in TRIAGE_CLASS_DELIVERY_TARGETS.items()
                },
                'priority_fairness_definition': (
                    'Jain fairness over delivery-rate/target fulfillment, '
                    'multiplied by mean target attainment'
                ),
                'triage_ordering_definition': (
                    'lexicographic delivery-rate ordering with normalized '
                    'mean response time as the tie-breaker'
                ),
                'response_time_tie_breaking': True,
                'response_time_is_decentralized_patient_state': True,
                'response_reward_is_reward_only': True,
                'response_times_normalized_by_patient_deadline': True,
                'delivery_reward_time_fraction_uses_patient_deadline': True,
                'patient_timer_observation_normalization': (
                    'remaining steps divided by maximum configured timer'
                ),
                'supervised_targets_or_demonstrations': False,
                'evaluation_trace_episode_limit': EVALUATION_TRACE_EPISODES,
                'evaluation_trace_record_limit': EVALUATION_TRACE_MAX_RECORDS,
                'obstacle_event_record_limit': (
                    MAX_RECORDED_OBSTACLE_EVENTS_PER_EPISODE
                ),
                'attention_diagnostic_interval': ATTENTION_DIAGNOSTIC_INTERVAL,
                'replay_event_priority_multipliers': {
                    'delivery': EVENT_PRIORITY_MULTIPLIERS[EVENT_DELIVERY],
                    'collision': EVENT_PRIORITY_MULTIPLIERS[EVENT_COLLISION],
                    'progress': EVENT_PRIORITY_MULTIPLIERS[EVENT_PROGRESS],
                    'landing': EVENT_PRIORITY_MULTIPLIERS[EVENT_LANDING],
                    'terminal': EVENT_PRIORITY_MULTIPLIERS[EVENT_TERMINAL],
                    'landing_phase': EVENT_PRIORITY_MULTIPLIERS[
                        EVENT_LANDING_PHASE
                    ],
                    'obstacle_collision': EVENT_PRIORITY_MULTIPLIERS[
                        EVENT_OBSTACLE_COLLISION
                    ],
                    'hazard': EVENT_PRIORITY_MULTIPLIERS[EVENT_HAZARD],
                },
                'observation_schema': {
                    'drone_features': [
                        'x', 'y', 'battery', 'landed', 'died',
                        'landing_x', 'landing_y', 'previous_dx', 'previous_dy',
                        'previous_collision', 'collision_streak',
                        *[f'previous_action_{name}' for name in ACTION_NAMES],
                        'landing_shortest_path_distance',
                        'safe_return_battery_margin',
                        'currently_in_wind', 'currently_in_low_signal',
                        'energy_return_mode',
                    ],
                    'patient_features': [
                        'x', 'y', 'timer', 'current_weight', 'initial_weight',
                        'active', 'pending', 'delivered', 'died',
                        'elapsed_response_fraction',
                    ],
                    'mission_features': [
                        'episode_progress', 'next_spawn_timer',
                        'spawned_fraction', 'pending_fraction',
                        'delivered_fraction', 'died_fraction',
                        'landed_fraction', 'all_spawned', 'landing_phase',
                        'w1_target_debt', 'w2_target_debt', 'w3_target_debt',
                    ],
                    'local_grid_channels': [
                        'obstacle_or_boundary', 'wind', 'low_signal'
                    ],
                    'local_grid_shape': [
                        3, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE
                    ],
                    'patient_attention_null_index': MAX_PATIENTS,
                    'drone_attention_null_index': NUM_AGENTS,
                },
                'architecture': {
                    'entity_embed_dim': ENTITY_EMBED_DIM,
                    'attention_heads': ATTENTION_HEADS,
                    'set_attention_blocks': SET_ATTENTION_BLOCKS,
                    'self_embed_dim': SELF_EMBED_DIM,
                    'grid_embed_dim': GRID_EMBED_DIM,
                    'mixer_embed_dim': MIXER_EMBED_DIM,
                    'phase_specific_dueling_heads': 2,
                    'global_embed_dim': GLOBAL_EMBED_DIM,
                    'agent_id_embed_dim': AGENT_ID_EMBED_DIM,
                    'decentralized_action_coordination_features': 6,
                    
                    
                    
                    
                    'decentralized_action_hazard_features': 3,
                    'decentralized_patient_action_edge_features': 14,
                    'candidate_rescue_energy_features': [
                        'risk_adjusted_rescue_to_pad_energy',
                        'post_rescue_energy_margin',
                        'post_rescue_energy_feasible',
                    ],
                },
                'reward_parameters': {
                    'goal': GOAL_REWARD,
                    'step': STEP_PENALTY,
                    'clean': CLEAN_STEP_BONUS,
                    'patient_death': PATIENT_DEATH_PENALTY,
                    'obstacle_collision': COLLISION_PENALTY,
                    'agent_collision': AGENT_COLLISION_PENALTY,
                    'dominated_obstacle_selection': (
                        DOMINATED_OBSTACLE_SELECTION_PENALTY
                    ),
                    'dominated_agent_conflict': (
                        DOMINATED_AGENT_CONFLICT_PENALTY
                    ),
                    'battery_depletion': BATTERY_DEPLETION_PENALTY,
                    'post_depletion_reminder': (
                        POST_DEPLETION_REMINDER_PENALTY
                    ),
                    'post_depletion_reminder_steps': (
                        POST_DEPLETION_REMINDER_STEPS
                    ),
                    'dead_landing': DEAD_LANDING_PENALTY,
                    'low_battery': LOW_BATTERY_PENALTY,
                    'energy_usage_per_battery_unit': (
                        ENERGY_USAGE_PENALTY_PER_UNIT
                    ),
                    'safe_return_reserve_violation': (
                        SAFE_RETURN_RESERVE_PENALTY
                    ),
                    'energy_margin_shaping_factor': (
                        ENERGY_MARGIN_SHAPING_FACTOR
                    ),
                    'response_wait_per_patient': (
                        RESPONSE_WAIT_PENALTY_PER_PATIENT
                    ),
                    'response_time_delivery': RESPONSE_TIME_DELIVERY_REWARD,
                    'wind': WIND_PENALTY,
                    'low_signal': LOW_SIGNAL_PENALTY,
                    'wind_entry': WIND_ENTRY_PENALTY,
                    'low_signal_entry': LOW_SIGNAL_ENTRY_PENALTY,
                    'wind_dominated_selection': (
                        WIND_DOMINATED_SELECTION_PENALTY
                    ),
                    'wind_shortcut_selection': (
                        WIND_SHORTCUT_SELECTION_PENALTY
                    ),
                    'low_signal_dominated_selection': (
                        LOW_SIGNAL_DOMINATED_SELECTION_PENALTY
                    ),
                    'low_signal_shortcut_selection': (
                        LOW_SIGNAL_SHORTCUT_SELECTION_PENALTY
                    ),
                    'potential_factor': SHAPING_FACTOR,
                    'landing': LANDING_REWARD,
                    'early_landing': EARLY_LANDING_PENALTY,
                    'wrong_landing': LAND_WRONG_PENALTY,
                    'mission_success': MISSION_SUCCESS_REWARD,
                    'mission_failure': MISSION_FAILURE_PENALTY,
                    'hover': HOVER_PENALTY,
                    'landing_hover': LANDING_HOVER_PENALTY,
                    'energy_standby_hover': (
                        ENERGY_STANDBY_HOVER_PENALTY
                    ),
                    'closeness': CLOSENESS_PENALTY,
                    'rescue_outcome_linear': RESCUE_OUTCOME_LINEAR_REWARD,
                    'rescue_outcome_quadratic': RESCUE_OUTCOME_QUADRATIC_REWARD,
                    'safe_return_base': SAFE_RETURN_BASE_REWARD,
                    'safe_return_quality': SAFE_RETURN_QUALITY_REWARD,
                    'fairness_outcome': FAIRNESS_OUTCOME_REWARD,
                    'priority_service_potential_scale': (
                        PRIORITY_SERVICE_POTENTIAL_SCALE
                    ),
                    'priority_death_penalty_growth': (
                        PRIORITY_DEATH_PENALTY_GROWTH
                    ),
                    'triage_ordering_outcome': (
                        TRIAGE_ORDERING_OUTCOME_REWARD
                    ),
                    'triage_response_outcome': (
                        TRIAGE_RESPONSE_OUTCOME_REWARD
                    ),
                },
            },
            'run_configuration': self.run_configuration,
            'episodes':                  self.episodes,
            'total_rewards':             self.total_rewards,
            'success_rate':              self.success_rate,
            'episode_modes':             self.episode_modes,
            'curriculum_success_rate':   self.curriculum_success_rate,
            'curriculum_max_distances':  self.curriculum_max_distances,
            'curriculum_max_landing_distances': (
                self.curriculum_max_landing_distances
            ),
            'curriculum_initial_distances': (
                self.curriculum_initial_distances
            ),
            'curriculum_start_steps':      self.curriculum_start_steps,
            **{
                f'agent_{agent_index}_delivered':
                    getattr(self, f'agent_{agent_index}_delivered')
                for agent_index in range(NUM_AGENTS)
            },
            'patients_delivered_counts': self.patients_delivered_counts,
            'patients_died_counts':      self.patients_died_counts,
            'patients_spawned_counts':   self.patients_spawned_counts,
            **{
                f'agent_{agent_index}_landed':
                    getattr(self, f'agent_{agent_index}_landed')
                for agent_index in range(NUM_AGENTS)
            },
            'steps_per_episode':         self.steps_per_episode,
            'collisions_obstacles':      self.collisions_obstacles,
            'collisions_agents':         self.collisions_agents,
            **{
                f'wind_entries_agent{agent_index}':
                    getattr(self, f'wind_entries_agent{agent_index}')
                for agent_index in range(NUM_AGENTS)
            },
            **{
                f'low_signal_entries_agent{agent_index}':
                    getattr(self, f'low_signal_entries_agent{agent_index}')
                for agent_index in range(NUM_AGENTS)
            },
            **{
                f'battery_remaining_agent{agent_index}':
                    getattr(self, f'battery_remaining_agent{agent_index}')
                for agent_index in range(NUM_AGENTS)
            },
            'epsilon_values':            self.epsilon_values,
            'qmix_losses':               self.qmix_losses,
            'q_total_values':            self.q_total_values,
            'local_q_values':            self.local_q_values,
            'gradient_norms':            self.gradient_norms,
            'absolute_td_errors':         self.absolute_td_errors,
            'mean_importance_weights':    self.mean_importance_weights,
            'gradient_clip_fractions':    self.gradient_clip_fractions,
            'replay_sizes':              self.replay_sizes,
            'replay_max_priorities':      self.replay_max_priorities,
            'per_beta_values':           self.per_beta_values,
            'learning_diagnostics':      self.learning_diagnostics,
            **{
                f'reward_{component_name}': self.reward_components[component_name]
                for component_name in self.reward_component_names
            },
            'delivered_w1':              self.delivered_w1,
            'delivered_w2':              self.delivered_w2,
            'delivered_w3':              self.delivered_w3,
            'died_w1':                   self.died_w1,
            'died_w2':                   self.died_w2,
            'died_w3':                   self.died_w3,
            'weighted_delivery_score':   self.weighted_delivery_score,
            'max_possible_weighted_score': self.max_possible_weighted_score,
            'triage_efficiency':         self.triage_efficiency,
            'lower_triage_delivery_floor': (
                self.lower_triage_delivery_floor
            ),
            'acuity_priority_score':     self.acuity_priority_score,
            'class_delivery_jain_fairness': (
                self.class_delivery_jain_fairness
            ),
            'priority_normalized_jain_fairness': (
                self.priority_normalized_jain_fairness
            ),
            'priority_target_attainment': self.priority_target_attainment,
            'priority_fairness_attainment': (
                self.priority_fairness_attainment
            ),
            'triage_delivery_ordering_score': (
                self.triage_delivery_ordering_score
            ),
            'triage_delivery_rate_ordering_score': (
                self.triage_delivery_rate_ordering_score
            ),
            'triage_response_time_ordering_score': (
                self.triage_response_time_ordering_score
            ),
            'triage_response_tiebreak_pairs': (
                self.triage_response_tiebreak_pairs
            ),
            'high_vs_low_response_advantage': (
                self.high_vs_low_response_advantage
            ),
            'w3_vs_w1_response_advantage_steps': (
                self.w3_vs_w1_response_advantage_steps
            ),
            'w3_before_w1_response_fraction': (
                self.w3_before_w1_response_fraction
            ),
            'mean_delivered_response_time': (
                self.mean_delivered_response_time
            ),
            **{
                f'mean_response_ratio_w{weight}': getattr(
                    self, f'mean_response_ratio_w{weight}'
                )
                for weight in (1, 2, 3)
            },
            **{
                f'{statistic}_response_time_w{weight}': getattr(
                    self, f'{statistic}_response_time_w{weight}'
                )
                for weight in (1, 2, 3)
                for statistic in ('mean', 'median', 'p90', 'first')
            },
            
            
            'inference_latency_ms':       self.inference_latency_ms,
            'environment_step_latency_ms': self.environment_step_latency_ms,
            'training_update_latency_ms': self.training_update_latency_ms,
            'total_decision_latency_ms':  self.total_decision_latency_ms,
            'process_rss_mb':             self.process_rss_mb,
            'complexity_summary':          self.complexity_summary,
            'evaluation_history':          self.evaluation_history,
            'episode_diagnostics':         self.episode_diagnostics,
        }
        with temporary_filename.open('w', encoding='utf-8') as f:
            json.dump(
                data, f, separators=(',', ':'), allow_nan=False
            )
            f.flush()
            os.fsync(f.fileno())
        temporary_filename.replace(filename)

        print(f"\n Data saved to {filename}")
        return filename

# Training and evaluation
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CEDA-FGCS CTDE agent."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable Pygame display and rendering for cloud training."
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("CEDA_OUTPUT_DIR", "."),
        help="Directory for training JSON and model files."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=NUM_EPISODES,
        help=f"Training episodes; default: {NUM_EPISODES}."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum steps per episode; default: {MAX_STEPS}."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed; default: 7."
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Save metrics and a model/optimizer checkpoint every N episodes."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or a CUDA device such as cuda:0."
    )
    parser.add_argument(
        "--fast-math",
        action="store_true",
        help="Enable NVIDIA TF32 math for faster FP32 training."
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable default CUDA BF16 mixed-precision training."
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable tracemalloc for diagnostic runs; it slows long training."
    )
    parser.add_argument(
        "--evaluation-every",
        type=int,
        default=EVALUATION_EVERY_EPISODES,
        help=(
            "Run isolated epsilon-zero evaluation every N training episodes; "
            f"default: {EVALUATION_EVERY_EPISODES}. Use 0 to disable."
        )
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=EVALUATION_EPISODES,
        help=(
            "Number of fixed-seed greedy episodes per evaluation; "
            f"default: {EVALUATION_EPISODES}."
        )
    )
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=ROLLOUT_WORKERS,
        help=(
            "Independent environments collected with batched policy inference; "
            f"default: {ROLLOUT_WORKERS}. Use 1 for the legacy serial path."
        )
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume policy, mixer, targets, optimizer, curriculum, and global "
            "step from a matching current-model checkpoint. Replay is rebuilt "
            "after restart."
        )
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run two short episodes to verify the environment and GPU."
    )
    return parser.parse_args()

def resolve_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested cuda:{index}, but only {torch.cuda.device_count()} GPU(s) are visible."
            )
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is unavailable.")
    return device

def get_peak_rss_mb():

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak_rss / (1024.0 ** 2) if sys.platform == 'darwin' else peak_rss / 1024.0

def epsilon_at_step(global_step):

    if global_step <= EPSILON_MID_STEP:
        fraction = global_step / max(1, EPSILON_MID_STEP)
        return EPSILON_START + fraction * (EPSILON_MID - EPSILON_START)
    if global_step <= EPSILON_END_STEP:
        fraction = (
            (global_step - EPSILON_MID_STEP)
            / max(1, EPSILON_END_STEP - EPSILON_MID_STEP)
        )
        return EPSILON_MID + fraction * (EPSILON_END - EPSILON_MID)
    return EPSILON_END

def training_epsilon_at_step(global_step, current_stage_episodes):

    base_epsilon = epsilon_at_step(global_step)
    stage_episodes = max(0, int(current_stage_episodes))
    stage_fraction = min(
        1.0,
        stage_episodes / max(1, CURRICULUM_EPSILON_DECAY_EPISODES),
    )
    stage_epsilon = (
        CURRICULUM_EPSILON_RESET
        + stage_fraction
        * (CURRICULUM_EPSILON_FLOOR - CURRICULUM_EPSILON_RESET)
    )
    return max(base_epsilon, stage_epsilon)

class CurriculumManager:

    def __init__(self):
        self.current_stage = 0
        self.current_stage_episodes = 0
        self.stage_episode_counts = [0] * len(CURRICULUM_STAGES)
        self.consecutive_passes = 0
        self.promotion_history = []

    def select_training_stage(self):
        final_stage = len(CURRICULUM_STAGES) - 1
        if self.current_stage == final_stage:
            return final_stage
        draw = random.random()
        if draw < CURRICULUM_CURRENT_PROBABILITY:
            return self.current_stage
        if draw < (
                CURRICULUM_CURRENT_PROBABILITY
                + CURRICULUM_PREVIOUS_PROBABILITY):
            return max(0, self.current_stage - 1)
        return final_stage

    def record_training_episode(self, stage_index):
        stage_index = int(stage_index)
        self.stage_episode_counts[stage_index] += 1
        if stage_index == self.current_stage:
            self.current_stage_episodes += 1

    def promotion_ready_for_evaluation(self):
        if self.current_stage >= len(CURRICULUM_STAGES) - 1:
            return False
        return self.current_stage_episodes >= int(
            CURRICULUM_STAGES[self.current_stage]['minimum_episodes']
        )

    def record_evaluation(self, evaluation_data, training_episode):
        stage_before = self.current_stage
        stage = CURRICULUM_STAGES[stage_before]
        delivery_rate = (
            evaluation_data['mean_delivered'] / max(1, stage['max_patients'])
        )
        triage_rate = evaluation_data['mean_triage_efficiency']
        landing_rate = (
            evaluation_data['mean_landed'] / max(1, NUM_AGENTS)
        )
        lower_triage_rate = evaluation_data[
            'mean_lower_triage_delivery_floor'
        ]
        acuity_priority = evaluation_data['mean_acuity_priority_score']
        priority_fairness = evaluation_data[
            'mean_priority_fairness_attainment'
        ]
        triage_ordering = evaluation_data[
            'mean_triage_delivery_ordering_score'
        ]
        response_ordering = evaluation_data[
            'mean_triage_response_time_ordering_score'
        ]
        battery_depletion_free = evaluation_data[
            'battery_depletion_free_rate'
        ]
        reserve_violation = evaluation_data[
            'mean_reserve_violation_rate'
        ]
        wind_avoidance = evaluation_data['aggregate_wind_avoidance_rate']
        low_signal_avoidance = evaluation_data[
            'aggregate_low_signal_avoidance_rate'
        ]
        wind_opportunities = evaluation_data[
            'total_wind_avoidance_opportunities'
        ]
        low_signal_opportunities = evaluation_data[
            'total_low_signal_avoidance_opportunities'
        ]
        hazard_evidence = bool(
            wind_opportunities >= stage['minimum_hazard_opportunities']
            and low_signal_opportunities
            >= stage['minimum_hazard_opportunities']
        )
        collision_rate = max(
            evaluation_data[
                'mean_obstacle_collision_rate_per_operational_step'
            ],
            evaluation_data[
                'mean_agent_collision_rate_per_operational_step'
            ],
        )
        gate_checks = {
            'delivery': delivery_rate >= stage['delivery_gate'],
            'triage': triage_rate >= stage['triage_gate'],
            'landing': landing_rate >= stage['landing_gate'],
            'lower_triage': lower_triage_rate >= stage['lower_triage_gate'],
            'collision': collision_rate <= stage['collision_rate_gate'],
            'acuity_priority': (
                acuity_priority >= stage['acuity_priority_gate']
            ),
            'priority_fairness': (
                priority_fairness >= stage['priority_fairness_gate']
            ),
            'triage_ordering': (
                triage_ordering >= stage['triage_ordering_gate']
            ),
            'response_ordering': (
                response_ordering >= stage['response_ordering_gate']
            ),
            'energy_safety': (
                battery_depletion_free
                >= stage['battery_depletion_free_gate']
                and reserve_violation <= stage['reserve_violation_gate']
            ),
            'wind_avoidance': (
                wind_avoidance >= stage['wind_avoidance_gate']
            ),
            'low_signal_avoidance': (
                low_signal_avoidance >= stage['low_signal_avoidance_gate']
            ),
            'hazard_evidence': hazard_evidence,
        }
        passed = bool(all(gate_checks.values()))
        
        
        
        
        
        
        foundational_gate_names = (
            'delivery', 'triage', 'landing', 'lower_triage',
            'collision', 'priority_fairness', 'response_ordering',
            'energy_safety',
        )
        foundational_passed = bool(all(
            gate_checks[name] for name in foundational_gate_names
        ))
        reached_maximum = bool(
            self.current_stage_episodes >= int(stage['maximum_episodes'])
        )
        self.consecutive_passes = (
            self.consecutive_passes + 1 if passed else 0
        )
        competence_promoted = bool(
            stage_before < len(CURRICULUM_STAGES) - 1
            and self.current_stage_episodes >= stage['minimum_episodes']
            and self.consecutive_passes >= CURRICULUM_REQUIRED_PASSES
        )
        maximum_promoted = bool(
            CURRICULUM_FORCE_PROMOTION_AT_MAXIMUM
            and stage_before < len(CURRICULUM_STAGES) - 1
            and reached_maximum
        )
        promoted = bool(competence_promoted or maximum_promoted)
        record = {
            'training_episode': int(training_episode),
            'stage_before': int(stage_before),
            'stage_name': stage['name'],
            'stage_training_episodes': int(self.current_stage_episodes),
            'delivery_rate': float(delivery_rate),
            'triage_rate': float(triage_rate),
            'landing_rate': float(landing_rate),
            'lower_triage_rate': float(lower_triage_rate),
            'collision_rate': float(collision_rate),
            'acuity_priority': float(acuity_priority),
            'priority_fairness': float(priority_fairness),
            'triage_ordering': float(triage_ordering),
            'response_ordering': float(response_ordering),
            'battery_depletion_free_rate': float(battery_depletion_free),
            'reserve_violation_rate': float(reserve_violation),
            'wind_avoidance': float(wind_avoidance),
            'low_signal_avoidance': float(low_signal_avoidance),
            'wind_avoidance_opportunities': int(wind_opportunities),
            'low_signal_avoidance_opportunities': int(
                low_signal_opportunities
            ),
            'hazard_evidence': hazard_evidence,
            'delivery_gate': float(stage['delivery_gate']),
            'triage_gate': float(stage['triage_gate']),
            'landing_gate': float(stage['landing_gate']),
            'lower_triage_gate': float(stage['lower_triage_gate']),
            'collision_rate_gate': float(stage['collision_rate_gate']),
            'acuity_priority_gate': float(stage['acuity_priority_gate']),
            'priority_fairness_gate': float(
                stage['priority_fairness_gate']
            ),
            'triage_ordering_gate': float(stage['triage_ordering_gate']),
            'response_ordering_gate': float(
                stage['response_ordering_gate']
            ),
            'battery_depletion_free_gate': float(
                stage['battery_depletion_free_gate']
            ),
            'reserve_violation_gate': float(
                stage['reserve_violation_gate']
            ),
            'wind_avoidance_gate': float(stage['wind_avoidance_gate']),
            'low_signal_avoidance_gate': float(
                stage['low_signal_avoidance_gate']
            ),
            'minimum_hazard_opportunities': int(
                stage['minimum_hazard_opportunities']
            ),
            'maximum_stage_episodes': int(stage['maximum_episodes']),
            'foundational_passed': foundational_passed,
            'reached_maximum': reached_maximum,
            'gate_checks': {
                name: bool(value) for name, value in gate_checks.items()
            },
            'failed_gates': [
                name for name, value in gate_checks.items() if not value
            ],
            'passed': passed,
            'consecutive_passes': int(self.consecutive_passes),
            'promoted': promoted,
            'promotion_reason': (
                'competence_gate' if competence_promoted
                else 'stage_episode_budget' if maximum_promoted
                else 'none'
            ),
        }
        if promoted:
            self.current_stage += 1
            self.current_stage_episodes = 0
            self.consecutive_passes = 0
            record['stage_after'] = int(self.current_stage)
        else:
            record['stage_after'] = int(stage_before)
        self.promotion_history.append(record)
        return record

    def state_dict(self):
        return {
            'current_stage': self.current_stage,
            'current_stage_episodes': self.current_stage_episodes,
            'stage_episode_counts': list(self.stage_episode_counts),
            'consecutive_passes': self.consecutive_passes,
            'promotion_history': list(self.promotion_history),
        }

    def load_state_dict(self, state):
        current_stage = int(state['current_stage'])
        if not 0 <= current_stage < len(CURRICULUM_STAGES):
            raise ValueError('Checkpoint contains an invalid curriculum stage')
        stage_counts = [int(value) for value in state['stage_episode_counts']]
        if len(stage_counts) != len(CURRICULUM_STAGES):
            raise ValueError('Checkpoint curriculum stage count is incompatible')
        self.current_stage = current_stage
        self.current_stage_episodes = int(state['current_stage_episodes'])
        self.stage_episode_counts = stage_counts
        self.consecutive_passes = int(state['consecutive_passes'])
        self.promotion_history = list(state.get('promotion_history', []))

def evaluate_policy(ctde_agent, max_steps, seed, num_episodes,
                    curriculum_stage=None):

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")

    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    was_training = ctde_agent.policy_net.training
    try:
        random.seed(seed)
        np.random.seed(seed)
        ctde_agent.policy_net.eval()
        rewards, delivered, died, landed, successes = [], [], [], [], []
        triage_efficiencies = []
        lower_triage_floors = []
        acuity_priority_scores = []
        delivery_jain_scores = []
        priority_fairness_scores = []
        triage_ordering_scores = []
        triage_rate_ordering_scores = []
        triage_response_ordering_scores = []
        class_response_ratios = {weight: [] for weight in (1, 2, 3)}
        class_response_times = {weight: [] for weight in (1, 2, 3)}
        w3_before_w1_response_fractions = []
        obstacle_collisions, agent_collisions, episode_lengths = [], [], []
        per_episode = []
        diagnostic_traces = []
        patient_attention_entropy = []
        patient_attention_max = []
        patient_attention_mass_by_weight = {
            weight: [] for weight in (1, 2, 3)
        }
        patient_attention_enrichment_by_weight = {
            weight: [] for weight in (1, 2, 3)
        }
        patient_attention_weighted_response_age = []
        drone_attention_entropy = []
        drone_attention_max = []
        action_query_similarity = []
        exact_q_tie_fractions = []
        q_action_gaps_by_phase = {'rescue': [], 'landing': []}

        for evaluation_index in range(num_episodes):
            
            
            
            evaluation_env = Environment(
                fixed_layout=False,
                episode_max_steps=max_steps
            )
            state = evaluation_env.reset(curriculum_stage=curriculum_stage)
            total_reward = 0.0
            collisions_obstacles = 0
            collisions_agents = 0
            steps = 0
            action_counts = np.zeros((NUM_AGENTS, ACTION_DIM), dtype=np.int64)
            collision_pair_matrix = np.zeros(
                (NUM_AGENTS, NUM_AGENTS), dtype=np.int64
            )
            collisions_by_agent = np.zeros(NUM_AGENTS, dtype=np.int64)
            obstacle_collisions_by_agent = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            obstacle_opportunities = np.zeros(NUM_AGENTS, dtype=np.int64)
            obstacle_actions_selected = np.zeros(NUM_AGENTS, dtype=np.int64)
            dominated_obstacle_selections = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            dominated_agent_conflict_selections = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            obstacle_recovery_action_counts = np.zeros(
                (NUM_AGENTS, ACTION_DIM), dtype=np.int64
            )
            previous_obstacle_flags = np.zeros(NUM_AGENTS, dtype=np.int64)
            obstacle_collision_events = []
            deliveries_by_agent = np.zeros(NUM_AGENTS, dtype=np.int64)
            local_reward_totals = np.zeros(NUM_AGENTS, dtype=np.float64)
            local_potential_reward_totals = np.zeros(
                NUM_AGENTS, dtype=np.float64
            )
            low_signal_failures = np.zeros(NUM_AGENTS, dtype=np.int64)
            wind_entries = np.zeros(NUM_AGENTS, dtype=np.int64)
            wind_exposure_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
            wind_exits = np.zeros(NUM_AGENTS, dtype=np.int64)
            wind_failures = np.zeros(NUM_AGENTS, dtype=np.int64)
            low_signal_entries = np.zeros(NUM_AGENTS, dtype=np.int64)
            low_signal_exposure_steps = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            low_signal_exits = np.zeros(NUM_AGENTS, dtype=np.int64)
            movement_actions = 0
            wind_command_attempts = 0
            low_signal_command_attempts = 0
            wind_avoidance_opportunities = 0
            wind_hazard_selections = 0
            wind_dominated_hazard_selections = 0
            wind_shortcut_hazard_selections = 0
            low_signal_avoidance_opportunities = 0
            low_signal_hazard_selections = 0
            low_signal_dominated_hazard_selections = 0
            low_signal_shortcut_hazard_selections = 0
            wind_entry_progress_cells = 0.0
            low_signal_entry_progress_cells = 0.0
            wind_zone_refreshes = 0
            low_signal_zone_refreshes = 0
            wind_refresh_onsets = np.zeros(NUM_AGENTS, dtype=np.int64)
            low_signal_refresh_onsets = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            operational_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
            reward_totals = {
                name: 0.0 for name in REWARD_COMPONENT_NAMES
            }
            phase_reward_totals = {
                phase: {
                    name: 0.0 for name in REWARD_COMPONENT_NAMES
                }
                for phase in ('rescue', 'landing', 'irrecoverable')
            }
            same_destination_collisions = 0
            head_on_collisions = 0
            collision_steps = 0
            obstacle_collision_steps = 0
            rescue_collisions = 0
            landing_collisions = 0
            maximum_collision_streak = 0
            maximum_agent_collision_streak = 0
            maximum_obstacle_collision_streak = 0
            minimum_agent_distance = 2 * GRID_SIZE
            valid_action_counts = []
            action_histories = [[] for _ in range(NUM_AGENTS)]
            phase_action_counts = {
                phase: np.zeros((NUM_AGENTS, ACTION_DIM), dtype=np.int64)
                for phase in ('rescue', 'landing', 'irrecoverable')
            }
            phase_obstacle_opportunities = {
                phase: np.zeros(NUM_AGENTS, dtype=np.int64)
                for phase in ('rescue', 'landing', 'irrecoverable')
            }
            phase_obstacle_actions_selected = {
                phase: np.zeros(NUM_AGENTS, dtype=np.int64)
                for phase in ('rescue', 'landing', 'irrecoverable')
            }
            phase_obstacle_collisions = {
                phase: np.zeros(NUM_AGENTS, dtype=np.int64)
                for phase in ('rescue', 'landing', 'irrecoverable')
            }
            phase_steps = {'rescue': 0, 'landing': 0, 'irrecoverable': 0}
            landing_progress_available = np.zeros(NUM_AGENTS, dtype=np.int64)
            landing_distance_reduced = np.zeros(NUM_AGENTS, dtype=np.int64)
            landing_distance_increased = np.zeros(NUM_AGENTS, dtype=np.int64)
            landing_distance_unchanged = np.zeros(NUM_AGENTS, dtype=np.int64)
            landing_hover_with_progress = np.zeros(NUM_AGENTS, dtype=np.int64)
            patient_events = []
            landing_events = []
            landing_zone_arrival_events = []
            landing_zone_departure_events = []
            landing_only_action_states = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            forced_terminal_landing_actions = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            landing_time_from_zone_arrival = [-1] * NUM_AGENTS
            battery_depletion_events = []
            dead_landing_events = []
            death_penalty_applications = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            death_reminder_penalty_applications = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            battery_drain = np.zeros(NUM_AGENTS, dtype=np.float64)
            wind_battery_drain = np.zeros(NUM_AGENTS, dtype=np.float64)
            landing_standby_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
            energy_return_mode_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
            energy_return_progress_steps = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            energy_return_regress_steps = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            energy_return_activations = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            previous_energy_return_flags = np.zeros(
                NUM_AGENTS, dtype=np.int64
            )
            energy_margin_delta = np.zeros(NUM_AGENTS, dtype=np.float64)
            reserve_violation_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
            minimum_safe_return_margin = np.full(
                NUM_AGENTS, np.inf, dtype=np.float64
            )
            potential_deltas = []
            raw_potential_changes = []
            episode_trace = []

            for step in range(max_steps + DEFAULT_LANDING_GRACE_STEPS):
                state_before = state
                positions_before = [
                    list(position) for position in evaluation_env.agents
                ]
                collect_attention = step % 25 == 0
                if collect_attention:
                    actions, policy_diagnostics = ctde_agent.select_actions(
                        state_before, epsilon=0.0, return_diagnostics=True
                    )
                else:
                    actions = ctde_agent.select_actions(
                        state_before, epsilon=0.0
                    )
                    policy_diagnostics = None
                valid_action_counts.extend(
                    state_before['action_masks'].sum(axis=1).tolist()
                )
                for agent_idx, action in enumerate(actions):
                    if not state_before['drones'][agent_idx, 3] \
                            and not state_before['drones'][agent_idx, 4]:
                        operational_steps[agent_idx] += 1
                    action_counts[agent_idx, action] += 1
                    action_histories[agent_idx].append(action)
                    if previous_obstacle_flags[agent_idx]:
                        obstacle_recovery_action_counts[agent_idx, action] += 1

                state, _, done, step_data = evaluation_env.step(actions)
                total_reward += step_data['team_reward']
                local_reward_totals += np.asarray(
                    step_data['local_rewards'], dtype=np.float64
                )
                local_potential_reward_totals += np.asarray(
                    step_data['local_potential_rewards'], dtype=np.float64
                )
                low_signal_failures += np.asarray(
                    step_data['low_signal_failures'], dtype=np.int64
                )
                wind_entries += np.asarray(
                    step_data['wind_entries'], dtype=np.int64
                )
                wind_exposure_steps += np.asarray(
                    step_data['wind_exposure_steps'], dtype=np.int64
                )
                wind_exits += np.asarray(
                    step_data['wind_exits'], dtype=np.int64
                )
                wind_failures += np.asarray(
                    step_data['wind_failures'], dtype=np.int64
                )
                low_signal_entries += np.asarray(
                    step_data['low_signal_entries'], dtype=np.int64
                )
                low_signal_exposure_steps += np.asarray(
                    step_data['low_signal_exposure_steps'], dtype=np.int64
                )
                low_signal_exits += np.asarray(
                    step_data['low_signal_exits'], dtype=np.int64
                )
                movement_actions += int(step_data['movement_actions'])
                wind_command_attempts += int(
                    step_data['wind_command_attempts']
                )
                low_signal_command_attempts += int(
                    step_data['low_signal_command_attempts']
                )
                wind_avoidance_opportunities += int(
                    step_data['wind_avoidance_opportunities']
                )
                wind_hazard_selections += int(
                    step_data['wind_hazard_selections']
                )
                wind_dominated_hazard_selections += int(
                    step_data['wind_dominated_hazard_selections']
                )
                wind_shortcut_hazard_selections += int(
                    step_data['wind_shortcut_hazard_selections']
                )
                low_signal_avoidance_opportunities += int(
                    step_data['low_signal_avoidance_opportunities']
                )
                low_signal_hazard_selections += int(
                    step_data['low_signal_hazard_selections']
                )
                low_signal_dominated_hazard_selections += int(
                    step_data['low_signal_dominated_hazard_selections']
                )
                low_signal_shortcut_hazard_selections += int(
                    step_data['low_signal_shortcut_hazard_selections']
                )
                wind_entry_progress_cells += float(
                    step_data['wind_entry_progress_cells']
                )
                low_signal_entry_progress_cells += float(
                    step_data['low_signal_entry_progress_cells']
                )
                wind_zone_refreshes += int(step_data['wind_zone_refreshed'])
                low_signal_zone_refreshes += int(
                    step_data['low_signal_zone_refreshed']
                )
                wind_refresh_onsets += np.asarray(
                    step_data['wind_refresh_onset_agents'], dtype=np.int64
                )
                low_signal_refresh_onsets += np.asarray(
                    step_data['low_signal_refresh_onset_agents'], dtype=np.int64
                )
                potential_deltas.append(step_data['potential_delta'])
                raw_potential_changes.append(step_data['raw_potential_change'])
                battery_depletion_events.extend(
                    step_data['battery_depletion_events']
                )
                dead_landing_events.extend(step_data['dead_landing_events'])
                death_penalty_applications += np.asarray(
                    step_data['death_penalty_applications'], dtype=np.int64
                )
                death_reminder_penalty_applications += np.asarray(
                    step_data[
                        'death_reminder_penalty_applications'
                    ], dtype=np.int64
                )
                step_battery_drain = np.asarray(
                    step_data['battery_drain_by_agent'], dtype=np.float64
                )
                battery_drain += step_battery_drain
                wind_battery_drain += np.asarray(
                    step_data['wind_battery_drain_by_agent'], dtype=np.float64
                )
                landing_standby_steps += np.asarray(
                    step_data['landing_standby_steps'], dtype=np.int64
                )
                energy_return_flags = np.asarray(
                    step_data['energy_return_mode_flags'], dtype=np.int64
                )
                energy_return_mode_steps += energy_return_flags
                energy_return_progress_steps += np.asarray(
                    step_data['energy_return_progress_flags'], dtype=np.int64
                )
                energy_return_regress_steps += np.asarray(
                    step_data['energy_return_regress_flags'], dtype=np.int64
                )
                energy_return_activations += (
                    (energy_return_flags > 0)
                    & (previous_energy_return_flags == 0)
                ).astype(np.int64)
                previous_energy_return_flags = energy_return_flags
                energy_margin_delta += np.asarray(
                    step_data['energy_margin_delta_by_agent'],
                    dtype=np.float64,
                )
                reserve_violation_steps += np.asarray(
                    step_data['reserve_violation_flags'], dtype=np.int64
                )
                operational_battery_mask = step_battery_drain > 0.0
                step_safe_return_margins = np.asarray(
                    step_data['safe_return_margin_after'], dtype=np.float64
                )
                minimum_safe_return_margin[
                    operational_battery_mask
                ] = np.minimum(
                    minimum_safe_return_margin[operational_battery_mask],
                    step_safe_return_margins[operational_battery_mask],
                )
                landing_events.extend(step_data['landing_events'])
                landing_zone_arrival_events.extend(
                    step_data['landing_zone_arrival_events']
                )
                landing_zone_departure_events.extend(
                    step_data['landing_zone_departure_events']
                )
                landing_only_action_states += np.asarray(
                    step_data['landing_only_action_flags'], dtype=np.int64
                )
                forced_terminal_landing_actions += np.asarray(
                    step_data['forced_terminal_landing_actions'],
                    dtype=np.int64
                )
                for landing_event in step_data['landing_events']:
                    if landing_event['successful']:
                        landing_time_from_zone_arrival[
                            landing_event['agent']
                        ] = landing_event['steps_from_zone_arrival']
                collisions_obstacles += step_data['obstacle_collisions']
                obstacle_collision_steps += int(
                    step_data['obstacle_collisions'] > 0
                )
                obstacle_opportunities += np.asarray(
                    step_data['obstacle_action_opportunities'], dtype=np.int64
                )
                obstacle_actions_selected += np.asarray(
                    step_data['obstacle_action_selected'], dtype=np.int64
                )
                dominated_obstacle_selections += np.asarray(
                    step_data['dominated_obstacle_selections'], dtype=np.int64
                )
                dominated_agent_conflict_selections += np.asarray(
                    step_data['dominated_agent_conflict_selections'],
                    dtype=np.int64,
                )
                obstacle_collisions_by_agent += np.asarray(
                    step_data['obstacle_collision_flags'], dtype=np.int64
                )
                previous_obstacle_flags = np.asarray(
                    step_data['obstacle_collision_flags'], dtype=np.int64
                )
                remaining_obstacle_event_slots = max(
                    0,
                    MAX_RECORDED_OBSTACLE_EVENTS_PER_EPISODE
                    - len(obstacle_collision_events),
                )
                if remaining_obstacle_event_slots:
                    obstacle_collision_events.extend(
                        step_data['obstacle_collision_events'][
                            :remaining_obstacle_event_slots
                        ]
                    )
                collisions_agents += step_data['agent_collisions']
                same_destination_collisions += step_data[
                    'same_destination_collisions'
                ]
                head_on_collisions += step_data['head_on_collisions']
                collision_steps += int(step_data['agent_collisions'] > 0)
                if step_data['phase_before'] == 'rescue':
                    rescue_collisions += step_data['agent_collisions']
                elif step_data['phase_before'] == 'landing':
                    landing_collisions += step_data['agent_collisions']
                maximum_collision_streak = max(
                    maximum_collision_streak,
                    step_data['max_collision_streak']
                )
                maximum_agent_collision_streak = max(
                    maximum_agent_collision_streak,
                    step_data['max_agent_collision_streak']
                )
                maximum_obstacle_collision_streak = max(
                    maximum_obstacle_collision_streak,
                    step_data['max_obstacle_collision_streak']
                )
                phase_before = step_data['phase_before']
                phase_steps[phase_before] += 1
                for agent_idx, action in enumerate(actions):
                    phase_action_counts[
                        phase_before
                    ][agent_idx, action] += 1
                phase_obstacle_opportunities[phase_before] += np.asarray(
                    step_data['obstacle_action_opportunities'], dtype=np.int64
                )
                phase_obstacle_actions_selected[phase_before] += np.asarray(
                    step_data['obstacle_action_selected'], dtype=np.int64
                )
                phase_obstacle_collisions[phase_before] += np.asarray(
                    step_data['obstacle_collision_flags'], dtype=np.int64
                )
                landing_progress_available += np.asarray(
                    step_data['landing_progress_actions_available'],
                    dtype=np.int64
                )
                landing_distance_reduced += np.asarray(
                    step_data['landing_distance_reduced'], dtype=np.int64
                )
                landing_distance_increased += np.asarray(
                    step_data['landing_distance_increased'], dtype=np.int64
                )
                landing_distance_unchanged += np.asarray(
                    step_data['landing_distance_unchanged'], dtype=np.int64
                )
                landing_hover_with_progress += np.asarray(
                    step_data['landing_hover_with_progress_available'],
                    dtype=np.int64
                )
                if step_data['minimum_agent_distance'] >= 0:
                    minimum_agent_distance = min(
                        minimum_agent_distance,
                        step_data['minimum_agent_distance']
                    )
                for pair in step_data['collision_pairs']:
                    collision_pair_matrix[pair[0], pair[1]] += 1
                    collision_pair_matrix[pair[1], pair[0]] += 1
                collisions_by_agent += np.asarray(
                    step_data['agent_collision_flags'], dtype=np.int64
                )
                for event in step_data['patient_delivery_events']:
                    deliveries_by_agent[event['agent']] += 1
                    patient_events.append({'type': 'delivery', **event})
                for event in step_data['patient_death_events']:
                    patient_events.append({'type': 'death', **event})
                for event in step_data['patient_weight_escalation_events']:
                    patient_events.append({
                        'type': 'triage_escalation', **event
                    })
                for patient_index in step_data['patient_spawn_events']:
                    patient_events.append({
                        'type': 'spawn',
                        'patient': patient_index,
                        'step': evaluation_env.episode_step,
                    })
                for component_name, component_value in (
                        step_data['reward_components'].items()):
                    reward_totals[component_name] += component_value
                    phase_reward_totals[phase_before][
                        component_name
                    ] += component_value

                important_event = bool(
                    step_data['patient_delivery_events']
                    or step_data['patient_death_events']
                    or step_data['patient_weight_escalation_events']
                    or step_data['collision_pairs']
                    or step_data['obstacle_collision_events']
                    or step_data['landing_zone_arrival_events']
                    or step_data['landing_zone_departure_events']
                    or step_data['landing_events']
                    or done
                )
                if important_event and policy_diagnostics is None:
                    _, policy_diagnostics = ctde_agent.select_actions(
                        state_before, epsilon=0.0, return_diagnostics=True
                    )
                if policy_diagnostics is not None:
                    patient_attention_entropy.append(
                        policy_diagnostics['patient_attention_entropy']
                    )
                    patient_attention_max.append(
                        policy_diagnostics['patient_attention_max']
                    )
                    for weight in (1, 2, 3):
                        patient_attention_mass_by_weight[weight].append(
                            policy_diagnostics[
                                'patient_attention_mass_by_initial_weight'
                            ][weight]
                        )
                        patient_attention_enrichment_by_weight[weight].append(
                            policy_diagnostics[
                                'patient_attention_enrichment_by_initial_weight'
                            ][weight]
                        )
                    patient_attention_weighted_response_age.append(
                        policy_diagnostics[
                            'patient_attention_weighted_response_age'
                        ]
                    )
                    drone_attention_entropy.append(
                        policy_diagnostics['drone_attention_entropy']
                    )
                    drone_attention_max.append(
                        policy_diagnostics['drone_attention_max']
                    )
                    action_query_similarity.append(
                        policy_diagnostics['action_query_similarity']
                    )
                    exact_q_tie_fractions.append(
                        policy_diagnostics['exact_q_tie_fraction']
                    )
                    q_action_gaps_by_phase[
                        step_data['phase_before']
                    ].extend(policy_diagnostics['action_gap'])
                if (evaluation_index < EVALUATION_TRACE_EPISODES
                        and (collect_attention or important_event)):
                    trace_record = {
                        'step': step + 1,
                        'positions_before': positions_before,
                        'positions': [list(position) for position in evaluation_env.agents],
                        'actions': actions,
                        'action_names': [ACTION_NAMES[action] for action in actions],
                        'action_masks': state_before['action_masks'].tolist(),
                        'batteries': [
                            float(battery) for battery in evaluation_env.batteries
                        ],
                        'wind_rectangles': [
                            list(rectangle)
                            for rectangle in evaluation_env.wind_rectangles
                        ],
                        'low_signal_rectangles': [
                            list(rectangle)
                            for rectangle in evaluation_env.low_signal_rectangles
                        ],
                        'pending_patients': int(sum(
                            evaluation_env.patient_active[p]
                            and not evaluation_env.patients_delivered[p]
                            for p in range(MAX_PATIENTS)
                        )),
                        'phase': step_data['phase_after'],
                        'collision_pairs': step_data['collision_pairs'],
                        'obstacle_collisions': step_data[
                            'obstacle_collision_events'
                        ],
                        'deliveries': step_data['patient_delivery_events'],
                        'deaths': step_data['patient_death_events'],
                        'triage_escalations': step_data[
                            'patient_weight_escalation_events'
                        ],
                        'landings': step_data['landing_events'],
                        'landing_zone_arrivals': step_data[
                            'landing_zone_arrival_events'
                        ],
                        'landing_zone_departures': step_data[
                            'landing_zone_departure_events'
                        ],
                        'landing_only_action_flags': step_data[
                            'landing_only_action_flags'
                        ],
                        'forced_terminal_landing_actions': step_data[
                            'forced_terminal_landing_actions'
                        ],
                        'battery_depletions': step_data[
                            'battery_depletion_events'
                        ],
                        'hazard_decisions': {
                            'wind_opportunities': step_data[
                                'wind_avoidance_opportunities'
                            ],
                            'wind_selections': step_data[
                                'wind_hazard_selections'
                            ],
                            'wind_dominated': step_data[
                                'wind_dominated_hazard_selections'
                            ],
                            'wind_shortcuts': step_data[
                                'wind_shortcut_hazard_selections'
                            ],
                            'low_signal_opportunities': step_data[
                                'low_signal_avoidance_opportunities'
                            ],
                            'low_signal_selections': step_data[
                                'low_signal_hazard_selections'
                            ],
                            'low_signal_dominated': step_data[
                                'low_signal_dominated_hazard_selections'
                            ],
                            'low_signal_shortcuts': step_data[
                                'low_signal_shortcut_hazard_selections'
                            ],
                        },
                        'team_reward': step_data['team_reward'],
                        'local_rewards': step_data['local_rewards'],
                        'local_potential_rewards': step_data[
                            'local_potential_rewards'
                        ],
                        'reward_components': step_data['reward_components'],
                        'potential_before': step_data['potential_before'],
                        'potential_after': step_data['potential_after'],
                        'potential_delta': step_data['potential_delta'],
                        'raw_potential_change': step_data[
                            'raw_potential_change'
                        ],
                        'pending_patient_snapshot': [
                            {
                                'patient': patient_index,
                                'position': list(
                                    evaluation_env.patient_positions[patient_index]
                                ),
                                'timer': evaluation_env.patient_timers[patient_index],
                                'initial_timer': (
                                    evaluation_env.patient_initial_timers[
                                        patient_index
                                    ]
                                ),
                                'current_weight': evaluation_env.patient_weights[
                                    patient_index
                                ],
                                'initial_weight': (
                                    evaluation_env.initial_patient_weights[
                                        patient_index
                                    ]
                                ),
                                'elapsed_response_fraction': float(np.clip(
                                    (
                                        evaluation_env.episode_step
                                        - evaluation_env.patient_spawn_steps[
                                            patient_index
                                        ]
                                    ) / max(
                                        1,
                                        evaluation_env.patient_initial_timers[
                                            patient_index
                                        ],
                                    ),
                                    0.0,
                                    1.0,
                                )),
                            }
                            for patient_index in range(MAX_PATIENTS)
                            if evaluation_env.patient_active[patient_index]
                            and not evaluation_env.patients_delivered[patient_index]
                        ],
                        'q_values': policy_diagnostics['q_values'],
                        'action_gap': policy_diagnostics['action_gap'],
                        'exact_top_tie': policy_diagnostics['exact_top_tie'],
                        'top_tie_count': policy_diagnostics['top_tie_count'],
                        'q_value_dtype': policy_diagnostics['q_value_dtype'],
                        'patient_top_indices': policy_diagnostics[
                            'patient_top_indices'
                        ],
                        'patient_top_weights': policy_diagnostics[
                            'patient_top_weights'
                        ],
                        'patient_attention_mass_by_initial_weight': (
                            policy_diagnostics[
                                'patient_attention_mass_by_initial_weight'
                            ]
                        ),
                        'patient_attention_enrichment_by_initial_weight': (
                            policy_diagnostics[
                                'patient_attention_enrichment_by_initial_weight'
                            ]
                        ),
                        'patient_attention_weighted_response_age': (
                            policy_diagnostics[
                                'patient_attention_weighted_response_age'
                            ]
                        ),
                        'drone_top_indices': policy_diagnostics[
                            'drone_top_indices'
                        ],
                        'drone_top_weights': policy_diagnostics[
                            'drone_top_weights'
                        ],
                    }
                    if len(episode_trace) < EVALUATION_TRACE_MAX_RECORDS:
                        episode_trace.append(trace_record)
                    elif done:
                        
                        
                        episode_trace[-1] = trace_record
                steps = step + 1
                if done:
                    break

            rewards.append(total_reward)
            delivered.append(sum(evaluation_env.patients_actually_delivered))
            died.append(sum(evaluation_env.patients_died))
            landed.append(sum(evaluation_env.landed))
            successes.append(1 if evaluation_env.mission_success() else 0)
            mission_outcome = evaluation_env.mission_outcome_metrics()
            triage_efficiencies.append(mission_outcome['triage_efficiency'])
            lower_triage_floors.append(
                mission_outcome['lower_triage_delivery_floor']
            )
            acuity_priority_scores.append(
                mission_outcome['acuity_priority_score']
            )
            delivery_jain_scores.append(
                mission_outcome['class_delivery_jain_fairness']
            )
            priority_fairness_scores.append(
                mission_outcome['priority_fairness_attainment']
            )
            triage_ordering_scores.append(
                mission_outcome['triage_delivery_ordering_score']
            )
            triage_rate_ordering_scores.append(
                mission_outcome['triage_delivery_rate_ordering_score']
            )
            triage_response_ordering_scores.append(
                mission_outcome['triage_response_time_ordering_score']
            )
            for weight in (1, 2, 3):
                class_response_ratios[weight].append(
                    mission_outcome[f'mean_response_ratio_w{weight}']
                )
                class_response_times[weight].extend([
                    evaluation_env.patient_resolution_steps[patient_index]
                    - evaluation_env.patient_spawn_steps[patient_index]
                    for patient_index in range(MAX_PATIENTS)
                    if evaluation_env.patient_active[patient_index]
                    and evaluation_env.patients_actually_delivered[patient_index]
                    and evaluation_env.initial_patient_weights[
                        patient_index
                    ] == weight
                ])
            w3_before_w1_response_fractions.append(
                mission_outcome['w3_before_w1_response_fraction']
            )
            obstacle_collisions.append(collisions_obstacles)
            agent_collisions.append(collisions_agents)
            episode_lengths.append(steps)
            action_switches = []
            two_action_oscillations = []
            for history in action_histories:
                action_switches.append(sum(
                    history[index] != history[index - 1]
                    for index in range(1, len(history))
                ))
                two_action_oscillations.append(sum(
                    history[index] == history[index - 2]
                    and history[index] != history[index - 1]
                    for index in range(2, len(history))
                ))
            episode_record = {
                'evaluation_index': evaluation_index,
                'suite_seed': seed,
                'scenario_index': evaluation_index,
                'curriculum_stage': int(
                    evaluation_env.curriculum_stage
                ),
                'curriculum_stage_name': (
                    evaluation_env.curriculum_stage_name
                ),
                'curriculum_max_distance': int(
                    evaluation_env.curriculum_max_distance
                ),
                'curriculum_max_landing_distance': int(
                    evaluation_env.curriculum_max_landing_distance
                ),
                'hazard_penalty_scale': float(
                    evaluation_env.hazard_penalty_scale
                ),
                'episode_patient_timer': int(
                    evaluation_env.episode_patient_timer
                ),
                'patient_spawn_interval': int(
                    evaluation_env.patient_spawn_interval
                ),
                'patient_spawn_jitter': int(
                    evaluation_env.patient_spawn_jitter
                ),
                'patient_spawn_batch_range': [
                    int(evaluation_env.minimum_patient_spawn_batch),
                    int(evaluation_env.maximum_patient_spawn_batch),
                ],
                'final_patient_spawn_step': int(
                    evaluation_env.final_patient_spawn_step
                ),
                'obstacles': [
                    list(position) for position in sorted(evaluation_env.obstacles)
                ],
                'start_positions': [
                    list(position) for position in evaluation_env.start_positions
                ],
                'patient_positions': [
                    list(position) for position in evaluation_env.patient_positions
                ],
                'landing_zones': [
                    list(position) for position in evaluation_env.landing_zones
                ],
                'reward': float(total_reward),
                'local_reward_totals': local_reward_totals.tolist(),
                'local_potential_reward_totals': (
                    local_potential_reward_totals.tolist()
                ),
                'local_reward_sum_error': float(
                    local_reward_totals.sum() - total_reward
                ),
                'delivered': int(delivered[-1]),
                'died': int(died[-1]),
                'landed': int(landed[-1]),
                'success': int(successes[-1]),
                'triage_efficiency': float(triage_efficiencies[-1]),
                'mission_outcome': mission_outcome,
                'perfect_rescue': bool(evaluation_env.perfect_rescue()),
                'steps': steps,
                'termination_reason': evaluation_env.termination_reason,
                'obstacle_collisions': int(collisions_obstacles),
                'obstacle_collision_steps': int(obstacle_collision_steps),
                'obstacle_action_opportunities_by_agent': (
                    obstacle_opportunities.tolist()
                ),
                'obstacle_actions_selected_by_agent': (
                    obstacle_actions_selected.tolist()
                ),
                'obstacle_collisions_by_agent': (
                    obstacle_collisions_by_agent.tolist()
                ),
                'obstacle_recovery_action_counts': (
                    obstacle_recovery_action_counts.tolist()
                ),
                'obstacle_collision_events': obstacle_collision_events,
                'agent_collisions': int(collisions_agents),
                'same_destination_collisions': int(same_destination_collisions),
                'head_on_collisions': int(head_on_collisions),
                'collision_steps': int(collision_steps),
                'rescue_collisions': int(rescue_collisions),
                'landing_collisions': int(landing_collisions),
                'maximum_collision_streak': int(maximum_collision_streak),
                'maximum_agent_collision_streak': int(
                    maximum_agent_collision_streak
                ),
                'maximum_obstacle_collision_streak': int(
                    maximum_obstacle_collision_streak
                ),
                'minimum_agent_distance': (
                    int(minimum_agent_distance)
                    if minimum_agent_distance < 2 * GRID_SIZE else -1
                ),
                'collisions_by_agent': collisions_by_agent.tolist(),
                'collision_pair_matrix': collision_pair_matrix.tolist(),
                'deliveries_by_agent': deliveries_by_agent.tolist(),
                'action_counts': action_counts.tolist(),
                'phase_action_counts': {
                    phase: counts.tolist()
                    for phase, counts in phase_action_counts.items()
                },
                'phase_obstacle_action_opportunities_by_agent': {
                    phase: counts.tolist()
                    for phase, counts in phase_obstacle_opportunities.items()
                },
                'phase_obstacle_actions_selected_by_agent': {
                    phase: counts.tolist()
                    for phase, counts in phase_obstacle_actions_selected.items()
                },
                'phase_obstacle_collisions_by_agent': {
                    phase: counts.tolist()
                    for phase, counts in phase_obstacle_collisions.items()
                },
                'phase_steps': phase_steps,
                'landing_progress_actions_available_by_agent': (
                    landing_progress_available.tolist()
                ),
                'landing_distance_reduced_by_agent': (
                    landing_distance_reduced.tolist()
                ),
                'landing_distance_increased_by_agent': (
                    landing_distance_increased.tolist()
                ),
                'landing_distance_unchanged_by_agent': (
                    landing_distance_unchanged.tolist()
                ),
                'landing_hover_with_progress_available_by_agent': (
                    landing_hover_with_progress.tolist()
                ),
                'action_switches': action_switches,
                'two_action_oscillations': two_action_oscillations,
                'operational_steps_by_agent': operational_steps.tolist(),
                'operational_agent_steps': int(operational_steps.sum()),
                'movement_actions': int(movement_actions),
                'wind_command_attempts': int(wind_command_attempts),
                'low_signal_command_attempts': int(
                    low_signal_command_attempts
                ),
                'obstacle_collision_rate_per_operational_step': float(
                    collisions_obstacles / max(1, operational_steps.sum())
                ),
                'agent_collision_rate_per_operational_step': float(
                    collisions_agents / max(1, operational_steps.sum())
                ),
                'wind_entries_by_agent': wind_entries.tolist(),
                'wind_exposure_steps_by_agent': (
                    wind_exposure_steps.tolist()
                ),
                'wind_exits_by_agent': wind_exits.tolist(),
                'wind_failures_by_agent': wind_failures.tolist(),
                'low_signal_entries_by_agent': low_signal_entries.tolist(),
                'low_signal_exposure_steps_by_agent': (
                    low_signal_exposure_steps.tolist()
                ),
                'low_signal_exits_by_agent': low_signal_exits.tolist(),
                'low_signal_failures_by_agent': low_signal_failures.tolist(),
                'wind_exposure_rate': float(
                    wind_exposure_steps.sum()
                    / max(1, operational_steps.sum())
                ),
                'low_signal_exposure_rate': float(
                    low_signal_exposure_steps.sum()
                    / max(1, operational_steps.sum())
                ),
                'wind_failure_rate': float(
                    wind_failures.sum() / max(1, wind_command_attempts)
                ),
                'low_signal_failure_rate': float(
                    low_signal_failures.sum()
                    / max(1, low_signal_command_attempts)
                ),
                'wind_avoidance_opportunities': int(
                    wind_avoidance_opportunities
                ),
                'wind_hazard_selections': int(wind_hazard_selections),
                'wind_dominated_hazard_selections': int(
                    wind_dominated_hazard_selections
                ),
                'wind_shortcut_hazard_selections': int(
                    wind_shortcut_hazard_selections
                ),
                'wind_avoidance_rate': float(
                    1.0 - wind_hazard_selections
                    / max(1, wind_avoidance_opportunities)
                ),
                'wind_dominated_avoidance_rate': float(
                    1.0 - wind_dominated_hazard_selections
                    / max(1, wind_avoidance_opportunities)
                ),
                'wind_rational_entry_fraction': float(
                    wind_shortcut_hazard_selections
                    / max(1, wind_hazard_selections)
                ),
                'low_signal_avoidance_opportunities': int(
                    low_signal_avoidance_opportunities
                ),
                'low_signal_hazard_selections': int(
                    low_signal_hazard_selections
                ),
                'low_signal_dominated_hazard_selections': int(
                    low_signal_dominated_hazard_selections
                ),
                'low_signal_shortcut_hazard_selections': int(
                    low_signal_shortcut_hazard_selections
                ),
                'low_signal_avoidance_rate': float(
                    1.0 - low_signal_hazard_selections
                    / max(1, low_signal_avoidance_opportunities)
                ),
                'low_signal_dominated_avoidance_rate': float(
                    1.0 - low_signal_dominated_hazard_selections
                    / max(1, low_signal_avoidance_opportunities)
                ),
                'low_signal_rational_entry_fraction': float(
                    low_signal_shortcut_hazard_selections
                    / max(1, low_signal_hazard_selections)
                ),
                'wind_entry_progress_cells': float(
                    wind_entry_progress_cells
                ),
                'low_signal_entry_progress_cells': float(
                    low_signal_entry_progress_cells
                ),
                'wind_progress_cells_per_entry': float(
                    wind_entry_progress_cells / max(1, wind_entries.sum())
                ),
                'low_signal_progress_cells_per_entry': float(
                    low_signal_entry_progress_cells
                    / max(1, low_signal_entries.sum())
                ),
                'wind_zone_refreshes': int(wind_zone_refreshes),
                'low_signal_zone_refreshes': int(
                    low_signal_zone_refreshes
                ),
                'wind_refresh_onsets_by_agent': (
                    wind_refresh_onsets.tolist()
                ),
                'low_signal_refresh_onsets_by_agent': (
                    low_signal_refresh_onsets.tolist()
                ),
                'mean_valid_actions': float(np.mean(valid_action_counts)),
                'agent_path_lengths': evaluation_env.agent_path_lengths,
                'agent_unique_cells': [
                    len(positions)
                    for positions in evaluation_env.agent_unique_positions
                ],
                'first_delivery_step': evaluation_env.first_delivery_step,
                'last_delivery_step': evaluation_env.last_delivery_step,
                'all_patients_resolved_step': (
                    evaluation_env.all_patients_resolved_step
                ),
                'irrecoverable_step': evaluation_env.irrecoverable_step,
                'patient_spawn_steps': evaluation_env.patient_spawn_steps,
                'patient_resolution_steps': evaluation_env.patient_resolution_steps,
                'patient_initial_timers': (
                    evaluation_env.patient_initial_timers
                ),
                'patient_time_to_resolution_ratio': [
                    (
                        (evaluation_env.patient_resolution_steps[patient_index]
                         - evaluation_env.patient_spawn_steps[patient_index])
                        / max(
                            1,
                            evaluation_env.patient_initial_timers[
                                patient_index
                            ],
                        )
                        if evaluation_env.patient_spawn_steps[patient_index] >= 0
                        and evaluation_env.patient_resolution_steps[
                            patient_index
                        ] >= 0 else -1.0
                    )
                    for patient_index in range(MAX_PATIENTS)
                ],
                'patient_delivery_agents': evaluation_env.patient_delivery_agents,
                'initial_patient_weights': evaluation_env.initial_patient_weights,
                'final_patient_weights': evaluation_env.patient_weights,
                'patient_survival_probabilities': (
                    evaluation_env.patient_survival_probabilities
                ),
                'patient_decay_rates': evaluation_env.patient_decay_rates,
                'patient_survival_offsets': (
                    evaluation_env.patient_survival_offsets
                ),
                'patient_serious_thresholds': (
                    evaluation_env.patient_serious_thresholds
                ),
                'patient_critical_thresholds': (
                    evaluation_env.patient_critical_thresholds
                ),
                'wind_rectangles': [
                    list(value) for value in evaluation_env.wind_rectangles
                ],
                'low_signal_rectangles': [
                    list(value)
                    for value in evaluation_env.low_signal_rectangles
                ],
                'hazard_route_challenges': [
                    dict(value)
                    for value in evaluation_env.hazard_route_challenges
                ],
                'triage_service_state': (
                    evaluation_env.triage_service_state()
                ),
                'dominated_obstacle_selections_by_agent': (
                    dominated_obstacle_selections.tolist()
                ),
                'dominated_agent_conflict_selections_by_agent': (
                    dominated_agent_conflict_selections.tolist()
                ),
                'patient_time_to_resolution': [
                    (
                        evaluation_env.patient_resolution_steps[patient_index]
                        - evaluation_env.patient_spawn_steps[patient_index]
                        if evaluation_env.patient_spawn_steps[patient_index] >= 0
                        and evaluation_env.patient_resolution_steps[patient_index] >= 0
                        else -1
                    )
                    for patient_index in range(MAX_PATIENTS)
                ],
                'patient_events': patient_events,
                'landing_events': landing_events,
                'landing_zone_arrival_events': landing_zone_arrival_events,
                'landing_zone_departure_events': (
                    landing_zone_departure_events
                ),
                'landing_only_action_states_by_agent': (
                    landing_only_action_states.tolist()
                ),
                'forced_terminal_landing_actions_by_agent': (
                    forced_terminal_landing_actions.tolist()
                ),
                'landing_time_from_zone_arrival_by_agent': (
                    landing_time_from_zone_arrival
                ),
                'battery_depletion_events': battery_depletion_events,
                'dead_landing_events': dead_landing_events,
                'death_penalty_applications_by_agent': (
                    death_penalty_applications.tolist()
                ),
                'death_reminder_penalty_applications_by_agent': (
                    death_reminder_penalty_applications.tolist()
                ),
                'battery_drain_by_agent': battery_drain.tolist(),
                'wind_battery_drain_by_agent': wind_battery_drain.tolist(),
                'landing_standby_steps_by_agent': (
                    landing_standby_steps.tolist()
                ),
                'energy_return_mode_steps_by_agent': (
                    energy_return_mode_steps.tolist()
                ),
                'energy_return_progress_steps_by_agent': (
                    energy_return_progress_steps.tolist()
                ),
                'energy_return_regress_steps_by_agent': (
                    energy_return_regress_steps.tolist()
                ),
                'energy_return_activations_by_agent': (
                    energy_return_activations.tolist()
                ),
                'energy_margin_delta_by_agent': energy_margin_delta.tolist(),
                'reserve_violation_steps_by_agent': (
                    reserve_violation_steps.tolist()
                ),
                'minimum_safe_return_margin_by_agent': [
                    float(value) if math.isfinite(value) else 0.0
                    for value in minimum_safe_return_margin
                ],
                'total_battery_drain': float(battery_drain.sum()),
                'total_wind_battery_drain': float(
                    wind_battery_drain.sum()
                ),
                'wind_energy_fraction': float(
                    wind_battery_drain.sum()
                    / max(1e-8, battery_drain.sum())
                ),
                'battery_units_per_delivery': float(
                    battery_drain.sum()
                    / max(1, mission_outcome['delivered_count'])
                ),
                'weighted_deliveries_per_battery_unit': float(
                    mission_outcome['weighted_delivered']
                    / max(1e-8, battery_drain.sum())
                ),
                'energy_return_mode_fraction': float(
                    energy_return_mode_steps.sum()
                    / max(1, operational_steps.sum())
                ),
                'energy_return_progress_rate': float(
                    energy_return_progress_steps.sum()
                    / max(1, energy_return_mode_steps.sum())
                ),
                'energy_return_regress_rate': float(
                    energy_return_regress_steps.sum()
                    / max(1, energy_return_mode_steps.sum())
                ),
                'energy_return_success_rate': float(
                    sum(
                        energy_return_activations[agent_index] > 0
                        and evaluation_env.landed[agent_index]
                        for agent_index in range(NUM_AGENTS)
                    ) / max(
                        1,
                        sum(energy_return_activations > 0),
                    )
                ),
                'mean_landing_battery': float(np.mean([
                    evaluation_env.batteries[agent_index]
                    for agent_index in range(NUM_AGENTS)
                    if evaluation_env.landed[agent_index]
                ])) if any(evaluation_env.landed) else 0.0,
                'minimum_landing_battery': float(min([
                    evaluation_env.batteries[agent_index]
                    for agent_index in range(NUM_AGENTS)
                    if evaluation_env.landed[agent_index]
                ])) if any(evaluation_env.landed) else 0.0,
                'reserve_violation_rate': float(
                    reserve_violation_steps.sum()
                    / max(1, operational_steps.sum())
                ),
                'potential_delta_mean': float(np.mean(potential_deltas)),
                'potential_delta_min': float(np.min(potential_deltas)),
                'potential_delta_max': float(np.max(potential_deltas)),
                'positive_potential_steps': int(sum(
                    delta > 0.0 for delta in potential_deltas
                )),
                'negative_potential_steps': int(sum(
                    delta < 0.0 for delta in potential_deltas
                )),
                'raw_potential_change_mean': float(np.mean(
                    raw_potential_changes
                )),
                'raw_potential_change_min': float(np.min(
                    raw_potential_changes
                )),
                'raw_potential_change_max': float(np.max(
                    raw_potential_changes
                )),
                'raw_progress_steps': int(sum(
                    delta > 1e-8 for delta in raw_potential_changes
                )),
                'final_potential_components': (
                    evaluation_env.fleet_potential_components()
                ),
                'final_batteries': [
                    float(battery) for battery in evaluation_env.batteries
                ],
                'final_landed': [
                    bool(value) for value in evaluation_env.landed
                ],
                'final_battery_depleted': [
                    bool(value) for value in evaluation_env.battery_depleted
                ],
                'final_drone_died': [
                    bool(value) for value in evaluation_env.drone_died
                ],
                'reward_components': {
                    name: float(value) for name, value in reward_totals.items()
                },
                'phase_reward_components': phase_reward_totals,
            }
            per_episode.append(episode_record)
            if evaluation_index < EVALUATION_TRACE_EPISODES:
                diagnostic_traces.append({
                    'evaluation_index': evaluation_index,
                    'records': episode_trace,
                })

        total_wind_opportunities = sum(
            record['wind_avoidance_opportunities'] for record in per_episode
        )
        total_wind_selections = sum(
            record['wind_hazard_selections'] for record in per_episode
        )
        total_wind_dominated_selections = sum(
            record['wind_dominated_hazard_selections']
            for record in per_episode
        )
        total_wind_shortcut_selections = sum(
            record['wind_shortcut_hazard_selections']
            for record in per_episode
        )
        total_low_signal_opportunities = sum(
            record['low_signal_avoidance_opportunities']
            for record in per_episode
        )
        total_low_signal_selections = sum(
            record['low_signal_hazard_selections'] for record in per_episode
        )
        total_low_signal_dominated_selections = sum(
            record['low_signal_dominated_hazard_selections']
            for record in per_episode
        )
        total_low_signal_shortcut_selections = sum(
            record['low_signal_shortcut_hazard_selections']
            for record in per_episode
        )
        all_route_challenges = [
            challenge
            for record in per_episode
            for challenge in record['hazard_route_challenges']
        ]

        return {
            'evaluation_mode': (
                'full_mission' if curriculum_stage is None
                else f'curriculum_stage_{int(curriculum_stage)}'
            ),
            'curriculum_stage': (
                len(CURRICULUM_STAGES) - 1
                if curriculum_stage is None else int(curriculum_stage)
            ),
            'episodes': int(num_episodes),
            'suite_seed': int(seed),
            'mean_reward': float(np.mean(rewards)),
            'mean_delivered': float(np.mean(delivered)),
            'mean_died': float(np.mean(died)),
            'mean_landed': float(np.mean(landed)),
            'mean_total_battery_drain': float(np.mean([
                record['total_battery_drain'] for record in per_episode
            ])),
            'mean_total_wind_battery_drain': float(np.mean([
                record['total_wind_battery_drain'] for record in per_episode
            ])),
            'mean_wind_energy_fraction': float(np.mean([
                record['wind_energy_fraction'] for record in per_episode
            ])),
            'mean_battery_units_per_delivery': float(np.mean([
                record['battery_units_per_delivery'] for record in per_episode
            ])),
            'mean_weighted_deliveries_per_battery_unit': float(np.mean([
                record['weighted_deliveries_per_battery_unit']
                for record in per_episode
            ])),
            'mean_energy_return_mode_fraction': float(np.mean([
                record['energy_return_mode_fraction'] for record in per_episode
            ])),
            'mean_reserve_violation_rate': float(np.mean([
                record['reserve_violation_rate'] for record in per_episode
            ])),
            'mean_energy_return_progress_rate': float(np.mean([
                record['energy_return_progress_rate'] for record in per_episode
            ])),
            'mean_energy_return_regress_rate': float(np.mean([
                record['energy_return_regress_rate'] for record in per_episode
            ])),
            'mean_energy_return_success_rate': float(np.mean([
                record['energy_return_success_rate'] for record in per_episode
            ])),
            'mean_landing_battery': float(np.mean([
                record['mean_landing_battery'] for record in per_episode
            ])),
            'minimum_landing_battery': float(min(
                record['minimum_landing_battery'] for record in per_episode
            )),
            'battery_depletion_free_rate': float(np.mean([
                not any(record['final_battery_depleted'])
                for record in per_episode
            ])),
            'total_landing_zone_arrivals': int(sum(
                len(record['landing_zone_arrival_events'])
                for record in per_episode
            )),
            'total_landing_zone_departures_without_landing': int(sum(
                len(record['landing_zone_departure_events'])
                for record in per_episode
            )),
            'total_forced_terminal_landing_actions': int(sum(
                sum(record['forced_terminal_landing_actions_by_agent'])
                for record in per_episode
            )),
            'total_dead_landing_events': int(sum(
                len(record['dead_landing_events'])
                for record in per_episode
            )),
            'total_drone_deaths': int(sum(
                sum(record['final_drone_died'])
                for record in per_episode
            )),
            'total_death_penalty_applications': int(sum(
                sum(record['death_penalty_applications_by_agent'])
                for record in per_episode
            )),
            'mean_steps_from_zone_arrival_to_land': float(np.mean([
                landing_time
                for record in per_episode
                for landing_time in record[
                    'landing_time_from_zone_arrival_by_agent'
                ]
                if landing_time >= 0
            ])) if any(
                landing_time >= 0
                for record in per_episode
                for landing_time in record[
                    'landing_time_from_zone_arrival_by_agent'
                ]
            ) else -1.0,
            'success_rate': float(np.mean(successes)),
            'mean_triage_efficiency': float(np.mean(triage_efficiencies)),
            'mean_lower_triage_delivery_floor': float(
                np.mean(lower_triage_floors)
            ),
            'mean_acuity_priority_score': float(
                np.mean(acuity_priority_scores)
            ),
            'mean_class_delivery_jain_fairness': float(
                np.mean(delivery_jain_scores)
            ),
            'mean_priority_fairness_attainment': float(
                np.mean(priority_fairness_scores)
            ),
            'mean_triage_delivery_ordering_score': float(
                np.mean(triage_ordering_scores)
            ),
            'mean_triage_delivery_rate_ordering_score': float(
                np.mean(triage_rate_ordering_scores)
            ),
            'mean_triage_response_time_ordering_score': float(
                np.mean(triage_response_ordering_scores)
            ),
            **{
                f'mean_response_ratio_w{weight}': float(np.mean(
                    class_response_ratios[weight]
                ))
                for weight in (1, 2, 3)
            },
            **{
                f'mean_response_time_w{weight}': float(np.mean(
                    class_response_times[weight]
                )) if class_response_times[weight] else -1.0
                for weight in (1, 2, 3)
            },
            **{
                f'median_response_time_w{weight}': float(np.median(
                    class_response_times[weight]
                )) if class_response_times[weight] else -1.0
                for weight in (1, 2, 3)
            },
            **{
                f'p90_response_time_w{weight}': float(np.percentile(
                    class_response_times[weight], 90
                )) if class_response_times[weight] else -1.0
                for weight in (1, 2, 3)
            },
            'mean_w3_before_w1_response_fraction': float(np.mean(
                w3_before_w1_response_fractions
            )),
            'aggregate_w3_vs_w1_response_advantage_steps': float(
                np.mean(class_response_times[1])
                - np.mean(class_response_times[3])
                if class_response_times[1] and class_response_times[3]
                else 0.0
            ),
            'mean_obstacle_collisions': float(np.mean(obstacle_collisions)),
            'mean_agent_collisions': float(np.mean(agent_collisions)),
            'mean_obstacle_collision_rate_per_operational_step': float(
                np.mean([
                    record['obstacle_collision_rate_per_operational_step']
                    for record in per_episode
                ])
            ),
            'mean_agent_collision_rate_per_operational_step': float(
                np.mean([
                    record['agent_collision_rate_per_operational_step']
                    for record in per_episode
                ])
            ),
            'total_dominated_obstacle_selections': int(sum(
                sum(record['dominated_obstacle_selections_by_agent'])
                for record in per_episode
            )),
            'total_dominated_agent_conflict_selections': int(sum(
                sum(record['dominated_agent_conflict_selections_by_agent'])
                for record in per_episode
            )),
            'total_route_intercept_challenges': int(
                len(all_route_challenges)
            ),
            'mean_route_intercept_challenges_per_episode': float(
                len(all_route_challenges) / max(1, len(per_episode))
            ),
            'mean_verified_safe_detour_ratio': float(np.mean([
                challenge['safe_detour_ratio']
                for challenge in all_route_challenges
            ])) if all_route_challenges else 0.0,
            'mean_hazard_path_intersection_cells': float(np.mean([
                challenge['path_intersection_cells']
                for challenge in all_route_challenges
            ])) if all_route_challenges else 0.0,
            'route_challenges_by_hazard_kind': {
                hazard_kind: int(sum(
                    challenge['hazard_kind'] == hazard_kind
                    for challenge in all_route_challenges
                ))
                for hazard_kind in ('wind', 'low_signal')
            },
            'route_challenges_by_initial_weight': {
                str(weight): int(sum(
                    challenge['initial_weight'] == weight
                    for challenge in all_route_challenges
                ))
                for weight in (1, 2, 3)
            },
            'mean_wind_avoidance_rate': float(np.mean([
                record['wind_avoidance_rate'] for record in per_episode
            ])),
            'mean_low_signal_avoidance_rate': float(np.mean([
                record['low_signal_avoidance_rate'] for record in per_episode
            ])),
            'total_wind_avoidance_opportunities': int(
                total_wind_opportunities
            ),
            'total_wind_hazard_selections': int(total_wind_selections),
            'total_wind_dominated_hazard_selections': int(
                total_wind_dominated_selections
            ),
            'total_wind_shortcut_hazard_selections': int(
                total_wind_shortcut_selections
            ),
            'aggregate_wind_avoidance_rate': float(
                1.0 - total_wind_selections
                / max(1, total_wind_opportunities)
            ),
            'aggregate_wind_dominated_avoidance_rate': float(
                1.0 - total_wind_dominated_selections
                / max(1, total_wind_opportunities)
            ),
            'aggregate_wind_rational_entry_fraction': float(
                total_wind_shortcut_selections
                / max(1, total_wind_selections)
            ),
            'total_low_signal_avoidance_opportunities': int(
                total_low_signal_opportunities
            ),
            'total_low_signal_hazard_selections': int(
                total_low_signal_selections
            ),
            'total_low_signal_dominated_hazard_selections': int(
                total_low_signal_dominated_selections
            ),
            'total_low_signal_shortcut_hazard_selections': int(
                total_low_signal_shortcut_selections
            ),
            'aggregate_low_signal_avoidance_rate': float(
                1.0 - total_low_signal_selections
                / max(1, total_low_signal_opportunities)
            ),
            'aggregate_low_signal_dominated_avoidance_rate': float(
                1.0 - total_low_signal_dominated_selections
                / max(1, total_low_signal_opportunities)
            ),
            'aggregate_low_signal_rational_entry_fraction': float(
                total_low_signal_shortcut_selections
                / max(1, total_low_signal_selections)
            ),
            'mean_wind_exposure_rate': float(np.mean([
                record['wind_exposure_rate'] for record in per_episode
            ])),
            'mean_low_signal_exposure_rate': float(np.mean([
                record['low_signal_exposure_rate'] for record in per_episode
            ])),
            'mean_wind_progress_cells_per_entry': float(np.mean([
                record['wind_progress_cells_per_entry']
                for record in per_episode
            ])),
            'mean_low_signal_progress_cells_per_entry': float(np.mean([
                record['low_signal_progress_cells_per_entry']
                for record in per_episode
            ])),
            'collision_free_rate': float(np.mean([
                obstacle == 0 and agent == 0
                for obstacle, agent in zip(
                    obstacle_collisions, agent_collisions
                )
            ])),
            'mean_steps': float(np.mean(episode_lengths)),
            'reward_std': float(np.std(rewards)),
            'delivered_std': float(np.std(delivered)),
            'agent_collision_std': float(np.std(agent_collisions)),
            'patient_attention_entropy': (
                float(np.mean(patient_attention_entropy))
                if patient_attention_entropy else 0.0
            ),
            'patient_attention_max': (
                float(np.mean(patient_attention_max))
                if patient_attention_max else 0.0
            ),
            'patient_attention_mass_by_initial_weight': {
                str(weight): (
                    float(np.mean(patient_attention_mass_by_weight[weight]))
                    if patient_attention_mass_by_weight[weight] else 0.0
                )
                for weight in (1, 2, 3)
            },
            'patient_attention_enrichment_by_initial_weight': {
                str(weight): (
                    float(np.mean(
                        patient_attention_enrichment_by_weight[weight]
                    ))
                    if patient_attention_enrichment_by_weight[weight] else 0.0
                )
                for weight in (1, 2, 3)
            },
            'patient_attention_weighted_response_age': (
                float(np.mean(patient_attention_weighted_response_age))
                if patient_attention_weighted_response_age else 0.0
            ),
            'drone_attention_entropy': (
                float(np.mean(drone_attention_entropy))
                if drone_attention_entropy else 0.0
            ),
            'drone_attention_max': (
                float(np.mean(drone_attention_max))
                if drone_attention_max else 0.0
            ),
            'action_query_similarity': (
                float(np.mean(action_query_similarity))
                if action_query_similarity else 0.0
            ),
            'patient_attention_entropy_std': (
                float(np.std(patient_attention_entropy))
                if patient_attention_entropy else 0.0
            ),
            'drone_attention_entropy_std': (
                float(np.std(drone_attention_entropy))
                if drone_attention_entropy else 0.0
            ),
            'action_query_similarity_std': (
                float(np.std(action_query_similarity))
                if action_query_similarity else 0.0
            ),
            'exact_q_tie_fraction': (
                float(np.mean(exact_q_tie_fractions))
                if exact_q_tie_fractions else 0.0
            ),
            'q_value_dtype': 'torch.float32',
            'q_action_gap_by_phase': {
                phase: (
                    float(np.mean(gaps)) if gaps else 0.0
                )
                for phase, gaps in q_action_gaps_by_phase.items()
            },
            'total_action_counts': np.sum(
                [record['action_counts'] for record in per_episode], axis=0
            ).astype(int).tolist(),
            'total_phase_action_counts': {
                phase: np.sum([
                    record['phase_action_counts'][phase]
                    for record in per_episode
                ], axis=0).astype(int).tolist()
                for phase in ('rescue', 'landing', 'irrecoverable')
            },
            'termination_reason_counts': {
                reason: sum(
                    record['termination_reason'] == reason
                    for record in per_episode
                )
                for reason in sorted({
                    record['termination_reason'] for record in per_episode
                })
            },
            'per_episode': per_episode,
            'diagnostic_traces': diagnostic_traces,
        }
    finally:
        if was_training:
            ctde_agent.policy_net.train()
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)

def train(headless=HEADLESS_MODE, output_dir=".", episodes=NUM_EPISODES,
          max_steps=MAX_STEPS, seed=7, checkpoint_every=250,
          device_name="auto", fast_math=False, mixed_precision=True,
          profile_memory=False,
          evaluation_every=EVALUATION_EVERY_EPISODES,
          evaluation_episodes=EVALUATION_EPISODES,
          rollout_workers=ROLLOUT_WORKERS, resume_checkpoint=None):
    if episodes < 1 or max_steps < 1 or checkpoint_every < 1:
        raise ValueError("episodes, max_steps, and checkpoint_every must be positive")
    if evaluation_every < 0 or evaluation_episodes < 1:
        raise ValueError(
            "evaluation_every must be non-negative and "
            "evaluation_episodes must be positive"
        )
    if rollout_workers < 1:
        raise ValueError('rollout_workers must be positive')
    for stage_index, stage in enumerate(CURRICULUM_STAGES):
        if stage['stage'] != stage_index:
            raise ValueError('Curriculum stage identifiers must be contiguous')
        if not 1 <= stage['initial_patients'] <= stage['max_patients'] \
                <= MAX_PATIENTS:
            raise ValueError(f'Invalid patient scale in curriculum stage {stage_index}')
        if not 0.0 <= stage['hazard_fraction'] <= 1.0:
            raise ValueError(f'Invalid hazard fraction in stage {stage_index}')
        if not 1 <= int(stage['patient_timer']) <= MAX_PATIENT_TIMER:
            raise ValueError(
                f'Invalid patient timer in curriculum stage {stage_index}'
            )
        if int(stage['spawn_interval']) < 1 or int(stage['spawn_jitter']) < 0:
            raise ValueError(
                f'Invalid spawn timing in curriculum stage {stage_index}'
            )
        if not (
                1 <= int(stage['spawn_batch_min'])
                <= int(stage['spawn_batch_max'])):
            raise ValueError(
                f'Invalid spawn batch in curriculum stage {stage_index}'
            )
        if int(stage['final_spawn_step']) < 1:
            raise ValueError(
                f'Invalid final spawn step in curriculum stage {stage_index}'
            )
        if (stage['max_patients'] > stage['initial_patients']
                and not stage['dynamic_spawning']):
            raise ValueError(
                f'Stage {stage_index} cannot spawn its configured patients'
            )
        for gate_name in (
                'delivery_gate', 'triage_gate', 'landing_gate',
                'lower_triage_gate', 'collision_rate_gate',
                'acuity_priority_gate', 'priority_fairness_gate',
                'triage_ordering_gate', 'wind_avoidance_gate',
                'low_signal_avoidance_gate'):
            if not 0.0 <= stage[gate_name] <= 1.0:
                raise ValueError(
                    f'Invalid {gate_name} in curriculum stage {stage_index}'
                )
        if stage['minimum_hazard_opportunities'] < 0:
            raise ValueError(
                'Minimum hazard opportunities cannot be negative in '
                f'curriculum stage {stage_index}'
            )

    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = resolve_device(device_name)
    tensor_core_math = False
    if device.type == "cuda":
        tensor_core_math = fast_math or mixed_precision
        torch.backends.cuda.matmul.allow_tf32 = tensor_core_math
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = tensor_core_math
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision(
            "high" if tensor_core_math else "highest"
        )

    training_start = time.perf_counter()
    if profile_memory:
        tracemalloc.start()
    print("\n" + "="*60)
    print("MARL TRAINING")
    print("="*60)
    print(f"Grid Size:            {GRID_SIZE}x{GRID_SIZE}")
    print(f"Episodes:             {episodes}")
    print(f"Max Steps:            {max_steps}")
    print(f"Initial Patients:     {NUM_INITIAL_PATIENTS}")
    print(f"Max Patients:         {MAX_PATIENTS}")
    print(f"Baseline Spawn Int.:  {NEW_PATIENT_SPAWN_INTERVAL} steps")
    print(f"Maximum Patient Timer:{MAX_PATIENT_TIMER:>5} steps")
    print(f"Learning Rate:        {LEARNING_RATE}")
    print(f"Discount / N-step:    {GAMMA} / {N_STEP}")
    print(f"TD Reward Scale:      {TD_REWARD_SCALE}")
    print(f"Batch / Replay:       {BATCH_SIZE} / {BUFFER_CAPACITY}")
    print(f"Replay Warm-up:       {REPLAY_WARMUP} transitions")
    print(f"PER Alpha / Cap:      {PER_ALPHA} / {PER_PRIORITY_MAX}")
    print(f"Attention:            {SET_ATTENTION_BLOCKS} blocks, {ATTENTION_HEADS} heads")
    print(f"Obstacles:            {NUM_OBSTACLES}")
    print(f"Battery Capacity:     {MAX_BATTERY:.0f}%")
    print(f"Clean Endurance:      "
          f"{MAX_BATTERY / BATTERY_DRAIN_PER_STEP:.0f} steps")
    print(f"Battery Drain:        {BATTERY_DRAIN_PER_STEP}/step")
    print(f"Wind Extra Drain:     {BATTERY_DRAIN_IN_WIND}/step")
    print(f"Pad Standby Drain:    {BATTERY_DRAIN_AT_LANDING_ZONE}/step")
    print(f"Return Safety Buffer: {SAFE_RETURN_BATTERY_BUFFER:>5.1f}%")
    print("Return Cost Map:      obstacle + wind + signal + failures")
    print(f"Wind Rectangles:      {NUM_WIND_ZONE_RECTANGLES}")
    print(f"Low-Signal Rectangles: {NUM_LOW_SIGNAL_ZONE_RECTANGLES}")
    print(f"Hazard Refresh:       {WIND_APPEAR_INTERVAL}/"
          f"{LOW_SIGNAL_APPEAR_INTERVAL} steps (wind/low-signal)")
    print(f"Local Hazard View:    {LOCAL_GRID_SIZE}x{LOCAL_GRID_SIZE}")
    print(f"Low Signal Fail Prob: {LOW_SIGNAL_FAILURE_PROB}")
    print(f"Training Device:      {device}")
    if device.type == 'cuda':
        print(f"CUDA GPU:             {torch.cuda.get_device_name(device)}")
    print(f"Random Seed:          {seed}")
    print(f"TF32 Tensor Cores:    {tensor_core_math}")
    print(f"Training Cadence:     {UPDATES_PER_TRAIN} update / {TRAIN_EVERY_STEPS} steps")
    print(f"Stage Epsilon Reset:  {CURRICULUM_EPSILON_RESET:.2f} -> "
          f"{CURRICULUM_EPSILON_FLOOR:.2f} over "
          f"{CURRICULUM_EPSILON_DECAY_EPISODES} stage episodes")
    print(f"Parallel Rollouts:    {rollout_workers}")
    print(f"Replay Samples/Step:  "
          f"{BATCH_SIZE * UPDATES_PER_TRAIN / TRAIN_EVERY_STEPS:.1f}")
    print("Mission Curriculum:   " + " -> ".join(
        f"S{stage['stage']}:{stage['max_patients']}P"
        for stage in CURRICULUM_STAGES
    ))
    print("Triage Scarcity:      " + " -> ".join(
        f"S{stage['stage']}:{stage['patient_timer']}T/"
        f"{stage['spawn_interval']}I"
        for stage in CURRICULUM_STAGES
    ))
    if evaluation_every:
        print(f"Greedy Evaluation:    {evaluation_episodes} episodes every "
              f"{evaluation_every} training episodes")
    else:
        print("Greedy Evaluation:    disabled")
    print(f"Memory Profiling:     {profile_memory}")
    print("="*60 + "\n")

    print(f"Using device: {device}\n")
    print(f"Headless mode: {headless}")
    print(f"Output directory: {output_dir_path}\n")

    env = Environment(fixed_layout=False, episode_max_steps=max_steps)
    curriculum = CurriculumManager()
    data = Data_Collection()
    cuda_device_data = None
    if device.type == 'cuda':
        cuda_properties = torch.cuda.get_device_properties(device)
        cuda_device_data = {
            'name': cuda_properties.name,
            'total_memory_mb': float(
                cuda_properties.total_memory / (1024.0 ** 2)
            ),
            'compute_capability': [
                int(cuda_properties.major), int(cuda_properties.minor)
            ],
        }
    data.run_configuration = {
        'started_at': datetime.now().isoformat(),
        'episodes_requested': int(episodes),
        'max_steps_per_episode': int(max_steps),
        'seed': int(seed),
        'checkpoint_every': int(checkpoint_every),
        'evaluation_every': int(evaluation_every),
        'evaluation_episodes': int(evaluation_episodes),
        'rollout_workers': int(rollout_workers),
        'headless': bool(headless),
        'output_directory': str(output_dir_path),
        'requested_device': device_name,
        'resolved_device': str(device),
        'cuda_device': cuda_device_data,
        'fast_math_requested': bool(fast_math),
        'mixed_precision_requested': bool(mixed_precision),
        'python_memory_profiling_enabled': bool(profile_memory),
        'competence_gated_full_curriculum': True,
        'hard_stage_budget_curriculum_promotion': (
            CURRICULUM_FORCE_PROMOTION_AT_MAXIMUM
        ),
        'guaranteed_final_stage_exposure': True,
        'battery_units': 'percentage_points',
        'maximum_battery': MAX_BATTERY,
        'clean_endurance_steps': float(
            MAX_BATTERY / BATTERY_DRAIN_PER_STEP
        ),
        'episode_continues_after_drone_death': True,
        'dead_drones_remain_physically_inactive': True,
        'pre_resolution_energy_return_landing': True,
        'per_drone_died_observation': True,
        'post_depletion_reminder_steps': int(
            POST_DEPLETION_REMINDER_STEPS
        ),
        'post_depletion_reminder_penalty': float(
            POST_DEPLETION_REMINDER_PENALTY
        ),
        'dead_landing_penalty': float(DEAD_LANDING_PENALTY),
        'curriculum_epsilon_reset': CURRICULUM_EPSILON_RESET,
        'curriculum_epsilon_floor': CURRICULUM_EPSILON_FLOOR,
        'curriculum_epsilon_decay_episodes': (
            CURRICULUM_EPSILON_DECAY_EPISODES
        ),
        'triage_response_time_tie_breaking': True,
        'patient_response_age_observation': True,
        'priority_weighted_response_reward': True,
        'progressive_hazard_penalty_curriculum': True,
        'candidate_rescue_energy_feasibility': True,
        'stage_specific_scarcity_curriculum': True,
        'constrained_energy_budget': True,
        'obstacle_aware_energy_return_reserve': True,
        'return_energy_risk_multiplier': RETURN_ENERGY_RISK_MULTIPLIER,
        'wind_signal_expected_cost_return_map': True,
        'single_return_safety_reserve': True,
        'pre_resolution_energy_return_shaping': True,
        'low_power_landing_zone_standby': True,
        'route_intercepting_hazards': True,
        'dense_priority_service_debt_reward': True,
        'counterfactual_collision_selection_penalties': True,
        'evaluation_trace_episode_limit': EVALUATION_TRACE_EPISODES,
        'evaluation_trace_record_limit': EVALUATION_TRACE_MAX_RECORDS,
        'obstacle_event_record_limit': (
            MAX_RECORDED_OBSTACLE_EVENTS_PER_EPISODE
        ),
        'curriculum_stages': [dict(stage) for stage in CURRICULUM_STAGES],
        'curriculum_required_passes': int(CURRICULUM_REQUIRED_PASSES),
        'curriculum_episode_mix': {
            'current': CURRICULUM_CURRENT_PROBABILITY,
            'previous': CURRICULUM_PREVIOUS_PROBABILITY,
            'full': CURRICULUM_FULL_PROBABILITY,
        },
        'host': {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
        
        
        'training_obstacles': [
            list(position) for position in sorted(env.obstacles)
        ],
    }

    def save_training_data():
        data.run_configuration['episodes_completed_this_process'] = len(
            data.episodes
        )
        data.run_configuration['episodes_completed'] = (
            resumed_completed_episodes + len(data.episodes)
        )
        data.run_configuration['last_saved_at'] = datetime.now().isoformat()
        filename = output_dir_path / "training_metrics.json"
        return data.save_to_json(str(filename))

    metrics_journal_path = output_dir_path / 'training_metrics.jsonl'
    data.run_configuration['metrics_journal'] = str(metrics_journal_path)

    def append_metrics_journal(learning_data):

        index = -1
        diagnostic = data.episode_diagnostics[index]
        record = {
            'episode': int(data.episodes[index]),
            'timestamp': datetime.now().isoformat(),
            'curriculum_stage': int(diagnostic['curriculum_stage']),
            'curriculum_controller_stage': int(
                diagnostic['curriculum_controller_stage']
            ),
            'episode_mode': data.episode_modes[index],
            'reward': float(data.total_rewards[index]),
            'steps': int(data.steps_per_episode[index]),
            'delivered': int(data.patients_delivered_counts[index]),
            'died': int(data.patients_died_counts[index]),
            'spawned': int(data.patients_spawned_counts[index]),
            'landed': int(sum(
                getattr(data, f'agent_{agent_index}_landed')[index]
                for agent_index in range(NUM_AGENTS)
            )),
            'episode_patient_timer': int(
                diagnostic['episode_patient_timer']
            ),
            'patient_spawn_interval': int(
                diagnostic['patient_spawn_interval']
            ),
            'final_patient_spawn_step': int(
                diagnostic['final_patient_spawn_step']
            ),
            'hazard_penalty_scale': float(
                diagnostic['hazard_penalty_scale']
            ),
            'triage_efficiency': float(data.triage_efficiency[index]),
            'triage_delivery_ordering_score': float(
                data.triage_delivery_ordering_score[index]
            ),
            'triage_delivery_rate_ordering_score': float(
                data.triage_delivery_rate_ordering_score[index]
            ),
            'triage_response_time_ordering_score': float(
                data.triage_response_time_ordering_score[index]
            ),
            'high_vs_low_response_advantage': float(
                data.high_vs_low_response_advantage[index]
            ),
            'w3_vs_w1_response_advantage_steps': float(
                data.w3_vs_w1_response_advantage_steps[index]
            ),
            'w3_before_w1_response_fraction': float(
                data.w3_before_w1_response_fraction[index]
            ),
            'mean_delivered_response_time': float(
                data.mean_delivered_response_time[index]
            ),
            'obstacle_collisions': int(data.collisions_obstacles[index]),
            'agent_collisions': int(data.collisions_agents[index]),
            'wind_entries': int(sum(diagnostic['wind_entries_by_agent'])),
            'wind_exposure_steps': int(sum(
                diagnostic['wind_exposure_steps_by_agent']
            )),
            'wind_avoidance_rate': float(
                diagnostic['wind_avoidance_rate']
            ),
            'wind_dominated_avoidance_rate': float(
                diagnostic['wind_dominated_avoidance_rate']
            ),
            'wind_rational_entry_fraction': float(
                diagnostic['wind_rational_entry_fraction']
            ),
            'low_signal_entries': int(sum(
                diagnostic['low_signal_entries_by_agent']
            )),
            'low_signal_exposure_steps': int(sum(
                diagnostic['low_signal_exposure_steps_by_agent']
            )),
            'low_signal_avoidance_rate': float(
                diagnostic['low_signal_avoidance_rate']
            ),
            'low_signal_dominated_avoidance_rate': float(
                diagnostic['low_signal_dominated_avoidance_rate']
            ),
            'low_signal_rational_entry_fraction': float(
                diagnostic['low_signal_rational_entry_fraction']
            ),
            'termination_reason': diagnostic['termination_reason'],
            'mission_outcome': diagnostic['mission_outcome'],
            'rescue_success': bool(diagnostic['rescue_success']),
            'perfect_rescue': bool(diagnostic.get('perfect_rescue', False)),
            'safe_return_complete': bool(
                diagnostic['safe_return_complete']
            ),
            'local_reward_sum_error': float(
                diagnostic['local_reward_sum_error']
            ),
            'reward_components': {
                name: float(data.reward_components[name][index])
                for name in data.reward_component_names
            },
            'learning': {
                name: (
                    float(value) if isinstance(
                        value, (int, float, np.integer, np.floating)
                    ) else value
                )
                for name, value in learning_data.items()
            },
            'curriculum_state': curriculum.state_dict(),
        }
        for optional_name in (
            'unconstrained_obstacle_preferences_by_agent',
            'unconstrained_boundary_preferences_by_agent',
            'unconstrained_occupied_cell_preferences_by_agent',
            'obstacle_collision_rate_per_operational_step',
            'agent_collision_rate_per_operational_step',
            'wind_exposure_rate',
            'low_signal_exposure_rate',
            'wind_avoidance_rate',
            'low_signal_avoidance_rate',
            'wind_dominated_avoidance_rate',
            'low_signal_dominated_avoidance_rate',
            'wind_rational_entry_fraction',
            'low_signal_rational_entry_fraction',
            'low_signal_failure_rate',
            'wind_failure_rate',
            'battery_drain_by_agent',
            'wind_battery_drain_by_agent',
            'landing_standby_steps_by_agent',
            'energy_return_mode_steps_by_agent',
            'energy_return_progress_steps_by_agent',
            'energy_return_regress_steps_by_agent',
            'energy_return_activations_by_agent',
            'energy_margin_delta_by_agent',
            'reserve_violation_steps_by_agent',
            'minimum_safe_return_margin_by_agent',
            'total_battery_drain',
            'total_wind_battery_drain',
            'wind_energy_fraction',
            'battery_units_per_delivery',
            'weighted_deliveries_per_battery_unit',
            'energy_return_mode_fraction',
            'energy_return_progress_rate',
            'energy_return_regress_rate',
            'energy_return_success_rate',
            'mean_landing_battery',
            'minimum_landing_battery',
            'reserve_violation_rate',
        ):
            if optional_name in diagnostic:
                record[optional_name] = diagnostic[optional_name]
        with metrics_journal_path.open('a', encoding='utf-8') as journal:
            journal.write(json.dumps(record, separators=(',', ':')) + '\n')

    state_example = env.get_state()
    action_dim = ACTION_DIM
    print(f"Drone entity table:     {state_example['drones'].shape}")
    print(f"Patient entity table:   {state_example['patients'].shape}")
    print(f"Local grid table:       {state_example['local_grids'].shape}")
    print(f"Action dimension:       {action_dim}\n")
    print(f"Actions:                {', '.join(ACTION_NAMES)}\n")

    ctde_agent = CTDEAgent(
        action_dim,
        LEARNING_RATE,
        GAMMA,
        device,
        mixed_precision=mixed_precision
    )
    print(f"BF16 Mixed Precision: {ctde_agent.use_mixed_precision}")
    print(f"Fused Adam Optimizer: {ctde_agent.uses_fused_optimizer}\n")
    if sys.platform.startswith('linux') and device.type != 'cuda':
        print(
            "WARNING: Linux cloud training is not using CUDA. "
            "Verify the CUDA PyTorch build and run with --device cuda:0.\n",
            flush=True
        )

    resumed_completed_episodes = 0
    resumed_global_step = 0
    if resume_checkpoint is not None:
        checkpoint_path = Path(resume_checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f'Resume checkpoint does not exist: {checkpoint_path}'
            )
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        architecture = checkpoint.get('architecture', {})
        for name, expected in {
            'grid_size': GRID_SIZE,
            'num_agents': NUM_AGENTS,
            'max_patients': MAX_PATIENTS,
            'action_dim': ACTION_DIM,
            'drone_state_dim': DRONE_STATE_DIM,
            'patient_state_dim': PATIENT_STATE_DIM,
            'local_grid_radius': LOCAL_GRID_RADIUS,
            'local_grid_size': LOCAL_GRID_SIZE,
        }.items():
            if int(architecture.get(name, -1)) != int(expected):
                raise ValueError(
                    f'Resume checkpoint {name} is incompatible: '
                    f"{architecture.get(name)} != {expected}"
                )
        ctde_agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        ctde_agent.target_net.load_state_dict(checkpoint['target_state_dict'])
        ctde_agent.mixer.load_state_dict(checkpoint['mixer_state_dict'])
        ctde_agent.target_mixer.load_state_dict(
            checkpoint['target_mixer_state_dict']
        )
        ctde_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ctde_agent.learner_steps = int(checkpoint.get('learner_steps', 0))
        if 'curriculum_state' in checkpoint:
            curriculum.load_state_dict(checkpoint['curriculum_state'])
        resumed_completed_episodes = int(
            checkpoint.get('completed_episodes', 0)
        )
        resumed_global_step = int(checkpoint.get('global_step', 0))
        if resumed_completed_episodes >= episodes:
            raise ValueError(
                f'Checkpoint already completed {resumed_completed_episodes} '
                f'episodes, but --episodes is {episodes}'
            )
        data.run_configuration.update({
            'resumed_from_checkpoint': str(checkpoint_path),
            'resumed_completed_episodes': resumed_completed_episodes,
            'resumed_global_step': resumed_global_step,
            'resume_replay_restored': False,
        })
        print(
            f"Resumed checkpoint: {checkpoint_path}\n"
            f"  completed episodes: {resumed_completed_episodes}\n"
            f"  global/learner steps: {resumed_global_step}/"
            f"{ctde_agent.learner_steps}\n"
            "  replay: rebuilt with the normal warm-up\n",
            flush=True,
        )

    def save_checkpoint(completed_episodes, current_epsilon, current_global_step):
        checkpoint_path = output_dir_path / f"checkpoint_ep{completed_episodes}.pt"
        temporary_checkpoint_path = checkpoint_path.with_suffix(".pt.tmp")
        torch.save({
            "model_version": MODEL_VERSION,
            "completed_episodes": completed_episodes,
            "epsilon": current_epsilon,
            "global_step": current_global_step,
            "learner_steps": ctde_agent.learner_steps,
            "seed": seed,
            "device": str(device),
            "policy_state_dict": ctde_agent.policy_net.state_dict(),
            "target_state_dict": ctde_agent.target_net.state_dict(),
            "mixer_state_dict": ctde_agent.mixer.state_dict(),
            "target_mixer_state_dict": ctde_agent.target_mixer.state_dict(),
            "optimizer_state_dict": ctde_agent.optimizer.state_dict(),
            "replay_size": len(ctde_agent.joint_buffer),
            "replay_position": ctde_agent.joint_buffer.position,
            "replay_max_priority": ctde_agent.joint_buffer.max_priority,
            "train_every_steps": TRAIN_EVERY_STEPS,
            "updates_per_train": UPDATES_PER_TRAIN,
            "battery_units": "percentage_points",
            "maximum_battery": MAX_BATTERY,
            "battery_drain_per_step": BATTERY_DRAIN_PER_STEP,
            "battery_drain_in_wind": BATTERY_DRAIN_IN_WIND,
            "battery_drain_at_landing_zone": (
                BATTERY_DRAIN_AT_LANDING_ZONE
            ),
            "low_battery_threshold": LOW_BATTERY_THRESHOLD,
            "safe_return_battery_buffer": SAFE_RETURN_BATTERY_BUFFER,
            "energy_return_trigger_margin": ENERGY_RETURN_TRIGGER_MARGIN,
            "return_energy_risk_multiplier": (
                RETURN_ENERGY_RISK_MULTIPLIER
            ),
            "wind_signal_expected_cost_return_map": True,
            "single_return_safety_reserve": True,
            "route_intercepting_hazards": True,
            "dense_priority_service_debt_reward": True,
            "priority_service_potential_scale": (
                PRIORITY_SERVICE_POTENTIAL_SCALE
            ),
            "counterfactual_collision_selection_penalties": True,
            "post_depletion_reminder_steps": (
                POST_DEPLETION_REMINDER_STEPS
            ),
            "post_depletion_reminder_penalty": (
                POST_DEPLETION_REMINDER_PENALTY
            ),
            "dead_landing_penalty": DEAD_LANDING_PENALTY,
            "episode_continues_after_drone_death": True,
            "pre_resolution_energy_return_landing": True,
            "per_drone_died_observation": True,
            "curriculum_epsilon_reset": CURRICULUM_EPSILON_RESET,
            "curriculum_epsilon_floor": CURRICULUM_EPSILON_FLOOR,
            "curriculum_epsilon_decay_episodes": (
                CURRICULUM_EPSILON_DECAY_EPISODES
            ),
            "energy_usage_penalty_per_unit": (
                ENERGY_USAGE_PENALTY_PER_UNIT
            ),
            "safe_return_reserve_penalty": SAFE_RETURN_RESERVE_PENALTY,
            "energy_margin_shaping_factor": ENERGY_MARGIN_SHAPING_FACTOR,
            "response_wait_penalty_per_patient": (
                RESPONSE_WAIT_PENALTY_PER_PATIENT
            ),
            "response_time_delivery_reward": RESPONSE_TIME_DELIVERY_REWARD,
            "energy_standby_hover_penalty": (
                ENERGY_STANDBY_HOVER_PENALTY
            ),
            "mixed_precision": ctde_agent.use_mixed_precision,
            "fused_optimizer": ctde_agent.uses_fused_optimizer,
            "tensor_core_math": tensor_core_math,
            "evaluation_every": evaluation_every,
            "evaluation_episodes": evaluation_episodes,
            "architecture": {
                "grid_size": GRID_SIZE,
                "num_agents": NUM_AGENTS,
                "max_patients": MAX_PATIENTS,
                "action_dim": ACTION_DIM,
                "action_names": ACTION_NAMES,
                "drone_state_dim": DRONE_STATE_DIM,
                "patient_state_dim": PATIENT_STATE_DIM,
                "local_grid_radius": LOCAL_GRID_RADIUS,
                "local_grid_size": LOCAL_GRID_SIZE,
                "entity_embed_dim": ENTITY_EMBED_DIM,
                "attention_heads": ATTENTION_HEADS,
                "set_attention_blocks": SET_ATTENTION_BLOCKS,
                "mixer_embed_dim": MIXER_EMBED_DIM,
                "agent_id_embed_dim": AGENT_ID_EMBED_DIM,
                "central_grid_embed_dim": CENTRAL_GRID_EMBED_DIM,
                "agent_mix_context_dim": AGENT_MIX_CONTEXT_DIM,
                "mission_state_dim": MISSION_STATE_DIM,
            },
            "td_reward_scale": TD_REWARD_SCALE,
            "local_td_loss_weight": LOCAL_TD_LOSS_WEIGHT,
            "per_alpha": PER_ALPHA,
            "per_priority_max": PER_PRIORITY_MAX,
            "replay_uniform_fraction": REPLAY_UNIFORM_FRACTION,
            "replay_rescue_fraction": REPLAY_RESCUE_FRACTION,
            "replay_landing_fraction": REPLAY_LANDING_FRACTION,
            "mixer_min_raw_weight": MIXER_MIN_RAW_WEIGHT,
            "epsilon_is_per_agent": True,
            "safe_patient_spawn_deadline": True,
            "boundary_invalid_action_masking": False,
            "obstacle_action_masking": False,
            "occupied_cell_action_masking": False,
            "learned_obstacle_and_agent_avoidance": True,
            "action_conditioned_entity_attention": True,
            "phase_specific_dueling_heads": True,
            "fp32_action_value_heads": True,
            "phase_balanced_replay": True,
            "stage_and_phase_balanced_replay": True,
            "obstacle_aware_landing_distance": True,
            "constrained_energy_budget": True,
            "obstacle_aware_energy_return_reserve": True,
            "pre_resolution_energy_return_shaping": True,
            "candidate_rescue_energy_feasibility": True,
            "low_power_landing_zone_standby": True,
            "connected_random_mission_layout": True,
            "competence_gated_full_curriculum": True,
            "hard_stage_budget_curriculum_promotion": (
                CURRICULUM_FORCE_PROMOTION_AT_MAXIMUM
            ),
            "guaranteed_final_stage_exposure": True,
            "curriculum_stages": CURRICULUM_STAGES,
            "curriculum_state": curriculum.state_dict(),
            "pending_only_local_patient_attention": True,
            "explicit_hover_action": True,
            "terminal_land_only_action_masking": True,
            "landing_zone_relative_direction_features": True,
            "irrecoverable_failure_termination": False,
            "post_resolution_landing_phase": True,
            "landing_after_patient_deaths": True,
            "reward_derived_local_td_credit": True,
            "agent_attributed_navigation_potential": True,
            "priority_neutral_unique_coverage_shaping": True,
            "individual_logistic_survival_dynamics": True,
            "expanded_decentralized_hazard_observation": True,
            "directional_hazard_corridor_features": True,
            "decision_aware_hazard_routing_reward": True,
            "triage_response_time_tie_breaking": True,
            "patient_response_age_observation": True,
            "priority_weighted_response_reward": True,
            "progressive_hazard_penalty_curriculum": True,
            "stage_specific_scarcity_curriculum": True,
            "wind_zone_refresh_interval_steps": WIND_APPEAR_INTERVAL,
            "low_signal_zone_refresh_interval_steps": (
                LOW_SIGNAL_APPEAR_INTERVAL
            ),
            "wind_penalty": WIND_PENALTY,
            "low_signal_penalty": LOW_SIGNAL_PENALTY,
            "wind_entry_penalty": WIND_ENTRY_PENALTY,
            "low_signal_entry_penalty": LOW_SIGNAL_ENTRY_PENALTY,
            "wind_dominated_selection_penalty": (
                WIND_DOMINATED_SELECTION_PENALTY
            ),
            "wind_shortcut_selection_penalty": (
                WIND_SHORTCUT_SELECTION_PENALTY
            ),
            "low_signal_dominated_selection_penalty": (
                LOW_SIGNAL_DOMINATED_SELECTION_PENALTY
            ),
            "low_signal_shortcut_selection_penalty": (
                LOW_SIGNAL_SHORTCUT_SELECTION_PENALTY
            ),
            "triage_class_delivery_targets": TRIAGE_CLASS_DELIVERY_TARGETS,
            "priority_fairness_outcome_reward": FAIRNESS_OUTCOME_REWARD,
            "triage_ordering_outcome_reward": (
                TRIAGE_ORDERING_OUTCOME_REWARD
            ),
            "triage_response_outcome_reward": (
                TRIAGE_RESPONSE_OUTCOME_REWARD
            ),
            "hazard_rectangles_may_overlap_obstacles": True,
            "supervised_targets_or_demonstrations": False,
        }, temporary_checkpoint_path)
        temporary_checkpoint_path.replace(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    
    
    online_parameters = (
        list(ctde_agent.policy_net.parameters())
        + list(ctde_agent.mixer.parameters())
    )
    model_parameters = sum(p.numel() for p in online_parameters)
    model_size_mb = sum(
        p.numel() * p.element_size() for p in online_parameters
    ) / (1024.0 ** 2)

    global_step = resumed_global_step
    epsilon = training_epsilon_at_step(
        global_step, curriculum.current_stage_episodes
    )

    print(
        f"Allocated joint replay: "
        f"{ctde_agent.joint_buffer.allocated_mb():.2f} MiB\n"
    )

    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        print(f"\nReceived signal {signum}; saving after the current episode.", flush=True)
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    screen = None
    clock = None
    if not headless:
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("MARL — Medical Supply Delivery")
        clock = pygame.time.Clock()

    if rollout_workers > 1:
        rollout_slots = []
        started_episodes = resumed_completed_episodes
        completed_episodes = resumed_completed_episodes
        next_evaluation_episode = (
            (
                (completed_episodes // evaluation_every + 1)
                * evaluation_every
            ) if evaluation_every > 0 else episodes + 1
        )
        next_checkpoint_episode = (
            (completed_episodes // checkpoint_every + 1) * checkpoint_every
        )
        next_train_step = (
            (global_step // TRAIN_EVERY_STEPS + 1) * TRAIN_EVERY_STEPS
        )

        def curriculum_selection_probability(selected_stage):
            if curriculum.current_stage == len(CURRICULUM_STAGES) - 1:
                return 1.0
            if selected_stage == curriculum.current_stage:
                return (
                    CURRICULUM_CURRENT_PROBABILITY
                    + (
                        CURRICULUM_PREVIOUS_PROBABILITY
                        if curriculum.current_stage == 0 else 0.0
                    )
                )
            if selected_stage == max(0, curriculum.current_stage - 1):
                return CURRICULUM_PREVIOUS_PROBABILITY
            return CURRICULUM_FULL_PROBABILITY

        def create_rollout_slot(stream_id):
            nonlocal started_episodes
            stage_index = curriculum.select_training_stage()
            rollout_env = Environment(
                fixed_layout=False, episode_max_steps=max_steps
            )
            rollout_state = rollout_env.reset(curriculum_stage=stage_index)
            tracker = TrainingEpisodeTracker(
                rollout_env,
                stage_index,
                curriculum_selection_probability(stage_index),
            )
            started_episodes += 1
            return {
                'stream_id': int(stream_id),
                'env': rollout_env,
                'state': rollout_state,
                'stage': stage_index,
                'tracker': tracker,
            }

        for stream_id in range(min(
                rollout_workers, episodes - started_episodes)):
            rollout_slots.append(create_rollout_slot(stream_id))

        while rollout_slots and completed_episodes < episodes:
            inference_start_ns = time.perf_counter_ns()
            batched_actions, batch_policy_diagnostics = (
                ctde_agent.select_actions_batch(
                    [slot['state'] for slot in rollout_slots],
                    epsilon,
                    return_diagnostics=True,
                )
            )
            batch_inference_ms = (
                time.perf_counter_ns() - inference_start_ns
            ) / 1_000_000.0
            inference_per_environment_ms = (
                batch_inference_ms / max(1, len(rollout_slots))
            )
            finished_slots = []
            for slot_index, (slot, actions) in enumerate(zip(
                    rollout_slots, batched_actions)):
                state_before = slot['state']
                next_state, _, done, step_data = slot['env'].step(actions)
                slot['tracker'].record_step(
                    state_before,
                    actions,
                    step_data,
                    inference_per_environment_ms,
                    unconstrained_actions=batch_policy_diagnostics[
                        'unconstrained_actions'
                    ][slot_index],
                )
                global_step += 1
                transition_done = bool(
                    done or slot['tracker'].steps >= (
                        max_steps + slot['env'].landing_grace_steps
                    )
                )
                ctde_agent.push(
                    state_before,
                    actions,
                    step_data['team_reward'],
                    step_data['local_rewards'],
                    step_data['event_flags'],
                    next_state,
                    transition_done,
                    curriculum_stage=slot['stage'],
                    stream_id=slot['stream_id'],
                )
                slot['state'] = next_state
                if transition_done:
                    finished_slots.append(slot)

            update_latency_ms = 0.0
            while global_step >= next_train_step:
                update_start_ns = time.perf_counter_ns()
                for _ in range(UPDATES_PER_TRAIN):
                    ctde_agent.train_step(
                        BATCH_SIZE,
                        global_step,
                        current_stage=curriculum.current_stage,
                    )
                update_latency_ms += (
                    time.perf_counter_ns() - update_start_ns
                ) / 1_000_000.0
                next_train_step += TRAIN_EVERY_STEPS
            if update_latency_ms > 0.0:
                for slot in rollout_slots:
                    slot['tracker'].training_update_latency_ms.append(
                        update_latency_ms
                    )
            epsilon = training_epsilon_at_step(
                global_step, curriculum.current_stage_episodes
            )

            if finished_slots:
                learning_data = ctde_agent.get_metrics()
                learning_data['per_beta'] = ctde_agent.beta_at_step(global_step)
                for slot in finished_slots:
                    curriculum.record_training_episode(slot['stage'])
                    result = slot['tracker'].finish(
                        slot['env'], global_step, ctde_agent.learner_steps, device
                    )
                    result['episode_diagnostics'][
                        'curriculum_controller_stage'
                    ] = int(curriculum.current_stage)
                    data.log_episode(
                        episode=completed_episodes,
                        total_reward=result['total_reward'],
                        success=result['success'],
                        agent_delivered=result['agent_delivered'],
                        patients_delivered_count=(
                            result['patients_delivered_count']
                        ),
                        patients_died_count=result['patients_died_count'],
                        patients_spawned_count=result['patients_spawned_count'],
                        landed=slot['env'].landed,
                        steps=slot['tracker'].steps,
                        collisions_obs=slot['tracker'].obstacle_collisions,
                        collisions_ag=slot['tracker'].agent_collisions,
                        wind_entries=slot['tracker'].wind_entries.tolist(),
                        low_signal_entries=(
                            slot['tracker'].low_signal_entries.tolist()
                        ),
                        epsilon=epsilon,
                        batteries=slot['env'].batteries,
                        triage_data=result['triage_data'],
                        complexity_data=result['complexity_data'],
                        learning_data=learning_data,
                        reward_component_totals=(
                            slot['tracker'].reward_components
                        ),
                        episode_diagnostics=result['episode_diagnostics'],
                        episode_mode=slot['env'].episode_mode,
                        curriculum_max_distance=(
                            slot['env'].curriculum_max_distance
                        ),
                        curriculum_max_landing_distance=(
                            slot['env'].curriculum_max_landing_distance
                        ),
                        curriculum_initial_distances=(
                            slot['env'].curriculum_initial_distances
                        ),
                        curriculum_success=result['curriculum_success'],
                        curriculum_start_step=0,
                    )
                    append_metrics_journal(learning_data)
                    completed_episodes += 1
                    print(
                        f"Ep {completed_episodes:>5} | "
                        f"Reward: {result['total_reward']:>8.2f} | "
                        f"Stage: S{slot['stage']} | ε: {epsilon:.3f} | "
                        f"Delivered: {result['patients_delivered_count']}/"
                        f"{result['patients_spawned_count']} | "
                        f"Died: {result['patients_died_count']} | "
                        f"Landed: {sum(slot['env'].landed)}/{NUM_AGENTS} | "
                        f"Steps: {slot['tracker'].steps} | "
                        f"End: {slot['env'].termination_reason} | "
                        f"Collisions(obs/ag): "
                        f"{slot['tracker'].obstacle_collisions}/"
                        f"{slot['tracker'].agent_collisions}",
                        flush=True,
                    )

                rollout_slots = [
                    slot for slot in rollout_slots
                    if slot not in finished_slots
                ]

                evaluation_due = bool(
                    evaluation_every > 0
                    and (
                        completed_episodes >= next_evaluation_episode
                        or completed_episodes >= episodes
                    )
                )
                if evaluation_due:
                    evaluated_stage = curriculum.current_stage
                    evaluation_data = evaluate_policy(
                        ctde_agent,
                        max_steps=max_steps,
                        seed=seed + 100000,
                        num_episodes=evaluation_episodes,
                        curriculum_stage=evaluated_stage,
                    )
                    if evaluated_stage < len(CURRICULUM_STAGES) - 1:
                        evaluation_data['full_mission_evaluation'] = (
                            evaluate_policy(
                                ctde_agent,
                                max_steps=max_steps,
                                seed=seed + 200000,
                                num_episodes=evaluation_episodes,
                                curriculum_stage=(
                                    len(CURRICULUM_STAGES) - 1
                                ),
                            )
                        )
                    gate_record = curriculum.record_evaluation(
                        evaluation_data, completed_episodes
                    )
                    epsilon = training_epsilon_at_step(
                        global_step, curriculum.current_stage_episodes
                    )
                    evaluation_data['curriculum_gate'] = gate_record
                    evaluation_data['training_episode'] = completed_episodes
                    evaluation_data['global_step'] = global_step
                    data.evaluation_history.append(evaluation_data)
                    print(
                        f"  Greedy S{evaluated_stage} | delivered "
                        f"{evaluation_data['mean_delivered']:.2f}/"
                        f"{CURRICULUM_STAGES[evaluated_stage]['max_patients']} | "
                        f"triage "
                        f"{evaluation_data['mean_triage_efficiency']:.3f} | "
                        f"lower-floor "
                        f"{evaluation_data['mean_lower_triage_delivery_floor']:.3f} | "
                        f"priority "
                        f"{evaluation_data['mean_acuity_priority_score']:.3f} | "
                        f"priority-fairness "
                        f"{evaluation_data['mean_priority_fairness_attainment']:.3f} | "
                        f"ordering(rate/response/combined) "
                        f"{evaluation_data['mean_triage_delivery_rate_ordering_score']:.3f}/"
                        f"{evaluation_data['mean_triage_response_time_ordering_score']:.3f}/"
                        f"{evaluation_data['mean_triage_delivery_ordering_score']:.3f} | "
                        f"response W3/W1 "
                        f"{evaluation_data['mean_response_time_w3']:.1f}/"
                        f"{evaluation_data['mean_response_time_w1']:.1f} | "
                        f"avoidance(w/ls) "
                        f"{evaluation_data['aggregate_wind_avoidance_rate']:.3f}/"
                        f"{evaluation_data['aggregate_low_signal_avoidance_rate']:.3f} | "
                        f"rational-entry(w/ls) "
                        f"{evaluation_data['aggregate_wind_rational_entry_fraction']:.3f}/"
                        f"{evaluation_data['aggregate_low_signal_rational_entry_fraction']:.3f} | "
                        f"collision-rate "
                        f"{max(evaluation_data['mean_obstacle_collision_rate_per_operational_step'], evaluation_data['mean_agent_collision_rate_per_operational_step']):.4f} | "
                        f"landed {evaluation_data['mean_landed']:.2f}/"
                        f"{NUM_AGENTS} | energy-free/progress "
                        f"{evaluation_data['battery_depletion_free_rate']:.3f}/"
                        f"{evaluation_data['mean_energy_return_progress_rate']:.3f} | "
                        f"gate pass="
                        f"{gate_record['passed']} | promoted="
                        f"{gate_record['promoted']}",
                        flush=True,
                    )
                    while next_evaluation_episode <= completed_episodes:
                        next_evaluation_episode += evaluation_every

                checkpoint_due = bool(
                    completed_episodes >= next_checkpoint_episode
                    or completed_episodes >= episodes
                    or stop_requested
                )
                if checkpoint_due:
                    save_training_data()
                    save_checkpoint(completed_episodes, epsilon, global_step)
                    while next_checkpoint_episode <= completed_episodes:
                        next_checkpoint_episode += checkpoint_every

                while (not stop_requested
                       and started_episodes < episodes
                       and len(rollout_slots) < rollout_workers):
                    used_stream_ids = {
                        slot['stream_id'] for slot in rollout_slots
                    }
                    stream_id = next(
                        candidate for candidate in range(rollout_workers)
                        if candidate not in used_stream_ids
                    )
                    rollout_slots.append(create_rollout_slot(stream_id))

            if stop_requested:
                save_training_data()
                save_checkpoint(completed_episodes, epsilon, global_step)
                print("Graceful stop requested; saved completed episodes.")
                break

    serial_episode_start = (
        episodes if rollout_workers > 1 else resumed_completed_episodes
    )
    for episode in range(serial_episode_start, episodes):
        episode_curriculum_stage = curriculum.select_training_stage()
        curriculum_probability = (
            1.0 if curriculum.current_stage == len(CURRICULUM_STAGES) - 1
            else (
                CURRICULUM_CURRENT_PROBABILITY
                + (
                    CURRICULUM_PREVIOUS_PROBABILITY
                    if curriculum.current_stage == 0 else 0.0
                )
                if episode_curriculum_stage == curriculum.current_stage
                else CURRICULUM_PREVIOUS_PROBABILITY
                if episode_curriculum_stage == max(0, curriculum.current_stage - 1)
                else CURRICULUM_FULL_PROBABILITY
            )
        )
        states = env.reset(curriculum_stage=episode_curriculum_stage)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        episode_local_reward_totals = np.zeros(NUM_AGENTS, dtype=np.float64)
        episode_local_potential_rewards = np.zeros(NUM_AGENTS, dtype=np.float64)
        total_reward = 0.0
        steps = 0
        episode_wind_entries = [0] * NUM_AGENTS
        episode_wind_exposure_steps = [0] * NUM_AGENTS
        episode_wind_exits = [0] * NUM_AGENTS
        episode_wind_failures = [0] * NUM_AGENTS
        episode_low_signal_entries = [0] * NUM_AGENTS
        episode_low_signal_exposure_steps = [0] * NUM_AGENTS
        episode_low_signal_exits = [0] * NUM_AGENTS
        episode_low_signal_failures = [0] * NUM_AGENTS
        episode_operational_steps = [0] * NUM_AGENTS
        episode_movement_actions = 0
        episode_wind_command_attempts = 0
        episode_low_signal_command_attempts = 0
        episode_wind_avoidance_opportunities = 0
        episode_wind_hazard_selections = 0
        episode_wind_dominated_hazard_selections = 0
        episode_wind_shortcut_hazard_selections = 0
        episode_low_signal_avoidance_opportunities = 0
        episode_low_signal_hazard_selections = 0
        episode_low_signal_dominated_hazard_selections = 0
        episode_low_signal_shortcut_hazard_selections = 0
        episode_wind_entry_progress_cells = 0.0
        episode_low_signal_entry_progress_cells = 0.0
        episode_wind_zone_refreshes = 0
        episode_low_signal_zone_refreshes = 0
        episode_wind_refresh_onsets = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_low_signal_refresh_onsets = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_agent_delivered = [False] * NUM_AGENTS
        episode_deliveries_by_agent = [0] * NUM_AGENTS
        episode_obstacle_collisions = 0
        episode_obstacle_collision_steps = 0
        episode_obstacle_opportunities = np.zeros(NUM_AGENTS, dtype=np.int64)
        episode_obstacle_actions_selected = np.zeros(NUM_AGENTS, dtype=np.int64)
        episode_dominated_obstacle_selections = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_dominated_agent_conflict_selections = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_obstacle_collisions_by_agent = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_obstacle_collision_events = []
        episode_agent_collisions = 0
        episode_same_destination_collisions = 0
        episode_head_on_collisions = 0
        episode_collision_steps = 0
        episode_rescue_collisions = 0
        episode_landing_collisions = 0
        episode_collision_pair_matrix = np.zeros(
            (NUM_AGENTS, NUM_AGENTS), dtype=np.int64
        )
        episode_collisions_by_agent = np.zeros(NUM_AGENTS, dtype=np.int64)
        episode_action_counts = np.zeros(
            (NUM_AGENTS, ACTION_DIM), dtype=np.int64
        )
        episode_collision_recovery_actions = np.zeros(
            (NUM_AGENTS, ACTION_DIM), dtype=np.int64
        )
        episode_obstacle_recovery_actions = np.zeros(
            (NUM_AGENTS, ACTION_DIM), dtype=np.int64
        )
        previous_step_collision_flags = [0] * NUM_AGENTS
        previous_step_obstacle_flags = [0] * NUM_AGENTS
        episode_action_histories = [[] for _ in range(NUM_AGENTS)]
        episode_valid_action_counts = []
        episode_minimum_agent_distance = 2 * GRID_SIZE
        episode_maximum_collision_streak = 0
        episode_maximum_agent_collision_streak = 0
        episode_maximum_obstacle_collision_streak = 0
        episode_phase_steps = {'rescue': 0, 'landing': 0, 'irrecoverable': 0}
        episode_phase_action_counts = {
            phase: np.zeros((NUM_AGENTS, ACTION_DIM), dtype=np.int64)
            for phase in ('rescue', 'landing', 'irrecoverable')
        }
        episode_phase_obstacle_opportunities = {
            phase: np.zeros(NUM_AGENTS, dtype=np.int64)
            for phase in ('rescue', 'landing', 'irrecoverable')
        }
        episode_phase_obstacle_actions_selected = {
            phase: np.zeros(NUM_AGENTS, dtype=np.int64)
            for phase in ('rescue', 'landing', 'irrecoverable')
        }
        episode_phase_obstacle_collisions = {
            phase: np.zeros(NUM_AGENTS, dtype=np.int64)
            for phase in ('rescue', 'landing', 'irrecoverable')
        }
        episode_landing_progress_available = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_landing_distance_reduced = np.zeros(NUM_AGENTS, dtype=np.int64)
        episode_landing_distance_increased = np.zeros(NUM_AGENTS, dtype=np.int64)
        episode_landing_distance_unchanged = np.zeros(NUM_AGENTS, dtype=np.int64)
        episode_landing_hover_with_progress = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_patient_events = []
        episode_landing_events = []
        episode_landing_zone_arrival_events = []
        episode_landing_zone_departure_events = []
        episode_landing_only_action_states = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_forced_terminal_landing_actions = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_landing_time_from_zone_arrival = [-1] * NUM_AGENTS
        episode_battery_depletion_events = []
        episode_dead_landing_events = []
        episode_death_penalty_applications = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_death_reminder_penalty_applications = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_battery_drain = np.zeros(NUM_AGENTS, dtype=np.float64)
        episode_wind_battery_drain = np.zeros(NUM_AGENTS, dtype=np.float64)
        episode_landing_standby_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
        episode_energy_return_mode_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
        episode_energy_return_progress_steps = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_energy_return_regress_steps = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_energy_return_activations = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        previous_episode_energy_return_flags = np.zeros(
            NUM_AGENTS, dtype=np.int64
        )
        episode_energy_margin_delta = np.zeros(
            NUM_AGENTS, dtype=np.float64
        )
        episode_reserve_violation_steps = np.zeros(NUM_AGENTS, dtype=np.int64)
        episode_minimum_safe_return_margin = np.full(
            NUM_AGENTS, np.inf, dtype=np.float64
        )
        episode_death_shaping_guards = 0
        episode_potential_deltas = []
        episode_raw_potential_changes = []
        episode_reward_components = {
            name: 0.0 for name in data.reward_component_names
        }
        episode_phase_reward_components = {
            phase: {
                name: 0.0 for name in data.reward_component_names
            }
            for phase in ('rescue', 'landing', 'irrecoverable')
        }
        episode_inference_ms = []
        episode_environment_ms = []
        episode_update_ms = []

        for step in range(max_steps + env.landing_grace_steps):
            steps += 1

            inference_start_ns = time.perf_counter_ns()
            actions = ctde_agent.select_actions(states, epsilon)
            episode_valid_action_counts.extend(
                states['action_masks'].sum(axis=1).tolist()
            )
            for agent_index in range(NUM_AGENTS):
                if not states['drones'][agent_index, 3] \
                        and not states['drones'][agent_index, 4]:
                    episode_operational_steps[agent_index] += 1
            for agent_index, action in enumerate(actions):
                episode_action_counts[agent_index, action] += 1
                episode_action_histories[agent_index].append(action)
                if previous_step_collision_flags[agent_index]:
                    episode_collision_recovery_actions[
                        agent_index, action
                    ] += 1
                if previous_step_obstacle_flags[agent_index]:
                    episode_obstacle_recovery_actions[
                        agent_index, action
                    ] += 1
            
            
            
            episode_inference_ms.append(
                (time.perf_counter_ns() - inference_start_ns) / 1_000_000.0
            )

            next_states, rewards, done, step_data = env.step(actions)
            episode_environment_ms.append(step_data['environment_step_latency_ms'])

            for delivery_event in step_data['patient_delivery_events']:
                agent_index = delivery_event['agent']
                episode_agent_delivered[agent_index] = True
                episode_deliveries_by_agent[agent_index] += 1
                episode_patient_events.append({
                    'type': 'delivery', **delivery_event
                })
            for death_event in step_data['patient_death_events']:
                episode_patient_events.append({'type': 'death', **death_event})
            for escalation_event in step_data[
                    'patient_weight_escalation_events']:
                episode_patient_events.append({
                    'type': 'triage_escalation', **escalation_event
                })
            for patient_index in step_data['patient_spawn_events']:
                episode_patient_events.append({
                    'type': 'spawn',
                    'patient': patient_index,
                    'step': env.episode_step,
                })
            episode_landing_events.extend(step_data['landing_events'])
            episode_landing_zone_arrival_events.extend(
                step_data['landing_zone_arrival_events']
            )
            episode_landing_zone_departure_events.extend(
                step_data['landing_zone_departure_events']
            )
            episode_landing_only_action_states += np.asarray(
                step_data['landing_only_action_flags'], dtype=np.int64
            )
            episode_forced_terminal_landing_actions += np.asarray(
                step_data['forced_terminal_landing_actions'], dtype=np.int64
            )
            for landing_event in step_data['landing_events']:
                if landing_event['successful']:
                    episode_landing_time_from_zone_arrival[
                        landing_event['agent']
                    ] = landing_event['steps_from_zone_arrival']
            episode_battery_depletion_events.extend(
                step_data['battery_depletion_events']
            )
            episode_dead_landing_events.extend(
                step_data['dead_landing_events']
            )
            episode_death_penalty_applications += np.asarray(
                step_data['death_penalty_applications'], dtype=np.int64
            )
            episode_death_reminder_penalty_applications += np.asarray(
                step_data[
                    'death_reminder_penalty_applications'
                ], dtype=np.int64
            )
            step_battery_drain = np.asarray(
                step_data['battery_drain_by_agent'], dtype=np.float64
            )
            episode_battery_drain += step_battery_drain
            episode_wind_battery_drain += np.asarray(
                step_data['wind_battery_drain_by_agent'], dtype=np.float64
            )
            episode_landing_standby_steps += np.asarray(
                step_data['landing_standby_steps'], dtype=np.int64
            )
            episode_energy_return_flags = np.asarray(
                step_data['energy_return_mode_flags'], dtype=np.int64
            )
            episode_energy_return_mode_steps += episode_energy_return_flags
            episode_energy_return_progress_steps += np.asarray(
                step_data['energy_return_progress_flags'], dtype=np.int64
            )
            episode_energy_return_regress_steps += np.asarray(
                step_data['energy_return_regress_flags'], dtype=np.int64
            )
            episode_energy_return_activations += (
                (episode_energy_return_flags > 0)
                & (previous_episode_energy_return_flags == 0)
            ).astype(np.int64)
            previous_episode_energy_return_flags = (
                episode_energy_return_flags
            )
            episode_energy_margin_delta += np.asarray(
                step_data['energy_margin_delta_by_agent'], dtype=np.float64
            )
            episode_reserve_violation_steps += np.asarray(
                step_data['reserve_violation_flags'], dtype=np.int64
            )
            operational_battery_mask = step_battery_drain > 0.0
            step_safe_return_margins = np.asarray(
                step_data['safe_return_margin_after'], dtype=np.float64
            )
            episode_minimum_safe_return_margin[
                operational_battery_mask
            ] = np.minimum(
                episode_minimum_safe_return_margin[operational_battery_mask],
                step_safe_return_margins[operational_battery_mask],
            )
            
            for i in range(NUM_AGENTS):
                episode_local_reward_totals[i] += step_data['local_rewards'][i]
                episode_local_potential_rewards[i] += step_data[
                    'local_potential_rewards'
                ][i]
                episode_wind_entries[i] += step_data['wind_entries'][i]
                episode_wind_exposure_steps[i] += step_data[
                    'wind_exposure_steps'
                ][i]
                episode_wind_exits[i] += step_data['wind_exits'][i]
                episode_wind_failures[i] += step_data['wind_failures'][i]
                episode_low_signal_entries[i] += step_data['low_signal_entries'][i]
                episode_low_signal_exposure_steps[i] += step_data[
                    'low_signal_exposure_steps'
                ][i]
                episode_low_signal_exits[i] += step_data[
                    'low_signal_exits'
                ][i]
                episode_low_signal_failures[i] += step_data[
                    'low_signal_failures'
                ][i]
            episode_movement_actions += int(step_data['movement_actions'])
            episode_wind_command_attempts += int(
                step_data['wind_command_attempts']
            )
            episode_low_signal_command_attempts += int(
                step_data['low_signal_command_attempts']
            )
            episode_wind_avoidance_opportunities += int(
                step_data['wind_avoidance_opportunities']
            )
            episode_wind_hazard_selections += int(
                step_data['wind_hazard_selections']
            )
            episode_wind_dominated_hazard_selections += int(
                step_data['wind_dominated_hazard_selections']
            )
            episode_wind_shortcut_hazard_selections += int(
                step_data['wind_shortcut_hazard_selections']
            )
            episode_low_signal_avoidance_opportunities += int(
                step_data['low_signal_avoidance_opportunities']
            )
            episode_low_signal_hazard_selections += int(
                step_data['low_signal_hazard_selections']
            )
            episode_low_signal_dominated_hazard_selections += int(
                step_data['low_signal_dominated_hazard_selections']
            )
            episode_low_signal_shortcut_hazard_selections += int(
                step_data['low_signal_shortcut_hazard_selections']
            )
            episode_wind_entry_progress_cells += float(
                step_data['wind_entry_progress_cells']
            )
            episode_low_signal_entry_progress_cells += float(
                step_data['low_signal_entry_progress_cells']
            )
            episode_wind_zone_refreshes += int(
                step_data['wind_zone_refreshed']
            )
            episode_low_signal_zone_refreshes += int(
                step_data['low_signal_zone_refreshed']
            )
            episode_wind_refresh_onsets += np.asarray(
                step_data['wind_refresh_onset_agents'], dtype=np.int64
            )
            episode_low_signal_refresh_onsets += np.asarray(
                step_data['low_signal_refresh_onset_agents'], dtype=np.int64
            )
            episode_obstacle_collisions += step_data['obstacle_collisions']
            episode_obstacle_collision_steps += int(
                step_data['obstacle_collisions'] > 0
            )
            episode_obstacle_opportunities += np.asarray(
                step_data['obstacle_action_opportunities'], dtype=np.int64
            )
            episode_obstacle_actions_selected += np.asarray(
                step_data['obstacle_action_selected'], dtype=np.int64
            )
            episode_dominated_obstacle_selections += np.asarray(
                step_data['dominated_obstacle_selections'], dtype=np.int64
            )
            episode_dominated_agent_conflict_selections += np.asarray(
                step_data['dominated_agent_conflict_selections'],
                dtype=np.int64,
            )
            episode_obstacle_collisions_by_agent += np.asarray(
                step_data['obstacle_collision_flags'], dtype=np.int64
            )
            remaining_obstacle_event_slots = max(
                0,
                MAX_RECORDED_OBSTACLE_EVENTS_PER_EPISODE
                - len(episode_obstacle_collision_events),
            )
            if remaining_obstacle_event_slots:
                episode_obstacle_collision_events.extend(
                    step_data['obstacle_collision_events'][
                        :remaining_obstacle_event_slots
                    ]
                )
            episode_agent_collisions += step_data['agent_collisions']
            episode_same_destination_collisions += step_data[
                'same_destination_collisions'
            ]
            episode_head_on_collisions += step_data['head_on_collisions']
            episode_collision_steps += int(step_data['agent_collisions'] > 0)
            episode_maximum_collision_streak = max(
                episode_maximum_collision_streak,
                step_data['max_collision_streak']
            )
            episode_maximum_agent_collision_streak = max(
                episode_maximum_agent_collision_streak,
                step_data['max_agent_collision_streak']
            )
            episode_maximum_obstacle_collision_streak = max(
                episode_maximum_obstacle_collision_streak,
                step_data['max_obstacle_collision_streak']
            )
            episode_phase_steps[step_data['phase_before']] += 1
            phase_before = step_data['phase_before']
            for agent_index, action in enumerate(actions):
                episode_phase_action_counts[
                    phase_before
                ][agent_index, action] += 1
            episode_phase_obstacle_opportunities[phase_before] += np.asarray(
                step_data['obstacle_action_opportunities'], dtype=np.int64
            )
            episode_phase_obstacle_actions_selected[phase_before] += np.asarray(
                step_data['obstacle_action_selected'], dtype=np.int64
            )
            episode_phase_obstacle_collisions[phase_before] += np.asarray(
                step_data['obstacle_collision_flags'], dtype=np.int64
            )
            episode_landing_progress_available += np.asarray(
                step_data['landing_progress_actions_available'], dtype=np.int64
            )
            episode_landing_distance_reduced += np.asarray(
                step_data['landing_distance_reduced'], dtype=np.int64
            )
            episode_landing_distance_increased += np.asarray(
                step_data['landing_distance_increased'], dtype=np.int64
            )
            episode_landing_distance_unchanged += np.asarray(
                step_data['landing_distance_unchanged'], dtype=np.int64
            )
            episode_landing_hover_with_progress += np.asarray(
                step_data['landing_hover_with_progress_available'], dtype=np.int64
            )
            if step_data['phase_before'] == 'rescue':
                episode_rescue_collisions += step_data['agent_collisions']
            elif step_data['phase_before'] == 'landing':
                episode_landing_collisions += step_data['agent_collisions']
            if step_data['minimum_agent_distance'] >= 0:
                episode_minimum_agent_distance = min(
                    episode_minimum_agent_distance,
                    step_data['minimum_agent_distance']
                )
            for pair in step_data['collision_pairs']:
                episode_collision_pair_matrix[pair[0], pair[1]] += 1
                episode_collision_pair_matrix[pair[1], pair[0]] += 1
            episode_collisions_by_agent += np.asarray(
                step_data['agent_collision_flags'], dtype=np.int64
            )
            previous_step_collision_flags = step_data['agent_collision_flags']
            previous_step_obstacle_flags = step_data[
                'obstacle_collision_flags'
            ]
            episode_death_shaping_guards += step_data[
                'death_positive_shaping_prevented'
            ]
            episode_potential_deltas.append(step_data['potential_delta'])
            episode_raw_potential_changes.append(
                step_data['raw_potential_change']
            )
            for component_name, component_value in step_data['reward_components'].items():
                episode_reward_components[component_name] += component_value
                episode_phase_reward_components[phase_before][
                    component_name
                ] += component_value

            global_step += 1
            transition_done = bool(
                done or step == max_steps + env.landing_grace_steps - 1
            )
            ctde_agent.push(
                states,
                actions,
                step_data['team_reward'],
                step_data['local_rewards'],
                step_data['event_flags'],
                next_states,
                transition_done,
                curriculum_stage=episode_curriculum_stage,
                stream_id=0,
            )
            total_reward += step_data['team_reward']

            if global_step % TRAIN_EVERY_STEPS == 0:
                update_start_ns = time.perf_counter_ns()
                for _ in range(UPDATES_PER_TRAIN):
                    ctde_agent.train_step(
                        BATCH_SIZE,
                        global_step,
                        current_stage=curriculum.current_stage,
                    )
                
                
                episode_update_ms.append(
                    (time.perf_counter_ns() - update_start_ns) / 1_000_000.0
                )

            states = next_states
            epsilon = training_epsilon_at_step(
                global_step, curriculum.current_stage_episodes
            )

            if not headless and episode % 1000 == 0:
                
                clock.tick(60)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        save_training_data()
                        return

            if done:
                break

        patients_delivered_count = sum(env.patients_actually_delivered)
        patients_died_count      = sum(env.patients_died)
        patients_spawned_count = sum(env.patient_active)

        triage_data = TrainingEpisodeTracker.triage_metrics(env)

        curriculum.record_training_episode(episode_curriculum_stage)
        curriculum_success = bool(
            episode_curriculum_stage < len(CURRICULUM_STAGES) - 1
            and env.mission_success()
        )
        
        
        success = (
            env.episode_mode == "full_mission"
            and env.mission_success()
        )
        learning_data = ctde_agent.get_metrics()
        learning_data['per_beta'] = ctde_agent.beta_at_step(global_step)

        
        
        
        complexity_data = {
            'inference_latency_ms': float(np.mean(episode_inference_ms)),
            'environment_step_latency_ms': float(np.mean(episode_environment_ms)),
            'training_update_latency_ms': (
                float(np.mean(episode_update_ms)) if episode_update_ms else 0.0
            ),
            'total_decision_latency_ms': float(np.mean(episode_inference_ms) +
                                               np.mean(episode_environment_ms)),
            'process_rss_mb': float(get_peak_rss_mb()),
        }
        action_switches = []
        two_action_oscillations = []
        for action_history in episode_action_histories:
            action_switches.append(sum(
                action_history[index] != action_history[index - 1]
                for index in range(1, len(action_history))
            ))
            two_action_oscillations.append(sum(
                action_history[index] == action_history[index - 2]
                and action_history[index] != action_history[index - 1]
                for index in range(2, len(action_history))
            ))
        episode_diagnostics = {
            'episode_mode': env.episode_mode,
            'curriculum_stage': int(episode_curriculum_stage),
            'curriculum_stage_name': env.curriculum_stage_name,
            'hazard_penalty_scale': float(env.hazard_penalty_scale),
            'episode_patient_timer': int(env.episode_patient_timer),
            'patient_spawn_interval': int(env.patient_spawn_interval),
            'patient_spawn_jitter': int(env.patient_spawn_jitter),
            'patient_spawn_batch_range': [
                int(env.minimum_patient_spawn_batch),
                int(env.maximum_patient_spawn_batch),
            ],
            'final_patient_spawn_step': int(
                env.final_patient_spawn_step
            ),
            'curriculum_controller_stage': int(curriculum.current_stage),
            'full_mission_success': bool(success),
            'curriculum_success': bool(curriculum_success),
            'curriculum_probability': float(curriculum_probability),
            'curriculum_max_distance': int(env.curriculum_max_distance),
            'curriculum_max_landing_distance': int(
                env.curriculum_max_landing_distance
            ),
            'curriculum_start_step': int(env.curriculum_start_step),
            'curriculum_initial_distances': list(
                env.curriculum_initial_distances
            ),
            'termination_reason': env.termination_reason,
            'mission_outcome': env.mission_outcome_metrics(),
            'rescue_success': bool(env.rescue_success()),
            'perfect_rescue': bool(env.perfect_rescue()),
            'safe_return_complete': bool(env.safe_return_complete()),
            'landing_deadline_step': int(env.landing_deadline),
            'landing_completion_step': int(env.landing_completion_step),
            'start_positions': [list(position) for position in env.start_positions],
            'patient_positions': [
                list(position) for position in env.patient_positions
            ],
            'landing_zones': [list(position) for position in env.landing_zones],
            'deliveries_by_agent': episode_deliveries_by_agent,
            'local_reward_totals': episode_local_reward_totals.tolist(),
            'local_potential_reward_totals': (
                episode_local_potential_rewards.tolist()
            ),
            'local_reward_sum_error': float(
                episode_local_reward_totals.sum() - total_reward
            ),
            'collisions_by_agent': episode_collisions_by_agent.tolist(),
            'obstacle_collision_steps': episode_obstacle_collision_steps,
            'obstacle_action_opportunities_by_agent': (
                episode_obstacle_opportunities.tolist()
            ),
            'obstacle_actions_selected_by_agent': (
                episode_obstacle_actions_selected.tolist()
            ),
            'dominated_obstacle_selections_by_agent': (
                episode_dominated_obstacle_selections.tolist()
            ),
            'dominated_agent_conflict_selections_by_agent': (
                episode_dominated_agent_conflict_selections.tolist()
            ),
            'obstacle_collisions_by_agent': (
                episode_obstacle_collisions_by_agent.tolist()
            ),
            'obstacle_collision_events': episode_obstacle_collision_events,
            'collision_pair_matrix': episode_collision_pair_matrix.tolist(),
            'same_destination_collisions': episode_same_destination_collisions,
            'head_on_collisions': episode_head_on_collisions,
            'collision_steps': episode_collision_steps,
            'rescue_collisions': episode_rescue_collisions,
            'landing_collisions': episode_landing_collisions,
            'maximum_collision_streak': episode_maximum_collision_streak,
            'maximum_agent_collision_streak': (
                episode_maximum_agent_collision_streak
            ),
            'maximum_obstacle_collision_streak': (
                episode_maximum_obstacle_collision_streak
            ),
            'minimum_agent_distance': (
                episode_minimum_agent_distance
                if episode_minimum_agent_distance < 2 * GRID_SIZE else -1
            ),
            'action_counts': episode_action_counts.tolist(),
            'collision_recovery_action_counts': (
                episode_collision_recovery_actions.tolist()
            ),
            'obstacle_recovery_action_counts': (
                episode_obstacle_recovery_actions.tolist()
            ),
            'phase_action_counts': {
                phase: counts.tolist()
                for phase, counts in episode_phase_action_counts.items()
            },
            'phase_obstacle_action_opportunities_by_agent': {
                phase: counts.tolist()
                for phase, counts in (
                    episode_phase_obstacle_opportunities.items()
                )
            },
            'phase_obstacle_actions_selected_by_agent': {
                phase: counts.tolist()
                for phase, counts in (
                    episode_phase_obstacle_actions_selected.items()
                )
            },
            'phase_obstacle_collisions_by_agent': {
                phase: counts.tolist()
                for phase, counts in episode_phase_obstacle_collisions.items()
            },
            'phase_reward_components': episode_phase_reward_components,
            'action_switches': action_switches,
            'two_action_oscillations': two_action_oscillations,
            'operational_steps_by_agent': episode_operational_steps,
            'operational_agent_steps': int(sum(episode_operational_steps)),
            'movement_actions': int(episode_movement_actions),
            'wind_command_attempts': int(episode_wind_command_attempts),
            'low_signal_command_attempts': int(
                episode_low_signal_command_attempts
            ),
            'obstacle_collision_rate_per_operational_step': float(
                episode_obstacle_collisions
                / max(1, sum(episode_operational_steps))
            ),
            'agent_collision_rate_per_operational_step': float(
                episode_agent_collisions
                / max(1, sum(episode_operational_steps))
            ),
            'mean_valid_actions': float(np.mean(episode_valid_action_counts)),
            'minimum_valid_actions': int(min(episode_valid_action_counts)),
            'wind_entries_by_agent': episode_wind_entries,
            'wind_exposure_steps_by_agent': episode_wind_exposure_steps,
            'wind_exits_by_agent': episode_wind_exits,
            'wind_failures_by_agent': episode_wind_failures,
            'low_signal_entries_by_agent': episode_low_signal_entries,
            'low_signal_exposure_steps_by_agent': (
                episode_low_signal_exposure_steps
            ),
            'low_signal_exits_by_agent': episode_low_signal_exits,
            'low_signal_failures_by_agent': episode_low_signal_failures,
            'wind_exposure_rate': float(
                sum(episode_wind_exposure_steps)
                / max(1, sum(episode_operational_steps))
            ),
            'wind_failure_rate': float(
                sum(episode_wind_failures)
                / max(1, episode_wind_command_attempts)
            ),
            'low_signal_exposure_rate': float(
                sum(episode_low_signal_exposure_steps)
                / max(1, sum(episode_operational_steps))
            ),
            'low_signal_failure_rate': float(
                sum(episode_low_signal_failures)
                / max(1, episode_low_signal_command_attempts)
            ),
            'wind_avoidance_opportunities': int(
                episode_wind_avoidance_opportunities
            ),
            'wind_hazard_selections': int(episode_wind_hazard_selections),
            'wind_dominated_hazard_selections': int(
                episode_wind_dominated_hazard_selections
            ),
            'wind_shortcut_hazard_selections': int(
                episode_wind_shortcut_hazard_selections
            ),
            'wind_avoidance_rate': float(
                1.0 - episode_wind_hazard_selections
                / max(1, episode_wind_avoidance_opportunities)
            ),
            'wind_dominated_avoidance_rate': float(
                1.0 - episode_wind_dominated_hazard_selections
                / max(1, episode_wind_avoidance_opportunities)
            ),
            'wind_rational_entry_fraction': float(
                episode_wind_shortcut_hazard_selections
                / max(1, episode_wind_hazard_selections)
            ),
            'low_signal_avoidance_opportunities': int(
                episode_low_signal_avoidance_opportunities
            ),
            'low_signal_hazard_selections': int(
                episode_low_signal_hazard_selections
            ),
            'low_signal_dominated_hazard_selections': int(
                episode_low_signal_dominated_hazard_selections
            ),
            'low_signal_shortcut_hazard_selections': int(
                episode_low_signal_shortcut_hazard_selections
            ),
            'low_signal_avoidance_rate': float(
                1.0 - episode_low_signal_hazard_selections
                / max(1, episode_low_signal_avoidance_opportunities)
            ),
            'low_signal_dominated_avoidance_rate': float(
                1.0 - episode_low_signal_dominated_hazard_selections
                / max(1, episode_low_signal_avoidance_opportunities)
            ),
            'low_signal_rational_entry_fraction': float(
                episode_low_signal_shortcut_hazard_selections
                / max(1, episode_low_signal_hazard_selections)
            ),
            'wind_entry_progress_cells': float(
                episode_wind_entry_progress_cells
            ),
            'low_signal_entry_progress_cells': float(
                episode_low_signal_entry_progress_cells
            ),
            'wind_progress_cells_per_entry': float(
                episode_wind_entry_progress_cells
                / max(1, sum(episode_wind_entries))
            ),
            'low_signal_progress_cells_per_entry': float(
                episode_low_signal_entry_progress_cells
                / max(1, sum(episode_low_signal_entries))
            ),
            'wind_zone_refreshes': int(episode_wind_zone_refreshes),
            'low_signal_zone_refreshes': int(
                episode_low_signal_zone_refreshes
            ),
            'wind_refresh_onsets_by_agent': (
                episode_wind_refresh_onsets.tolist()
            ),
            'low_signal_refresh_onsets_by_agent': (
                episode_low_signal_refresh_onsets.tolist()
            ),
            'phase_steps': episode_phase_steps,
            'landing_progress_actions_available_by_agent': (
                episode_landing_progress_available.tolist()
            ),
            'landing_distance_reduced_by_agent': (
                episode_landing_distance_reduced.tolist()
            ),
            'landing_distance_increased_by_agent': (
                episode_landing_distance_increased.tolist()
            ),
            'landing_distance_unchanged_by_agent': (
                episode_landing_distance_unchanged.tolist()
            ),
            'landing_hover_with_progress_available_by_agent': (
                episode_landing_hover_with_progress.tolist()
            ),
            'first_delivery_step': env.first_delivery_step,
            'last_delivery_step': env.last_delivery_step,
            'all_patients_resolved_step': env.all_patients_resolved_step,
            'irrecoverable_step': env.irrecoverable_step,
            'agent_path_lengths': env.agent_path_lengths,
            'agent_unique_cells': [
                len(positions) for positions in env.agent_unique_positions
            ],
            'patient_spawn_steps': env.patient_spawn_steps,
            'patient_resolution_steps': env.patient_resolution_steps,
            'patient_initial_timers': env.patient_initial_timers,
            'patient_time_to_resolution_ratio': [
                (
                    (env.patient_resolution_steps[patient_index]
                     - env.patient_spawn_steps[patient_index])
                    / max(1, env.patient_initial_timers[patient_index])
                    if env.patient_spawn_steps[patient_index] >= 0
                    and env.patient_resolution_steps[patient_index] >= 0
                    else -1.0
                )
                for patient_index in range(MAX_PATIENTS)
            ],
            'patient_delivery_agents': env.patient_delivery_agents,
            'initial_patient_weights': env.initial_patient_weights,
            'final_patient_weights': env.patient_weights,
            'patient_survival_probabilities': (
                env.patient_survival_probabilities
            ),
            'patient_decay_rates': env.patient_decay_rates,
            'patient_survival_offsets': env.patient_survival_offsets,
            'patient_serious_thresholds': env.patient_serious_thresholds,
            'patient_critical_thresholds': env.patient_critical_thresholds,
            'wind_rectangles': [list(value) for value in env.wind_rectangles],
            'low_signal_rectangles': [
                list(value) for value in env.low_signal_rectangles
            ],
            'hazard_route_challenges': [
                dict(value) for value in env.hazard_route_challenges
            ],
            'triage_service_state': env.triage_service_state(),
            'patient_time_to_resolution': [
                (
                    env.patient_resolution_steps[patient_index]
                    - env.patient_spawn_steps[patient_index]
                    if env.patient_spawn_steps[patient_index] >= 0
                    and env.patient_resolution_steps[patient_index] >= 0
                    else -1
                )
                for patient_index in range(MAX_PATIENTS)
            ],
            'patient_events': episode_patient_events,
            'landing_events': episode_landing_events,
            'landing_zone_arrival_events': (
                episode_landing_zone_arrival_events
            ),
            'landing_zone_departure_events': (
                episode_landing_zone_departure_events
            ),
            'landing_only_action_states_by_agent': (
                episode_landing_only_action_states.tolist()
            ),
            'forced_terminal_landing_actions_by_agent': (
                episode_forced_terminal_landing_actions.tolist()
            ),
            'landing_time_from_zone_arrival_by_agent': (
                episode_landing_time_from_zone_arrival
            ),
            'battery_depletion_events': episode_battery_depletion_events,
            'dead_landing_events': episode_dead_landing_events,
            'death_penalty_applications_by_agent': (
                episode_death_penalty_applications.tolist()
            ),
            'death_reminder_penalty_applications_by_agent': (
                episode_death_reminder_penalty_applications.tolist()
            ),
            'battery_drain_by_agent': episode_battery_drain.tolist(),
            'wind_battery_drain_by_agent': (
                episode_wind_battery_drain.tolist()
            ),
            'landing_standby_steps_by_agent': (
                episode_landing_standby_steps.tolist()
            ),
            'energy_return_mode_steps_by_agent': (
                episode_energy_return_mode_steps.tolist()
            ),
            'energy_return_progress_steps_by_agent': (
                episode_energy_return_progress_steps.tolist()
            ),
            'energy_return_regress_steps_by_agent': (
                episode_energy_return_regress_steps.tolist()
            ),
            'energy_return_activations_by_agent': (
                episode_energy_return_activations.tolist()
            ),
            'energy_margin_delta_by_agent': (
                episode_energy_margin_delta.tolist()
            ),
            'reserve_violation_steps_by_agent': (
                episode_reserve_violation_steps.tolist()
            ),
            'minimum_safe_return_margin_by_agent': [
                float(value) if math.isfinite(value) else 0.0
                for value in episode_minimum_safe_return_margin
            ],
            'total_battery_drain': float(episode_battery_drain.sum()),
            'total_wind_battery_drain': float(
                episode_wind_battery_drain.sum()
            ),
            'wind_energy_fraction': float(
                episode_wind_battery_drain.sum()
                / max(1e-8, episode_battery_drain.sum())
            ),
            'battery_units_per_delivery': float(
                episode_battery_drain.sum()
                / max(1, patients_delivered_count)
            ),
            'weighted_deliveries_per_battery_unit': float(
                triage_data['weighted_delivered']
                / max(1e-8, episode_battery_drain.sum())
            ),
            'energy_return_mode_fraction': float(
                episode_energy_return_mode_steps.sum()
                / max(1, sum(episode_operational_steps))
            ),
            'energy_return_progress_rate': float(
                episode_energy_return_progress_steps.sum()
                / max(1, episode_energy_return_mode_steps.sum())
            ),
            'energy_return_regress_rate': float(
                episode_energy_return_regress_steps.sum()
                / max(1, episode_energy_return_mode_steps.sum())
            ),
            'energy_return_success_rate': float(
                sum(
                    episode_energy_return_activations[agent_index] > 0
                    and env.landed[agent_index]
                    for agent_index in range(NUM_AGENTS)
                ) / max(
                    1,
                    sum(episode_energy_return_activations > 0),
                )
            ),
            'mean_landing_battery': float(np.mean([
                env.batteries[agent_index]
                for agent_index in range(NUM_AGENTS)
                if env.landed[agent_index]
            ])) if any(env.landed) else 0.0,
            'minimum_landing_battery': float(min([
                env.batteries[agent_index]
                for agent_index in range(NUM_AGENTS)
                if env.landed[agent_index]
            ])) if any(env.landed) else 0.0,
            'reserve_violation_rate': float(
                episode_reserve_violation_steps.sum()
                / max(1, sum(episode_operational_steps))
            ),
            'death_positive_shaping_prevented': episode_death_shaping_guards,
            'potential_delta_mean': float(np.mean(episode_potential_deltas)),
            'potential_delta_min': float(np.min(episode_potential_deltas)),
            'potential_delta_max': float(np.max(episode_potential_deltas)),
            'positive_potential_steps': int(sum(
                delta > 0.0 for delta in episode_potential_deltas
            )),
            'negative_potential_steps': int(sum(
                delta < 0.0 for delta in episode_potential_deltas
            )),
            'raw_potential_change_mean': float(np.mean(
                episode_raw_potential_changes
            )),
            'raw_potential_change_min': float(np.min(
                episode_raw_potential_changes
            )),
            'raw_potential_change_max': float(np.max(
                episode_raw_potential_changes
            )),
            'raw_progress_steps': int(sum(
                delta > 1e-8 for delta in episode_raw_potential_changes
            )),
            'final_potential_components': env.fleet_potential_components(),
            'final_batteries': [float(battery) for battery in env.batteries],
            'final_landed': [bool(value) for value in env.landed],
            'final_battery_depleted': [
                bool(value) for value in env.battery_depleted
            ],
            'final_drone_died': [
                bool(value) for value in env.drone_died
            ],
            'global_step': global_step,
            'learner_steps': ctde_agent.learner_steps,
            'gpu_memory_allocated_mb': (
                float(torch.cuda.memory_allocated(device) / (1024.0 ** 2))
                if device.type == 'cuda' else 0.0
            ),
            'gpu_memory_reserved_mb': (
                float(torch.cuda.memory_reserved(device) / (1024.0 ** 2))
                if device.type == 'cuda' else 0.0
            ),
            'gpu_peak_memory_allocated_mb': (
                float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
                if device.type == 'cuda' else 0.0
            ),
        }

        data.log_episode(
            episode                  = episode,
            total_reward             = total_reward,
            success                  = success,
            agent_delivered          = episode_agent_delivered,
            patients_delivered_count = patients_delivered_count,
            patients_died_count      = patients_died_count,
            patients_spawned_count   = patients_spawned_count,
            landed                   = env.landed,
            steps                    = steps,
            collisions_obs           = episode_obstacle_collisions,
            collisions_ag            = episode_agent_collisions,
            wind_entries             = episode_wind_entries,
            low_signal_entries       = episode_low_signal_entries,
            epsilon                  = epsilon,
            batteries                = env.batteries,
            triage_data              = triage_data,
            complexity_data          = complexity_data,
            learning_data            = learning_data,
            reward_component_totals  = episode_reward_components,
            episode_diagnostics      = episode_diagnostics,
            episode_mode             = env.episode_mode,
            curriculum_max_distance  = env.curriculum_max_distance,
            curriculum_max_landing_distance = (
                env.curriculum_max_landing_distance
            ),
            curriculum_initial_distances = env.curriculum_initial_distances,
            curriculum_success       = curriculum_success,
            curriculum_start_step    = env.curriculum_start_step,
        )
        append_metrics_journal(learning_data)

        print(
            f"Ep {episode+1:>5} | Reward: {total_reward:>8.2f} | "
            f"Mode: {env.episode_mode:>18} | "
            f"ε: {epsilon:.3f} | Landed: {env.landed} | "
            f"Spawned: {patients_spawned_count}/{MAX_PATIENTS} | "
            f"Delivered: {patients_delivered_count}/{patients_spawned_count} | "
            f"Died: {patients_died_count}/{patients_spawned_count} | "
            f"Steps: {steps} | "
            f"End: {env.termination_reason} | "
            f"Loss/Qtot: {learning_data['loss']:.4f}/{learning_data['q_total']:.2f} | "
            f"Wind: {episode_wind_entries} | LS: {episode_low_signal_entries} | "
            f"Collisions(obs/ag): {episode_obstacle_collisions}/{episode_agent_collisions}"
        )

        completed_episodes = episode + 1
        if (evaluation_every > 0
                and (completed_episodes % evaluation_every == 0
                     or completed_episodes == episodes)):
            evaluated_stage = curriculum.current_stage
            evaluation_data = evaluate_policy(
                ctde_agent,
                max_steps=max_steps,
                seed=seed + 100000,
                num_episodes=evaluation_episodes,
                curriculum_stage=evaluated_stage,
            )
            if evaluated_stage < len(CURRICULUM_STAGES) - 1:
                evaluation_data['full_mission_evaluation'] = evaluate_policy(
                    ctde_agent,
                    max_steps=max_steps,
                    seed=seed + 200000,
                    num_episodes=evaluation_episodes,
                    curriculum_stage=len(CURRICULUM_STAGES) - 1,
                )
            gate_record = curriculum.record_evaluation(
                evaluation_data, completed_episodes
            )
            epsilon = training_epsilon_at_step(
                global_step, curriculum.current_stage_episodes
            )
            evaluation_data['curriculum_gate'] = gate_record
            evaluation_data['training_episode'] = completed_episodes
            evaluation_data['global_step'] = global_step
            previous_evaluation = (
                data.evaluation_history[-1]
                if data.evaluation_history else None
            )
            data.evaluation_history.append(evaluation_data)
            evaluated_patient_count = CURRICULUM_STAGES[
                evaluated_stage
            ]['max_patients']
            print(
                f"  Greedy Eval S{evaluated_stage} | Delivered: "
                f"{evaluation_data['mean_delivered']:.2f}/"
                f"{evaluated_patient_count} | "
                f"Died: {evaluation_data['mean_died']:.2f} | "
                f"Triage: {evaluation_data['mean_triage_efficiency']:.3f} | "
                f"Priority Fairness: "
                f"{evaluation_data['mean_priority_fairness_attainment']:.3f} | "
                f"Ordering(rate/response/combined): "
                f"{evaluation_data['mean_triage_delivery_rate_ordering_score']:.3f}/"
                f"{evaluation_data['mean_triage_response_time_ordering_score']:.3f}/"
                f"{evaluation_data['mean_triage_delivery_ordering_score']:.3f} | "
                f"Response W3/W1: "
                f"{evaluation_data['mean_response_time_w3']:.1f}/"
                f"{evaluation_data['mean_response_time_w1']:.1f} | "
                f"Hazard Avoid(w/ls): "
                f"{evaluation_data['aggregate_wind_avoidance_rate']:.3f}/"
                f"{evaluation_data['aggregate_low_signal_avoidance_rate']:.3f} | "
                f"Rational Entry(w/ls): "
                f"{evaluation_data['aggregate_wind_rational_entry_fraction']:.3f}/"
                f"{evaluation_data['aggregate_low_signal_rational_entry_fraction']:.3f} | "
                f"Landed: {evaluation_data['mean_landed']:.2f}/{NUM_AGENTS} | "
                f"Energy Free/Progress: "
                f"{evaluation_data['battery_depletion_free_rate']:.3f}/"
                f"{evaluation_data['mean_energy_return_progress_rate']:.3f} | "
                f"Success: {100.0 * evaluation_data['success_rate']:.1f}% | "
                f"Collisions(obs/ag): "
                f"{evaluation_data['mean_obstacle_collisions']:.2f}/"
                f"{evaluation_data['mean_agent_collisions']:.2f}"
            )
            print(
                f"  Curriculum Gate | delivery "
                f"{gate_record['delivery_rate']:.3f}/"
                f"{gate_record['delivery_gate']:.3f}, triage "
                f"{gate_record['triage_rate']:.3f}/"
                f"{gate_record['triage_gate']:.3f}, return "
                f"{gate_record['landing_rate']:.3f}/"
                f"{gate_record['landing_gate']:.3f}, response "
                f"{gate_record['response_ordering']:.3f}/"
                f"{gate_record['response_ordering_gate']:.3f}, energy-free "
                f"{gate_record['battery_depletion_free_rate']:.3f}/"
                f"{gate_record['battery_depletion_free_gate']:.3f}, passes "
                f"{gate_record['consecutive_passes']}/"
                f"{CURRICULUM_REQUIRED_PASSES}, promoted="
                f"{gate_record['promoted']}"
            )
            if (previous_evaluation is not None
                    and previous_evaluation.get('curriculum_stage')
                    == evaluation_data.get('curriculum_stage')):
                print(
                    f"  Eval Change | Delivered: "
                    f"{evaluation_data['mean_delivered'] - previous_evaluation['mean_delivered']:+.2f} | "
                    f"Triage: "
                    f"{evaluation_data['mean_triage_efficiency'] - previous_evaluation['mean_triage_efficiency']:+.3f} | "
                    f"Agent Collisions: "
                    f"{evaluation_data['mean_agent_collisions'] - previous_evaluation['mean_agent_collisions']:+.2f}"
                )
            if (curriculum.current_stage_episodes >=
                    CURRICULUM_STAGES[curriculum.current_stage][
                        'maximum_episodes'
                    ] and not gate_record['passed']):
                print(
                    "  CONVERGENCE WARNING: the current curriculum stage has "
                    "exceeded its expected episode budget without passing "
                    "the competence gate. Inspect evaluation_history.",
                    flush=True
                )
        if (completed_episodes % checkpoint_every == 0
                or completed_episodes == episodes or stop_requested):
            save_training_data()
            save_checkpoint(completed_episodes, epsilon, global_step)
        if stop_requested:
            print("Graceful stop requested; finalizing saved results.", flush=True)
            break

    pygame.quit()

    
    print("\nFINAL STATISTICS")
    print("="*60)
    total_ep      = len(data.episodes)
    full_mission_indices = [
        index for index, mode in enumerate(data.episode_modes)
        if mode == "full_mission"
    ]
    full_mission_successes = [
        data.success_rate[index] for index in full_mission_indices
    ]
    final_full_mission_successes = full_mission_successes[-100:]
    overall_succ = (
        100.0 * sum(full_mission_successes) / len(full_mission_successes)
        if full_mission_successes else 0.0
    )
    final_succ = (
        100.0 * sum(final_full_mission_successes)
        / len(final_full_mission_successes)
        if final_full_mission_successes else 0.0
    )
    avg_rew_all   = np.mean(data.total_rewards)
    avg_rew_final = np.mean(data.total_rewards[-100:])
    avg_patients = (
        np.mean([
            data.patients_delivered_counts[index]
            for index in full_mission_indices
        ]) if full_mission_indices else 0.0
    )
    avg_died = (
        np.mean([
            data.patients_died_counts[index]
            for index in full_mission_indices
        ]) if full_mission_indices else 0.0
    )

    
    
    total_training_seconds = time.perf_counter() - training_start
    if profile_memory:
        python_current_bytes, python_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        python_current_bytes, python_peak_bytes = 0, 0
    data.complexity_summary = {
        'device': str(device),
        'total_training_time_seconds': float(total_training_seconds),
        'training_time_per_episode_seconds': float(total_training_seconds / total_ep),
        'average_inference_latency_ms': float(np.mean(data.inference_latency_ms)),
        'p95_inference_latency_ms': float(np.percentile(data.inference_latency_ms, 95)),
        'average_environment_step_latency_ms': float(np.mean(data.environment_step_latency_ms)),
        'average_training_update_latency_ms': float(np.mean(data.training_update_latency_ms)),
        'average_total_decision_latency_ms': float(np.mean(data.total_decision_latency_ms)),
        'peak_process_rss_mb': float(max(data.process_rss_mb)),
        'peak_python_allocated_mb': float(python_peak_bytes / (1024.0 ** 2)),
        'policy_parameter_count': int(model_parameters),
        'policy_tensor_size_mb': float(model_size_mb),
        'replay_allocated_mb': float(ctde_agent.joint_buffer.allocated_mb()),
        'replay_final_size': int(len(ctde_agent.joint_buffer)),
        'learner_steps': int(ctde_agent.learner_steps),
        'environment_steps': int(global_step),
        'train_every_steps': int(TRAIN_EVERY_STEPS),
        'updates_per_train': int(UPDATES_PER_TRAIN),
        'replay_samples_per_environment_step': float(
            BATCH_SIZE * UPDATES_PER_TRAIN / TRAIN_EVERY_STEPS
        ),
        'full_mission_episode_count': int(len(full_mission_indices)),
        'landing_curriculum_episode_count': int(
            total_ep - len(full_mission_indices)
        ),
        'mixed_precision': bool(ctde_agent.use_mixed_precision),
        'fused_optimizer': bool(ctde_agent.uses_fused_optimizer),
        'tensor_core_math': bool(tensor_core_math),
        'evaluation_every': int(evaluation_every),
        'evaluation_episodes': int(evaluation_episodes),
        'evaluation_checkpoints': int(len(data.evaluation_history)),
        'python_memory_profiling_enabled': bool(profile_memory),
    }
    data.run_configuration['finished_at'] = datetime.now().isoformat()
    data.run_configuration['stopped_by_signal'] = bool(stop_requested)
    data.run_configuration['environment_steps_completed'] = int(global_step)
    data.run_configuration['learner_steps_completed'] = int(
        ctde_agent.learner_steps
    )

    print(f"Total Episodes:                {total_ep}")
    print(f"Full-Mission/Curriculum Eps:   "
          f"{len(full_mission_indices)}/{total_ep - len(full_mission_indices)}")
    print(f"Overall Mission Success Rate:  {overall_succ:.2f}%")
    print(f"Final 100 Mission Success:     {final_succ:.2f}%")
    print(f"Average Reward (All):          {avg_rew_all:.2f}")
    print(f"Average Reward (Final 100):    {avg_rew_final:.2f}")
    print(f"Avg Patients Delivered/Ep:     {avg_patients:.2f}")
    print(f"Avg Patients Died/Ep:          {avg_died:.2f}")
    total_wind_entries = sum(
        sum(getattr(data, f'wind_entries_agent{agent_index}'))
        for agent_index in range(NUM_AGENTS)
    )
    total_low_signal_entries = sum(
        sum(getattr(data, f'low_signal_entries_agent{agent_index}'))
        for agent_index in range(NUM_AGENTS)
    )
    print(f"Total Wind Entries:            {total_wind_entries}")
    print(f"Total Low Signal Entries:      {total_low_signal_entries}")
    print(f"Total Collisions (obs/ag):     {sum(data.collisions_obstacles)}/{sum(data.collisions_agents)}")
    print("-"*60)
    print(f"COMPUTATIONAL FEASIBILITY (joint {NUM_AGENTS}-agent decision)")
    print(f"Total Training Time:           {total_training_seconds:.2f} s")
    print(f"Training Time/Episode:         {total_training_seconds/total_ep:.4f} s")
    print(f"Average Inference Latency:     {np.mean(data.inference_latency_ms):.4f} ms")
    print(f"P95 Inference Latency:         {np.percentile(data.inference_latency_ms, 95):.4f} ms")
    print(f"Environment Step Overhead:     {np.mean(data.environment_step_latency_ms):.4f} ms")
    print(f"Training Update Overhead:      {np.mean(data.training_update_latency_ms):.4f} ms")
    print(f"Online Decision Overhead:      {np.mean(data.total_decision_latency_ms):.4f} ms")
    print(f"Peak Process Memory (RSS):     {max(data.process_rss_mb):.2f} MiB")
    print(f"Peak Python Allocations:       {python_peak_bytes/(1024.0**2):.2f} MiB")
    print(f"Policy Parameters:             {model_parameters:,}")
    print(f"Policy Tensor Size:            {model_size_mb:.3f} MiB")
    print(f"Replay Allocation:             {ctde_agent.joint_buffer.allocated_mb():.2f} MiB")
    print("="*60)

    
    
    save_training_data()

    model_path = output_dir_path / "ctde_agent_marl_FGCS.pth"
    temporary_model_path = model_path.with_suffix(".pth.tmp")
  
    torch.save({
        'model_version': MODEL_VERSION,
        'policy_state_dict': ctde_agent.policy_net.state_dict(),
        'mixer_state_dict': ctde_agent.mixer.state_dict(),
        'target_state_dict': ctde_agent.target_net.state_dict(),
        'target_mixer_state_dict': ctde_agent.target_mixer.state_dict(),
        'optimizer_state_dict': ctde_agent.optimizer.state_dict(),
        'global_step': global_step,
        'learner_steps': ctde_agent.learner_steps,
        'num_agents': NUM_AGENTS,
        'grid_size': GRID_SIZE,
        'max_patients': MAX_PATIENTS,
        'action_dim': action_dim,
        'action_names': ACTION_NAMES,
        'drone_state_dim': DRONE_STATE_DIM,
        'patient_state_dim': PATIENT_STATE_DIM,
        'local_grid_radius': LOCAL_GRID_RADIUS,
        'local_grid_size': LOCAL_GRID_SIZE,
        'gamma': GAMMA,
        'n_step': N_STEP,
        'battery_units': 'percentage_points',
        'maximum_battery': MAX_BATTERY,
        'battery_drain_per_step': BATTERY_DRAIN_PER_STEP,
        'battery_drain_in_wind': BATTERY_DRAIN_IN_WIND,
        'battery_drain_at_landing_zone': BATTERY_DRAIN_AT_LANDING_ZONE,
        'clean_endurance_steps': (
            MAX_BATTERY / BATTERY_DRAIN_PER_STEP
        ),
        'post_depletion_reminder_steps': POST_DEPLETION_REMINDER_STEPS,
        'post_depletion_reminder_penalty': (
            POST_DEPLETION_REMINDER_PENALTY
        ),
        'dead_landing_penalty': DEAD_LANDING_PENALTY,
        'episode_continues_after_drone_death': True,
        'pre_resolution_energy_return_landing': True,
        'per_drone_died_observation': True,
        'low_battery_threshold': LOW_BATTERY_THRESHOLD,
        'safe_return_battery_buffer': SAFE_RETURN_BATTERY_BUFFER,
        'energy_return_trigger_margin': ENERGY_RETURN_TRIGGER_MARGIN,
        'return_energy_risk_multiplier': RETURN_ENERGY_RISK_MULTIPLIER,
        'wind_signal_expected_cost_return_map': True,
        'single_return_safety_reserve': True,
        'route_intercepting_hazards': True,
        'dense_priority_service_debt_reward': True,
        'priority_service_potential_scale': (
            PRIORITY_SERVICE_POTENTIAL_SCALE
        ),
        'counterfactual_collision_selection_penalties': True,
        'curriculum_epsilon_reset': CURRICULUM_EPSILON_RESET,
        'curriculum_epsilon_floor': CURRICULUM_EPSILON_FLOOR,
        'curriculum_epsilon_decay_episodes': (
            CURRICULUM_EPSILON_DECAY_EPISODES
        ),
        'energy_usage_penalty_per_unit': ENERGY_USAGE_PENALTY_PER_UNIT,
        'safe_return_reserve_penalty': SAFE_RETURN_RESERVE_PENALTY,
        'energy_margin_shaping_factor': ENERGY_MARGIN_SHAPING_FACTOR,
        'response_wait_penalty_per_patient': (
            RESPONSE_WAIT_PENALTY_PER_PATIENT
        ),
        'response_time_delivery_reward': RESPONSE_TIME_DELIVERY_REWARD,
        'energy_standby_hover_penalty': ENERGY_STANDBY_HOVER_PENALTY,
        'entity_embed_dim': ENTITY_EMBED_DIM,
        'attention_heads': ATTENTION_HEADS,
        'set_attention_blocks': SET_ATTENTION_BLOCKS,
        'mixer_embed_dim': MIXER_EMBED_DIM,
        'agent_id_embed_dim': AGENT_ID_EMBED_DIM,
        'central_grid_embed_dim': CENTRAL_GRID_EMBED_DIM,
        'agent_mix_context_dim': AGENT_MIX_CONTEXT_DIM,
        'mission_state_dim': MISSION_STATE_DIM,
        'batch_size': BATCH_SIZE,
        'train_every_steps': TRAIN_EVERY_STEPS,
        'updates_per_train': UPDATES_PER_TRAIN,
        'mixed_precision': ctde_agent.use_mixed_precision,
        'fused_optimizer': ctde_agent.uses_fused_optimizer,
        'tensor_core_math': tensor_core_math,
        'td_reward_scale': TD_REWARD_SCALE,
        'local_td_loss_weight': LOCAL_TD_LOSS_WEIGHT,
        'per_alpha': PER_ALPHA,
        'per_priority_max': PER_PRIORITY_MAX,
        'replay_uniform_fraction': REPLAY_UNIFORM_FRACTION,
        'replay_rescue_fraction': REPLAY_RESCUE_FRACTION,
        'replay_landing_fraction': REPLAY_LANDING_FRACTION,
        'mixer_min_raw_weight': MIXER_MIN_RAW_WEIGHT,
        'epsilon_is_per_agent': True,
        'safe_patient_spawn_deadline': True,
        'boundary_invalid_action_masking': False,
        'obstacle_action_masking': False,
        'occupied_cell_action_masking': False,
        'learned_obstacle_and_agent_avoidance': True,
        'action_conditioned_entity_attention': True,
        'phase_specific_dueling_heads': True,
        'fp32_action_value_heads': True,
        'phase_balanced_replay': True,
        'stage_and_phase_balanced_replay': True,
        'obstacle_aware_landing_distance': True,
        'constrained_energy_budget': True,
        'obstacle_aware_energy_return_reserve': True,
        'pre_resolution_energy_return_shaping': True,
        'candidate_rescue_energy_feasibility': True,
        'low_power_landing_zone_standby': True,
        'connected_random_mission_layout': True,
        'competence_gated_full_curriculum': True,
        'hard_stage_budget_curriculum_promotion': (
            CURRICULUM_FORCE_PROMOTION_AT_MAXIMUM
        ),
        'guaranteed_final_stage_exposure': True,
        'curriculum_stages': CURRICULUM_STAGES,
        'curriculum_state': curriculum.state_dict(),
        'pending_only_local_patient_attention': True,
        'explicit_hover_action': True,
        'terminal_land_only_action_masking': True,
        'energy_return_land_only_action_masking': True,
        'post_death_motion_allowed': False,
        'fleet_death_immediate_termination': False,
        'landing_zone_relative_direction_features': True,
        'irrecoverable_failure_termination': False,
        'post_resolution_landing_phase': True,
        'landing_after_patient_deaths': True,
        'reward_derived_local_td_credit': True,
        'agent_attributed_navigation_potential': True,
        'priority_neutral_unique_coverage_shaping': True,
        'individual_logistic_survival_dynamics': True,
        'expanded_decentralized_hazard_observation': True,
        'directional_hazard_corridor_features': True,
        'decision_aware_hazard_routing_reward': True,
        'triage_response_time_tie_breaking': True,
        'patient_response_age_observation': True,
        'priority_weighted_response_reward': True,
        'progressive_hazard_penalty_curriculum': True,
        'stage_specific_scarcity_curriculum': True,
        'wind_zone_refresh_interval_steps': WIND_APPEAR_INTERVAL,
        'low_signal_zone_refresh_interval_steps': (
            LOW_SIGNAL_APPEAR_INTERVAL
        ),
        'wind_penalty': WIND_PENALTY,
        'low_signal_penalty': LOW_SIGNAL_PENALTY,
        'wind_entry_penalty': WIND_ENTRY_PENALTY,
        'low_signal_entry_penalty': LOW_SIGNAL_ENTRY_PENALTY,
        'wind_dominated_selection_penalty': (
            WIND_DOMINATED_SELECTION_PENALTY
        ),
        'wind_shortcut_selection_penalty': (
            WIND_SHORTCUT_SELECTION_PENALTY
        ),
        'low_signal_dominated_selection_penalty': (
            LOW_SIGNAL_DOMINATED_SELECTION_PENALTY
        ),
        'low_signal_shortcut_selection_penalty': (
            LOW_SIGNAL_SHORTCUT_SELECTION_PENALTY
        ),
        'triage_class_delivery_targets': TRIAGE_CLASS_DELIVERY_TARGETS,
        'priority_fairness_outcome_reward': FAIRNESS_OUTCOME_REWARD,
        'triage_ordering_outcome_reward': (
            TRIAGE_ORDERING_OUTCOME_REWARD
        ),
        'triage_response_outcome_reward': TRIAGE_RESPONSE_OUTCOME_REWARD,
        'hazard_rectangles_may_overlap_obstacles': True,
        'supervised_targets_or_demonstrations': False,
        'evaluation_every': evaluation_every,
        'evaluation_episodes': evaluation_episodes,
    }, temporary_model_path)
    temporary_model_path.replace(model_path)
    print(f"\nCEDA-QMIX model saved: {model_path}")
    print("="*60 + "\n")

if __name__ == '__main__':
    args = parse_args()
    if args.smoke_test:
        
        
        BATCH_SIZE = 8
        BUFFER_CAPACITY = 128
        REPLAY_WARMUP = 8
        TARGET_UPDATE_STEPS = 4
        EVALUATION_EVERY_EPISODES = 1
        EVALUATION_EPISODES = 2
    requested_episodes = 2 if args.smoke_test else args.episodes
    requested_max_steps = min(args.max_steps, 100) if args.smoke_test else args.max_steps
    train(
        headless=args.headless or HEADLESS_MODE,
        output_dir=args.output_dir,
        episodes=requested_episodes,
        max_steps=requested_max_steps,
        seed=args.seed,
        checkpoint_every=1 if args.smoke_test else args.checkpoint_every,
        device_name=args.device,
        fast_math=args.fast_math,
        mixed_precision=not args.no_amp,
        profile_memory=args.profile_memory,
        evaluation_every=(
            EVALUATION_EVERY_EPISODES
            if args.smoke_test else args.evaluation_every
        ),
        evaluation_episodes=(
            EVALUATION_EPISODES
            if args.smoke_test else args.evaluation_episodes
        ),
        rollout_workers=(
            min(2, args.rollout_workers)
            if args.smoke_test else args.rollout_workers
        ),
        resume_checkpoint=args.resume,
    )
