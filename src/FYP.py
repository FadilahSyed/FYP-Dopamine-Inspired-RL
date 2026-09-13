# =============================================================================
# test4.py — Dopamine-Inspired Reinforcement Learning (FYP)
#
# Project   : Dopamine-Inspired RL for Improved Exploration
# Author    : Sharifah Fadilah Syed Azlan  (Student ID: 20509986)
# Supervisor: Tomás Maul
#
# Environments : CartPole-v1, MountainCar-v0, Acrobot-v1
# Conditions   : TD, TD+NOV, TD+UNC, FULL, MULT, GATE
# Seeds        : 30 (configured in config.py)
#
# Dependencies:
#   config.py         — all experiment hyperparameters and output settings
#   analysis_plots.py — all plotting and reporting functions
#
# Usage:
#   python test4.py
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# All experiment settings (SEEDS, CONDITIONS, RUN_ENVS, ENV_CONFIGS, DEVICE, etc.)
from config import *

# All plotting and reporting functions
from analysis_plot import plot_environment_results, print_and_save_environment_tables



# =============================================================================
# NETWORKS
# =============================================================================

# ── Q-Network ────────────────────────────────────────────────────────────────
# two hidden MLP Q-network used for all ensemble heads.
# architecture: state_dim --> 128 --> 128 --> action_dim (linear output)
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ── Random Network Distillation (RND) module for novelty estimation.──────────
#    Consists of:
#      - target:    fixed, randomly initialised network (never updated)
#      - predictor: trainable network that learns to match the target's output
class RNDModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, lr: float = 3e-4):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        tgt = self.target(states)
        pred = self.predictor(states)
        return (pred - tgt).pow(2).mean(dim=1)

    def train_step(self, states: torch.Tensor) -> float:
        with torch.enable_grad():
            loss = self.forward(states).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return float(loss.item())

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

# Hyperparameters for the BootstrapBioAgent  ----------------------------------
# all values are fixed across all conditions and environments
@dataclass
class AgentConfig:
    ensemble_size: int  = 5       # number of Q-network heads (K)
    lr:            float = 3e-4   # base Adam learning rate (scaled by neuromodulation)
    gamma:         float = 0.99   # discount factor for Bellman targets
    replay_size:   int  = 20000   # maximum replay buffer capacity
    batch_size:    int  = 64      # transitions sampled per training step
    mask_prob:     float = 0.8    # Bernoulli probability for bootstrap mask
    grad_clip:     float = 10.0   # max L2 norm for gradient clipping
    alpha_max:     float = 0.05    # max novelty weighting coefficient α
    beta_max:      float = 0.05    # max uncertainty weighting coefficient β
    eps_min:       float = 0.01   # minimum exploration probability (ε floor)
    eps_smooth:    float = 0.01   # EMA smoothing factor for ε updates
    target_update: str   = "hard" # target network update mode: "hard" or "soft"
    tau:           float = 0.01   # soft update interpolation factor (used if mode="soft")


# =============================================================================
# AGENT
# =============================================================================

# Bootstrapped Deep Q-Network with dopamine-inspired bio-RPE modulation -------
#    Core components:
#      - Ensemble of K Q-networks with bootstrapped experience masking
#      - RND module for novelty estimation (δ_nov)
#      - Ensemble disagreement for uncertainty estimation (δ_unc)
#      - Bio-RPE integration: additive, multiplicative, or gated
#      - Neuromodulation: learning rate and ε adapted from |δ_bio|
class BootstrapBioAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        rnd_model: RNDModel,
        cfg: AgentConfig,
        use_novelty: bool = True,
        use_uncertainty: bool = True,
        integration_mode: str = "additive",
    ):
        self.K = cfg.ensemble_size
        self.gamma = cfg.gamma
        self.rnd = rnd_model
        self.cfg = cfg
        self.use_novelty = use_novelty
        self.use_uncertainty = use_uncertainty
        self.integration_mode = integration_mode

        self.q_ensemble = nn.ModuleList([QNetwork(state_dim, action_dim).to(DEVICE) for _ in range(self.K)])
        self.target_ensemble = nn.ModuleList([QNetwork(state_dim, action_dim).to(DEVICE) for _ in range(self.K)])
        for k in range(self.K):
            self.target_ensemble[k].load_state_dict(self.q_ensemble[k].state_dict())

        self.optimizers = [optim.Adam(net.parameters(), lr=cfg.lr) for net in self.q_ensemble]
        self.memory = deque(maxlen=cfg.replay_size)

        self.action_dim = action_dim
        self.epsilon = 1.0
        self.base_epsilon = 1.0
        self.alpha = cfg.alpha_max
        self.beta = cfg.beta_max
        self.current_head = 0

    def begin_episode(self) -> None:
        self.current_head = random.randint(0, self.K - 1)

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            qvals = self.q_ensemble[self.current_head](s)
            return int(qvals.argmax(dim=1).item())

    def store(self, transition: Tuple[np.ndarray, int, float, np.ndarray, float]) -> None:
        self.memory.append(transition)

    def estimate_uncertainty(self, states: torch.Tensor) -> torch.Tensor:
        q_list = [net(states) for net in self.q_ensemble]
        q_stack = torch.stack(q_list, dim=0)
        var = q_stack.var(dim=0).mean(dim=1, keepdim=True)
        mean = var.mean()
        std = var.std()
        if std > 0:
            var = (var - mean) / (std + 1e-6)
        var = torch.relu(var)
        var = torch.clamp(var, max=5.0)
        return var.detach()

    def neuromodulate(self, td_error: torch.Tensor, bio_rpe: torch.Tensor) -> None:
        td_abs = float(td_error.abs().mean().item())
        signal = float(np.tanh(td_abs / 10.0))
        self.alpha = self.cfg.alpha_max * signal
        self.beta = self.cfg.beta_max * signal

        bio_abs = float(bio_rpe.abs().mean().item())
        lr_scale = float(np.clip(0.5 + 0.5 * np.tanh(bio_abs / 10.0), 0.5, 1.0))
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg["lr"] = self.cfg.lr * lr_scale

        bio_signal = float(np.tanh(bio_abs/10.0))
        target_eps = self.cfg.eps_min + (self.base_epsilon - self.cfg.eps_min) * bio_signal
        self.epsilon = (1.0 - self.cfg.eps_smooth) * self.epsilon + self.cfg.eps_smooth * target_eps
        self.epsilon = float(np.clip(self.epsilon, self.cfg.eps_min, 1.0))
        return lr_scale

    def _integrate_bio_rpe(self, td_error: torch.Tensor, novelty: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        if self.integration_mode == "additive":
            return td_error + self.alpha * novelty + self.beta * uncertainty
        if self.integration_mode == "multiplicative":
            nov_n = torch.clamp(novelty / (novelty.mean() + 1e-6), 0.0, 1.0)
            unc_n = torch.clamp(uncertainty / (uncertainty.mean() + 1e-6), 0.0, 1.0)
            scale = 1.0 + self.alpha * nov_n + self.beta * unc_n
            return td_error * scale
        if self.integration_mode == "gated":
            tau = 0.5
            k = 5.0
            gate = torch.sigmoid(k * (uncertainty - tau))
            return td_error + self.alpha * (1 - gate) * novelty + self.beta * gate * uncertainty
        raise ValueError(f"Unknown integration mode: {self.integration_mode}")

    def train_step(self) -> Dict[str, float] | None:
        if len(self.memory) < self.cfg.batch_size:
            return None

        with torch.enable_grad():
            batch = random.sample(self.memory, self.cfg.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states)).to(DEVICE)
            next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
            actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

            all_td_errors = []
            for k in range(self.K):
                mask = torch.bernoulli(torch.full((self.cfg.batch_size, 1), self.cfg.mask_prob, device=DEVICE))
                if mask.sum() == 0:
                    continue

                q_net = self.q_ensemble[k]
                target_net = self.target_ensemble[k]
                opt = self.optimizers[k]

                q_vals = q_net(states).gather(1, actions)

                with torch.no_grad():
                    next_online = q_net(next_states)
                    best_actions = next_online.argmax(dim=1, keepdim=True)
                    next_target = target_net(next_states)
                    next_q = next_target.gather(1, best_actions)
                    td_target = rewards + self.gamma * next_q * (1 - dones)

                td_error_k = td_target - q_vals
                masked_td = td_error_k * mask
                loss_k = masked_td.pow(2).sum() / (mask.sum() + 1e-6)

                opt.zero_grad()
                loss_k.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), self.cfg.grad_clip)
                opt.step()

                all_td_errors.append(td_error_k.detach())

        if not all_td_errors:
            return None

        td_error = torch.stack(all_td_errors, dim=0).mean(dim=0)

        if self.use_novelty:
            novelty = self.rnd.forward(states).unsqueeze(1).detach()
            _ = self.rnd.train_step(states)
        else:
            novelty = torch.zeros_like(td_error)

        if self.use_uncertainty:
            uncertainty = self.estimate_uncertainty(states)
        else:
            uncertainty = torch.zeros_like(td_error)

        bio_rpe = self._integrate_bio_rpe(td_error, novelty, uncertainty)
        lr_scale=self.neuromodulate(td_error, bio_rpe)
        bio_rpe = torch.clamp(bio_rpe, -50.0, 50.0)

        return {
            "td": float(td_error.abs().mean().item()),
            "nov": float(novelty.mean().item()),
            "unc": float(uncertainty.mean().item()),
            "bio": float(bio_rpe.abs().mean().item()),
            "eps": float(self.epsilon),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "lr_scale": float(lr_scale),
        }

    def update_targets(self) -> None:
        if self.cfg.target_update == "hard":
            for k in range(self.K):
                self.target_ensemble[k].load_state_dict(self.q_ensemble[k].state_dict())
        else:
            for k in range(self.K):
                for tgt_p, src_p in zip(self.target_ensemble[k].parameters(), self.q_ensemble[k].parameters()):
                    tgt_p.data.copy_(self.cfg.tau * src_p.data + (1.0 - self.cfg.tau) * tgt_p.data)

# =============================================================================
# HELPERS
# =============================================================================

# Seeds all RNG sources for full reproducibility across runs
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_grad_enabled(True)

# For ablations: Maps a condition name to its agent configuration flags
def parse_condition(condition_name: str) -> Tuple[bool, bool, str]:
    use_nov = condition_name in ["TD+NOV", "FULL", "MULT", "GATE"]
    use_unc = condition_name in ["TD+UNC", "FULL", "MULT", "GATE"]
    integration_mode = {
        "TD": "additive",
        "TD+NOV": "additive",
        "TD+UNC": "additive",
        "FULL": "additive",
        "MULT": "multiplicative",
        "GATE": "gated",
    }[condition_name]
    return use_nov, use_unc, integration_mode

# Evaluation: runs a fixed number of greedy evaluation episodes
def eval_agent(env: gym.Env, agent: BootstrapBioAgent, episodes: int = 5) -> float:
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    scores = []
    for _ in range(episodes):
        s, _ = env.reset()
        agent.begin_episode()
        done = False
        total = 0.0
        while not done:
            a = agent.act(s)
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r
        scores.append(total)
    agent.epsilon = old_eps
    return float(np.mean(scores))

# =============================================================================
# TRAINING LOOP
# =============================================================================
# --- Single seed run ---
def run_one_seed(env_name: str, condition_name: str, seed: int, episodes: int, eval_every: int, eval_episodes: int) -> Dict[str, np.ndarray | List[np.ndarray]]:
    set_seed(seed)
    env = gym.make(env_name)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    use_nov, use_unc, integration_mode = parse_condition(condition_name)

    cfg = AgentConfig()
    rnd = RNDModel(state_dim)
    agent = BootstrapBioAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        rnd_model=rnd,
        cfg=cfg,
        use_novelty=use_nov,
        use_uncertainty=use_unc,
        integration_mode=integration_mode,
    )

    step_count=0
    rewards: List[float] = []
    eval_rewards: List[float] = []
    td_vals: List[float] = []
    nov_vals: List[float] = []
    unc_vals: List[float] = []
    bio_vals: List[float] = []
    eps_vals: List[float] = []
    alpha_vals:    List[float] = []
    beta_vals:     List[float] = []
    lr_scale_vals: List[float] = []
    visited_states: List[np.ndarray] = []

    for ep in range(1, episodes + 1):
        s, _ = env.reset(seed=seed + ep)
        agent.begin_episode()
        done = False
        total = 0.0
        last_metrics = None

        while not done:
            a = agent.act(s)
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            agent.store((s, a, float(r), ns, float(done)))
            step_count+=1
            visited_states.append(ns)
            s = ns
            total += r
            if step_count % 4 == 0:
                m = agent.train_step()
                if m is not None:
                    last_metrics = m

        agent.update_targets()

        rewards.append(total)

        if last_metrics is None:
            td_vals.append(np.nan)
            nov_vals.append(np.nan)
            unc_vals.append(np.nan)
            bio_vals.append(np.nan)
            eps_vals.append(np.nan)
            alpha_vals.append(np.nan)
            beta_vals.append(np.nan)
            lr_scale_vals.append(np.nan)
        else:
            td_vals.append(last_metrics["td"])
            nov_vals.append(last_metrics["nov"])
            unc_vals.append(last_metrics["unc"])
            bio_vals.append(last_metrics["bio"])
            eps_vals.append(last_metrics["eps"])
            alpha_vals.append(last_metrics["alpha"])
            beta_vals.append(last_metrics["beta"])
            lr_scale_vals.append(last_metrics["lr_scale"])

        if eval_every and ep % eval_every == 0:
            eval_rewards.append(eval_agent(env, agent, episodes=eval_episodes))

        if ep % 100 == 0 or ep == episodes:
            print(f"[{env_name} | {condition_name} | seed={seed}] ep={ep} trainR={total:.2f} eps={agent.epsilon:.3f}")

    env.close()
    return {
        "reward": np.array(rewards, dtype=np.float32),
        "eval_reward": np.array(eval_rewards, dtype=np.float32),
        "td": np.array(td_vals, dtype=np.float32),
        "nov": np.array(nov_vals, dtype=np.float32),
        "unc": np.array(unc_vals, dtype=np.float32),
        "bio": np.array(bio_vals, dtype=np.float32),
        "eps": np.array(eps_vals, dtype=np.float32),
        "alpha": np.array(alpha_vals, dtype=np.float32),
        "beta": np.array(beta_vals, dtype=np.float32),
        "lr_scale": np.array(lr_scale_vals, dtype=np.float32),
        "visited_states": visited_states,
    }

# ----- Multi-seed runner -----

def run_condition(env_name: str, condition_name: str, seeds: Tuple[int, ...], episodes: int, eval_every: int, eval_episodes: int) -> Dict[str, np.ndarray | List[List[np.ndarray]]]:
    per_seed = [run_one_seed(env_name, condition_name, sd, episodes, eval_every, eval_episodes) for sd in seeds]
    stacked = {}
    for key in ["reward", "eval_reward", "td", "nov", "unc", "bio", "eps","alpha","beta","lr_scale"]:
        stacked[key] = np.stack([ps[key] for ps in per_seed], axis=0)
    stacked["visited_states"] = [ps["visited_states"] for ps in per_seed]
    return stacked

# =============================================================================
# MAIN
# =============================================================================

# iterates over all environments and conditions defined in config.py
# runs the full multi-seed experiment
# calls plot_environment_results() and print_and_save_environment_tables()
def main() -> None:
    all_results = {}

    for env_name in RUN_ENVS:
        cfg = ENV_CONFIGS[env_name]

        print(f"\n{'#'*90}\nRunning environment: {env_name}\n{'#'*90}")
        env_results = {}

        for cond in CONDITIONS:
            print(f"\n=== Running {env_name} | condition: {cond} ===")
            env_results[cond] = run_condition(
                env_name=env_name,
                condition_name=cond,
                seeds=SEEDS,
                episodes=cfg["episodes"],
                eval_every=cfg["eval_every"],
                eval_episodes=cfg["eval_episodes"],
            )

        all_results[env_name] = env_results
        plot_environment_results(env_name, env_results, cfg["episodes"], cfg["eval_every"])
        print_and_save_environment_tables(env_name, env_results, cfg["episodes"], cfg["eval_every"])

    print(f"\nAll runs complete. Files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

