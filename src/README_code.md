# Dopamine-Inspired Reinforcement Learning for Improved Exploration

**FYP Project — University of Nottingham Malaysia**  
**Author:** Sharifah Fadilah Syed Azlan (Student ID: 20509986)  
**Supervisor:** Tomás Maul  

---

## Overview

This project investigates whether a **dopamine-inspired biological reward prediction error (bio-RPE)** — combining temporal difference error, novelty, and uncertainty into a single unified signal — can improve exploration and learning in deep reinforcement learning agents.

The agent (BootstrapBioAgent) is evaluated across three OpenAI Gymnasium environments under six experimental conditions that ablate and combine three integration modes (additive, multiplicative, gated).

**Biological motivation:** Dopamine neurons in the brain encode not just reward prediction errors but also novelty and uncertainty (Schultz et al., 1997; Wang et al., 2024). This framework attempts to computationally reproduce that composite signal and use it to adaptively modulate both the learning rate and exploration behaviour.

---

## Repository Structure

```
project/src
│
├── FYP.py           # Main experiment script — agent, training loop, entry point
├── analysis_plots.py  # All plotting and reporting functions
├── config.py          # All hyperparameters and experiment settings (edit this first)
└── README.md          # This file
```

---

## File Descriptions

### `config.py`
Central configuration file. **All settings are defined here** — seeds, environments, episode budgets, output switches, and plot switches. Both `FYP.py` and `analysis_plots.py` import from this file, so any change only needs to be made in one place.

Key settings:
| Variable | Default | Description |
|---|---|---|
| `OUTPUT_DIR` | `~/results` | Where all plots and CSVs are saved |
| `DEVICE` | `cpu` | PyTorch device — change to `"cuda"` for GPU |
| `SEEDS` | `tuple(range(30))` | 30 random seeds (0–29) |
| `CONDITIONS` | 6 conditions | TD, TD+NOV, TD+UNC, FULL, MULT, GATE |
| `RUN_ENVS` | 3 environments | CartPole-v1, MountainCar-v0, Acrobot-v1 |
| `ENV_CONFIGS` | per-env dict | Episodes, eval frequency, eval episodes |
| `ROLLING_WINDOW` | 20 | Smoothing window for rolling reward plot |

---

### `FYP.py`
The main experiment script. Contains:

- **`QNetwork`** — two-hidden-layer MLP (128 units, ReLU) used for all ensemble Q-network heads
- **`RNDModel`** — Random Network Distillation module: fixed target network + trainable predictor. Novelty = MSE between predictor and target outputs. Predictor is updated online every training step.
- **`AgentConfig`** — dataclass holding all agent hyperparameters (see table below)
- **`BootstrapBioAgent`** — the main agent class:
  - Ensemble of K=5 Q-networks with bootstrapped experience masking
  - `estimate_uncertainty()` — computes δ_unc from ensemble Q-value variance
  - `neuromodulate()` — adapts learning rate and ε from |δ_bio|
  - `_integrate_bio_rpe()` — combines δ_TD, δ_nov, δ_unc using the selected mode
  - `train_step()` — samples replay buffer, updates all heads, logs all signals
  - `update_targets()` — hard or soft target network update
- **`set_seed()`** — seeds Python, NumPy, and PyTorch for full reproducibility
- **`parse_condition()`** — maps condition name to agent flags (use_novelty, use_uncertainty, integration_mode)
- **`eval_agent()`** — runs greedy evaluation episodes (ε=0), returns mean reward
- **`run_one_seed()`** — runs one full training run for one seed, returns all metrics as arrays
- **`run_condition()`** — runs all seeds for one condition, stacks results into [seeds × episodes] arrays
- **`main()`** — iterates over all environments and conditions, calls plotting and reporting

---

### `analysis_plots.py`
All plotting and reporting functions. Called automatically from `FYP.py` — you do not need to run this file directly.

Contains two main groups of functions:

**Standard plots** (called by `plot_environment_results()`):
- Training reward, rolling training reward, evaluation reward, internal signals, heatmaps, final eval bar chart

**7 analysis plots** (called by `plot_analysis_figures()`):
- Novelty vs. state familiarity scatter, α/β adaptive weights, lr_scale, bio-RPE decomposition, rolling correlation, novelty spaghetti, uncertainty heatmap

---

## Experimental Conditions

| Condition | δ_nov | δ_unc | Integration mode | Description |
|---|---|---|---|---|
| **TD** | ✗ | ✗ | — | Baseline bootstrapped DQN, no intrinsic signals |
| **TD+NOV** | ✓ | ✗ | Additive | Novelty only: δ_bio = δ_TD + α·δ_nov |
| **TD+UNC** | ✗ | ✓ | Additive | Uncertainty only: δ_bio = δ_TD + β·δ_unc |
| **FULL** | ✓ | ✓ | Additive | All signals: δ_bio = δ_TD + α·δ_nov + β·δ_unc |
| **MULT** | ✓ | ✓ | Multiplicative | δ_bio = δ_TD · (1 + α·ñ_nov + β·ũ_unc) |
| **GATE** | ✓ | ✓ | Gated | δ_bio = δ_TD + α·(1−g)·δ_nov + β·g·δ_unc, g = σ(k·(δ_unc − τ)) |

---

## Agent Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `ensemble_size` | 5 | Number of Q-network heads (K) |
| `lr` | 3×10⁻⁴ | Base Adam learning rate (scaled dynamically by neuromodulation) |
| `gamma` | 0.99 | Discount factor |
| `replay_size` | 20,000 | Replay buffer capacity |
| `batch_size` | 64 | Transitions sampled per training step |
| `mask_prob` | 0.8 | Bernoulli probability for bootstrapped mask |
| `grad_clip` | 10.0 | Max gradient L2 norm per head |
| `alpha_max` | 0.05 | Max novelty weighting coefficient α |
| `beta_max` | 0.05 | Max uncertainty weighting coefficient β |
| `eps_min` | 0.01 | Minimum ε (exploration floor) |
| `eps_smooth` | 0.01 | EMA smoothing factor for ε updates |
| `target_update` | hard | Target network update mode (hard = full copy per episode) |
| `tau` | 0.01 | Soft update factor (only used if target_update = "soft") |

---

## Environment Settings

| Environment | Episodes | Eval every | Reward structure | Primary purpose |
|---|---|---|---|---|
| CartPole-v1 | 500 | 25 | +1/step (max 500) | Dense reward validation |
| MountainCar-v0 | 700 | 25 | −1/step (min −200) | Sparse reward / hard exploration |
| Acrobot-v1 | 500 | 25 | −1/step (min −500) | Sparse reward / moderate exploration |

---

## Installation

### Requirements

```bash
pip install gymnasium torch numpy matplotlib scipy
```

Tested with:
- Python 3.10+
- PyTorch 2.x
- Gymnasium 0.29+
- scipy 1.11+

### On a VM / remote server

```bash
# Install dependencies
pip install gymnasium torch numpy matplotlib scipy

# Upload all three files to the same directory
scp FYP.py analysis_plots.py config.py username@your-vm-ip:~/

# Verify files are present
ls ~/ | grep -E "test4|analysis|config"
```

---

## How to Run

### 1. Configure your experiment

Open `config.py` and adjust:

```python
# Change output folder
OUTPUT_DIR = os.path.expanduser("~/results_myrun")

# Change number of seeds 
SEEDS = tuple(range(30))

# Run a quick test with fewer episodes first
ENV_CONFIGS = {
    "CartPole-v1":   {"episodes": 50, "eval_every": 25, "eval_episodes": 5},
    "MountainCar-v0": {"episodes": 50, "eval_every": 25, "eval_episodes": 5},
    "Acrobot-v1":    {"episodes": 50, "eval_every": 25, "eval_episodes": 5},
}
```

### 2. Run the experiment

```bash
python FYP.py
```

### 3. Monitor progress

The script prints progress every 100 episodes per seed:
```
[CartPole-v1 | TD+NOV | seed=0] ep=100 trainR=42.00 eps=0.312
[CartPole-v1 | TD+NOV | seed=0] ep=200 trainR=198.50 eps=0.101
...
```

### 4. Find your results

All outputs are saved to `OUTPUT_DIR` (default: `~/results`):

```bash
ls ~/results/
```

### Run on a single environment only

Edit `config.py`:
```python
RUN_ENVS = ["MountainCar-v0"]   # only run MountainCar
```

### Run on a single condition only

Edit `config.py`:
```python
CONDITIONS = ["TD", "TD+NOV"]   # only run TD baseline and TD+NOV
```

### Disable slow plots

Edit `config.py`:
```python
PLOT_ANALYSIS_FIGURES = False   # skip the 7 analysis plots
PLOT_HEATMAPS         = False   # skip state visitation heatmaps
```

---

## Output Files

All files are saved to `OUTPUT_DIR`. File names follow the pattern `{ENV}_{description}.{ext}`.

### Standard plots (PNG)

| Filename | Description |
|---|---|
| `{ENV}_training_reward.png` | Raw episodic training reward — mean ± std across seeds for all conditions |
| `{ENV}_training_reward_rolling.png` | Smoothed training reward using a rolling average (window = `ROLLING_WINDOW`) |
| `{ENV}_eval_reward.png` | Greedy evaluation reward (ε=0) at every eval checkpoint — mean ± std |
| `{ENV}_final_eval_bar.png` | Bar chart comparing final evaluation reward mean ± std across all conditions |
| `{ENV}_td.png` | TD error \|δ_TD\| over training — mean ± std across seeds |
| `{ENV}_nov.png` | RND novelty signal δ_nov over training — mean ± std across seeds |
| `{ENV}_unc.png` | Ensemble uncertainty δ_unc over training — mean ± std across seeds |
| `{ENV}_bio.png` | Bio-RPE magnitude \|δ_bio\| over training — mean ± std across seeds |
| `{ENV}_eps.png` | Exploration parameter ε over training — mean ± std across seeds |
| `{ENV}_{COND}_heatmap.png` | 2D state visitation heatmap for each condition (one file per condition) |

### Analysis plots (PNG)

| Filename | Description |
|---|---|
| `{ENV}_plot1_novelty_vs_visits.png` | Scatter of δ_nov vs state-bin visit count (seed 0). 2×2 grid for TD+NOV, FULL, MULT, GATE. Negative trend = novelty decays as states become familiar (habituation). Colour = episode number. |
| `{ENV}_plot2_alpha_beta.png` | Adaptive weighting coefficients α(t) and β(t) over training. Both are driven by tanh(\|δ_TD\|/10) and decay as TD error diminishes. Shows how the novelty/uncertainty contribution changes across training. |
| `{ENV}_plot3_lr_scale.png` | Learning rate scale factor lr_scale = clip(0.5 + 0.5·tanh(\|δ_bio\|/10), 0.5, 1.0) over training, alongside \|δ_bio\| for reference. High lr_scale = full plasticity; low = reduced plasticity. |
| `{ENV}_plot4_bio_rpe_decomposition.png` | Stacked area chart per condition showing the three components of \|δ_bio\|: \|δ_TD\| (blue, bottom), \|α·δ_nov\| (orange, middle), \|β·δ_unc\| (green, top). Shows how the balance between components shifts across training. |
| `{ENV}_plot5_bio_reward_correlation.png` | Rolling Pearson correlation between \|δ_bio\|(t) and reward improvement Δreward(t+1) over a 30-episode window. Positive correlation = bio-RPE tracks genuine learning progress. |
| `{ENV}_plot6_novelty_spaghetti.png` | Individual seed novelty trajectories for all novelty-enabled conditions (2×2 grid). Green lines = seeds that solved the environment (final eval above threshold); red dashed = failed/stuck seeds. Threshold: CartPole ≥ 200, MountainCar ≥ −150, Acrobot ≥ −100. |
| `{ENV}_plot7_uncertainty_state_heatmap.png` | MountainCar only. 2-row grid: top row = visit count heatmap (Blues), bottom row = mean δ_unc heatmap (Reds, shared colour scale) for all 6 conditions. Shows that δ_unc is uniformly high in all unvisited regions — not selectively near the goal — explaining why TD+UNC underperforms in sparse environments. |

### CSV files

| Filename | Description |
|---|---|
| `{ENV}_summary_last5.csv` | Mean and std of training and eval reward over the last 5 checkpoints per condition |
| `{ENV}_final_best_eval.csv` | Final eval mean/std and best eval mean/std per condition across all seeds |
| `{ENV}_per_seed_final_best_eval.csv` | Per-seed final and best evaluation scores — one row per seed per condition |
| `{ENV}_detailed_eval_points.csv` | Mean eval reward at every evaluation checkpoint — one row per checkpoint, one column per condition |
| `{ENV}_per_seed_curves.csv` | Full per-seed training and eval curves — one row per (condition, seed, episode) |
| `{ENV}_reward_curve_values.csv` | Mean ± std training reward at every episode |
| `{ENV}_eval_reward_curve_values.csv` | Mean ± std eval reward at every checkpoint |
| `{ENV}_td_curve_values.csv` | Mean ± std TD error at every episode |
| `{ENV}_nov_curve_values.csv` | Mean ± std novelty signal at every episode |
| `{ENV}_unc_curve_values.csv` | Mean ± std uncertainty signal at every episode |
| `{ENV}_bio_curve_values.csv` | Mean ± std bio-RPE at every episode |
| `{ENV}_eps_curve_values.csv` | Mean ± std epsilon at every episode |

---

## Approximate Runtime

These are rough estimates on a standard CPU VM. GPU will be significantly faster.

| Environment | Seeds | Approx. time |
|---|---|---|
| CartPole-v1 (500 ep) | 30 seeds × 6 conditions | ~2–3 hours |
| MountainCar-v0 (700 ep) | 30 seeds × 6 conditions | ~4–6 hours |
| Acrobot-v1 (500 ep) | 30 seeds × 6 conditions | ~2–3 hours |
| **All three environments** | | **~8–12 hours total** |

> **Tip:** Run a quick sanity check first by setting all environments to 50 episodes in `config.py`. The full run should complete in a few minutes.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'analysis_plots'`**  
Make sure `analysis_plots.py` is in the same directory as `FYP.py`. Check with:
```bash
ls ~/ | grep -E "test4|analysis|config"
```

**`ModuleNotFoundError: No module named 'scipy'`**  
Install scipy:
```bash
pip install scipy
```

**`ModuleNotFoundError: No module named 'gymnasium'`**  
Install gymnasium:
```bash
pip install gymnasium
```


**Out of memory / very slow**  
Reduce the number of seeds or disable some conditions in `config.py`:
```python
SEEDS      = tuple(range(10))     # use 10 seeds instead of 30
CONDITIONS = ["TD", "TD+NOV"]    # only run two conditions
```

**`[Plot 2] Error: 'alpha' key missing`**  
The `alpha`, `beta`, and `lr_scale` keys require the updated version of `neuromodulate()` that returns `lr_scale`, and the updated `train_step()` that logs these values. Make sure you are using the latest version of `FYP.py`.

---
