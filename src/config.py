# =============================================================================
# config.py
# =============================================================================
# PURPOSE:
#   Central configuration file for the Dopamine-Inspired RL experiment.
#   All other files (test4.py, analysis_plot.py) import from here,
#   so any change to seeds, environments, episodes, or output settings only
#   ever needs to be made in one place.
#
# USAGE:
#   Add `from config import *` at the top of test4.py and analysis_plot.py.
#
# SECTIONS:
#   1. Output directory: Results (plots, CSVs, tables) are saved here. 
#                        Created automatically if it does not already exist
#   2. Device (CPU/GPU): Controls whether PyTorch uses CPU or GPU ("cpu"=CPU; "cuda"=GPU)
#   3. Experiment settings — seeds, conditions, environments, episode budgets
#   4. Output switches — control what gets saved/printed
#   5. Plot switches — control which figures are generated
# =============================================================================
import os
import torch

# ── Output ────────────────────────────────────────────────
OUTPUT_DIR = os.path.expanduser("~/results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Device ────────────────────────────────────────────────
DEVICE = torch.device("cpu")


# ── Experiment settings ───────────────────────────────────
SEEDS = tuple(range(10))
CONDITIONS = ["TD", "TD+NOV", "TD+UNC", "FULL", "MULT", "GATE"]
RUN_ENVS   = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]

ENV_CONFIGS = {
   "CartPole-v1": {"episodes": 500, "eval_every": 25, "eval_episodes": 5},
   "MountainCar-v0": {"episodes": 700, "eval_every": 25, "eval_episodes": 5},
    "Acrobot-v1": {"episodes": 500, "eval_every": 25, "eval_episodes": 5},
}


# ── Output switches ───────────────────────────────────────
SAVE_PLOTS               = True   # Save all matplotlib figures as .png files
SHOW_PLOTS               = False  # Display plots interactively 
SAVE_SUMMARY_TABLES      = True   # Save summary CSVs (final/best eval per condition)
SAVE_DETAILED_EVAL_TABLES = True  # Save detailed per-checkpoint eval CSVs
SAVE_CURVE_VALUES_CSV    = True   # Save full training/eval curve values as CSVs
PRINT_SUMMARY_TABLES     = True   # Print summary tables to console
PRINT_DETAILED_EVAL_TABLES = True # Print per-checkpoint eval table to console
PRINT_CURVE_VALUES       = False  # Print full curve values to console (very verbose)


# ── Plot switches ─────────────────────────────────────────
PLOT_TRAINING_REWARD         = True   # Raw episodic training reward per condition
PLOT_EVAL_REWARD             = True   # Greedy evaluation reward per checkpoint
PLOT_FINAL_BAR               = True   # Bar chart of final evaluation mean ± std
PLOT_INTERNAL_SIGNALS        = True   # TD error, novelty, uncertainty, bio-RPE, epsilon
PLOT_HEATMAPS                = True   # State visitation heatmaps (2D histograms)
PLOT_ROLLING_TRAINING_REWARD = True   # Smoothed training reward (rolling window)
PLOT_ANALYSIS_FIGURES        = True   # 7 analysis plots

# Rolling window size for the smoothed training reward plot (episodes).
ROLLING_WINDOW            = 20 

# Restric plots to specific environments or conditions
INTERNAL_SIGNAL_ENVS  = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
HEATMAP_ENVS          = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
HEATMAP_CONDITIONS    = ["TD", "TD+NOV", "TD+UNC", "FULL", "MULT", "GATE"]
