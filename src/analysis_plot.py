# =============================================================================
# analysis_plots.py
# All plotting functions for the FYP experiment.
# Imports config from config.py — make sure config.py is in the same directory.
#
# Contains:
#   A) Shared helper functions (save/show, smoothing. state axes, etc)
#   B) Standard plots 
#           1a. Training reward (raw)
#           1b. Training reward (rolling average)
#            2.  Evaluation reward
#          3-7. Internal signals (td, nov, unc, bio, eps)
#            8.  State visitation heatmaps
#            9.  Final evaluation bar chart
#     plot_environment_results()  — training/eval curves, internal signals,
#                                   heatmaps, bar chart, rolling reward
#     print_and_save_environment_tables() — CSV/console reporting
#
#   C) Analysis Plots
#     plot_analysis_figures()     — master call for all 7 below
#     1. plot_novelty_vs_visits()         — novelty vs state familiarity scatter
#     2. plot_alpha_beta()                — adaptive weights α(t) and β(t)
#     3. plot_lr_scale()                  — learning rate scale over training
#     4. plot_bio_rpe_decomposition()     — stacked area bio-RPE components
#     5. plot_bio_rpe_reward_correlation()— rollzing correlation bio-RPE ↔ reward
#     6. plot_novelty_spaghetti()         — per-seed novelty (solved vs stuck)
#     7. plot_uncertainty_state_heatmap() — uncertainty over state space
# =============================================================================

from __future__ import annotations

import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr

#All experiment settings from config.py
from config import (
    OUTPUT_DIR, SEEDS, CONDITIONS,
    SAVE_PLOTS, SHOW_PLOTS,
    SAVE_SUMMARY_TABLES, SAVE_DETAILED_EVAL_TABLES, SAVE_CURVE_VALUES_CSV,
    PRINT_SUMMARY_TABLES, PRINT_DETAILED_EVAL_TABLES, PRINT_CURVE_VALUES,
    PLOT_TRAINING_REWARD, PLOT_EVAL_REWARD, PLOT_FINAL_BAR,
    PLOT_INTERNAL_SIGNALS, PLOT_HEATMAPS,
    PLOT_ROLLING_TRAINING_REWARD, PLOT_ANALYSIS_FIGURES, ROLLING_WINDOW,
    INTERNAL_SIGNAL_ENVS, HEATMAP_ENVS, HEATMAP_CONDITIONS,
)

# ── Colour / style constants ──────────────────────────────────────────────────
CMAP_NAME  = "tab10"
ALPHA_MAX  = 0.05   #should match AgentConfig.alpha_max in test.py    

# Per-environment success thresholds for the spaghetti plot 
SPAGHETTI_THRESHOLDS = {
    "CartPole-v1":    200.0,
    "MountainCar-v0": -150.0,
    "Acrobot-v1":    -100.0,
}


# =============================================================================
# SHARED HELPERS
# =============================================================================

# ---- Saves current matplotlib figure to OUTPUT_DIR and optionally display it
def _save_show(fname: str) -> None:
    if SAVE_PLOTS:
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=160, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ---- Dictionary for conditions and the colours representing them across all plots
def _cond_colors() -> dict:
    cmap = plt.get_cmap(CMAP_NAME)
    return {c: cmap(i / max(len(CONDITIONS) - 1, 1)) for i, c in enumerate(CONDITIONS)}


# ---- NaN-aware uniform box filter: to smooth noisy training signal curves
def _smooth(arr: np.ndarray, w: int = 10) -> np.ndarray:
    arr = arr.copy().astype(float)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = 0.0
    s     = uniform_filter1d(arr, size=w, mode="nearest")
    count = uniform_filter1d((~nan_mask).astype(float), size=w, mode="nearest")
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, s / count, np.nan)


# ---- plots training/eval mean curve with shaded std band
def plot_mean_std(x: np.ndarray, mean: np.ndarray, std: np.ndarray,
                  label: str, color=None) -> None:
    kw = {"label": label}
    if color is not None:
        kw["color"] = color
    plt.plot(x, mean, **kw)
    plt.fill_between(x, mean - std, mean + std, alpha=0.15,
                     **({"color": color} if color else {}))


# ---- Non-NaN aware moving average for plots where NaN values are rare
def _moving_average(arr: np.ndarray, window: int = 20) -> np.ndarray:
    if window <= 1 or len(arr) < window:
        return arr.copy()
    return np.convolve(arr, np.ones(window) / window, mode="same")


# ---- writes numeric results into a CSV file 
def save_csv(filepath: str, header: List[str], rows: List[List]) -> None:
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---- extracts state dimensions for 2D state visitation heatmaps
def get_state_axes(env_name: str, states: np.ndarray):
    if env_name == "CartPole-v1":
        return states[:, 0], states[:, 2], "Cart Position", "Pole Angle"
    if env_name == "MountainCar-v0":
        return states[:, 0], states[:, 1], "Position", "Velocity"
    if env_name == "Acrobot-v1":
        t1 = np.arctan2(states[:, 1], states[:, 0])
        t2 = np.arctan2(states[:, 3], states[:, 2])
        return t1, t2, "theta1", "theta2"
    return states[:, 0], states[:, 1], "State 0", "State 1"


# =============================================================================
# STANDARD PLOTS
# =============================================================================


# ---- generates all standard per-environment plots 
# (training reward (raw & rolling avg), eval reward, internal signals, 
# state visitation heatmaps, final eval bar chart)
# also calls out analysis plots
def plot_environment_results(env_name: str, results: dict,
                              episodes: int, eval_every: int) -> None:
    colors = _cond_colors()
    x      = np.arange(1, episodes + 1)
    eval_x = np.arange(1, (episodes // eval_every) + 1) * eval_every

    # ── 1. Training reward (raw) ───────────────────────────────────────
    if PLOT_TRAINING_REWARD:
        plt.figure(figsize=(11, 5))
        for cond, data in results.items():
            arr = data["reward"]
            plot_mean_std(x, arr.mean(axis=0), arr.std(axis=0),
                          cond, color=colors[cond])
        plt.title(f"{env_name}: Training Reward (mean ± std, {len(SEEDS)} seeds)")
        plt.xlabel("Episode"); plt.ylabel("Reward")
        plt.grid(True); plt.legend(); plt.tight_layout()
        _save_show(f"{env_name}_training_reward.png")

    # ── 1b. Training reward (rolling average) ─────────────────────────
    if PLOT_ROLLING_TRAINING_REWARD:
        plt.figure(figsize=(11, 5))
        for cond, data in results.items():
            arr = data["reward"]
            m = np.convolve(arr.mean(axis=0), np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid")
            s = np.convolve(arr.std(axis=0),  np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid")

            x_roll = x[ROLLING_WINDOW - 1:]

            plt.plot(x_roll, m, label=cond, color=colors[cond])
            plt.fill_between(x_roll, m - s, m + s, alpha=0.15, color=colors[cond])
        plt.title(f"{env_name}: Training Reward (rolling mean, window={ROLLING_WINDOW})")
        plt.xlabel("Episode"); plt.ylabel("Reward")
        plt.grid(True); plt.legend(); plt.tight_layout()
        _save_show(f"{env_name}_training_reward_rolling.png")

    # ── 2. Evaluation reward ──────────────────────────────────────────
    if PLOT_EVAL_REWARD:
        plt.figure(figsize=(11, 5))
        for cond, data in results.items():
            arr = data["eval_reward"]
            plot_mean_std(eval_x, arr.mean(axis=0), arr.std(axis=0),
                          cond, color=colors[cond])
        plt.title(f"{env_name}: Evaluation Reward (ε=0, mean ± std, {len(SEEDS)} seeds)")
        plt.xlabel("Episode"); plt.ylabel("Eval Reward")
        plt.grid(True); plt.legend(); plt.tight_layout()
        _save_show(f"{env_name}_eval_reward.png")

    # ── 3-7. Internal signals ─────────────────────────────────────────
    if PLOT_INTERNAL_SIGNALS and env_name in INTERNAL_SIGNAL_ENVS:
        for key, title in [
            ("td",  "TD Error |δ_TD|"),
            ("nov", "Novelty (RND) δ_nov"),
            ("unc", "Uncertainty δ_unc"),
            ("bio", "Bio-RPE |δ_bio|"),
            ("eps", "Exploration ε"),
        ]:
            plt.figure(figsize=(11, 5))
            for cond, data in results.items():
                arr = data[key]
                plot_mean_std(x, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0),
                              cond, color=colors[cond])
            plt.title(f"{env_name}: {title} (mean ± std, {len(SEEDS)} seeds)")
            plt.xlabel("Episode")
            plt.grid(True); plt.legend(); plt.tight_layout()
            _save_show(f"{env_name}_{key}.png")

    # ── 8. State visitation heatmaps ──────────────────────────────────
    if PLOT_HEATMAPS and env_name in HEATMAP_ENVS:
        for cond, data in results.items():
            if cond not in HEATMAP_CONDITIONS:
                continue
            all_states = []
            for seed_states in data["visited_states"]:
                all_states.extend(seed_states)
            states = np.array(all_states, dtype=np.float32)
            if len(states) == 0:
                continue

            xs, ys, xlabel, ylabel = get_state_axes(env_name, states)
            plt.figure(figsize=(7, 5))
            plt.hist2d(xs, ys, bins=40, cmin=1)
            plt.colorbar(label="Frequency")
            plt.title(f"{env_name}: State Visitation ({cond}, all seeds)")
            plt.xlabel(xlabel); plt.ylabel(ylabel)
            plt.tight_layout()
            _save_show(f"{env_name}_{cond}_heatmap.png")

    # ── 9. Final eval bar chart ───────────────────────────────────────
    if PLOT_FINAL_BAR:
        labels, means, stds = [], [], []
        for cond, data in results.items():
            vals = data["eval_reward"][:, -1]
            labels.append(cond); means.append(vals.mean()); stds.append(vals.std())

        plt.figure(figsize=(9, 5))
        x_pos = np.arange(len(labels))
        bars  = plt.bar(x_pos, means, yerr=stds, capsize=5,
                        color=[colors[c] for c in labels])
        plt.xticks(x_pos, labels, rotation=45)
        plt.title(f"{env_name}: Final Evaluation Reward (mean ± std, {len(SEEDS)} seeds)")
        plt.ylabel("Final Eval Reward")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        _save_show(f"{env_name}_final_eval_bar.png")

    # ── 7 analysis plots ──────────────────────────────────────────
    if PLOT_ANALYSIS_FIGURES:
        plot_analysis_figures(env_name, results, episodes)


# =============================================================================
# Reporting
# =============================================================================

# ---- generates and saves numerical result tables 
def print_and_save_environment_tables(env_name: str, results: dict,
                                       episodes: int, eval_every: int) -> None:
    eval_x = np.arange(1, (episodes // eval_every) + 1) * eval_every

    # ── Summary table: mean of last 5 checkpoints ─────────────────────────────
    rows = []
    if PRINT_SUMMARY_TABLES:
        print(f"\n{'='*70}\n{env_name} — Summary ({len(SEEDS)} seeds)\n{'='*70}")
        print(f"{'Condition':<15} | {'Type':<10} | {'Mean (Last 5)':>20} | {'Std':>9}")
        print("-" * 70)

    for cond, data in results.items():
        for label, key in [("Training", "reward"), ("Eval", "eval_reward")]:
            arr   = data[key]
            m     = arr[:, -5:].mean()
            s     = arr[:, -5:].std()
            rows.append([cond, label, m, s])
            if PRINT_SUMMARY_TABLES:
                print(f"{cond:<15} | {label:<10} | {m:20.2f} | {s:9.2f}")
        if PRINT_SUMMARY_TABLES:
            print("-" * 70)

    if SAVE_SUMMARY_TABLES:
        save_csv(os.path.join(OUTPUT_DIR, f"{env_name}_summary_last5.csv"),
                 ["Condition", "Type", "Mean Reward (Last 5)", "Std Dev"], rows)

    # ── Detailed evaluation table: one row per checkpoint ─────────────────────
    csv_rows = []
    if PRINT_DETAILED_EVAL_TABLES:
        hdr = "Episode".ljust(12) + "".join(f"{c:>14}" for c in results)
        print(f"\n{env_name} — Detailed Evaluation Points:\n{hdr}\n{'-'*len(hdr)}")

    for idx, ep in enumerate(eval_x):
        row = [ep] + [results[c]["eval_reward"].mean(axis=0)[idx] for c in results]
        csv_rows.append(row)
        if PRINT_DETAILED_EVAL_TABLES:
            print(f"{ep:<12}" + "".join(f"{v:14.2f}" for v in row[1:]))

    if SAVE_DETAILED_EVAL_TABLES:
        save_csv(os.path.join(OUTPUT_DIR, f"{env_name}_detailed_eval_points.csv"),
                 ["Episode"] + list(results.keys()), csv_rows)

    # ── Final and best eval summary ────────────────────────────────────────────
    final_rows = []
    if PRINT_SUMMARY_TABLES:
        print(f"\n{env_name} — Final/Best Eval Summary")
        print(f"{'Condition':8s} | {'Final Mean':>10} | {'Final Std':>9} | {'Best Mean':>9} | {'Best Std':>8}")

    for cond, data in results.items():
        arr   = data["eval_reward"]
        final = arr[:, -1]
        best  = arr.max(axis=1)
        fm, fs = final.mean(), final.std()
        bm, bs = best.mean(),  best.std()
        final_rows.append([cond, fm, fs, bm, bs])
        if PRINT_SUMMARY_TABLES:
            print(f"{cond:8s} | {fm:10.2f} | {fs:9.2f} | {bm:9.2f} | {bs:8.2f}")

    if SAVE_SUMMARY_TABLES:
        save_csv(os.path.join(OUTPUT_DIR, f"{env_name}_final_best_eval.csv"),
                 ["Condition","Final Eval Mean","Final Eval Std","Best Eval Mean","Best Eval Std"],
                 final_rows)

     # ── Per-seed final and best eval ───────────────────────────────────────────
    per_seed_rows = []
    for cond, data in results.items():
        arr = data["eval_reward"]
        for s in range(arr.shape[0]):
            per_seed_rows.append([cond, s, arr[s, -1], arr[s].max()])

    save_csv(os.path.join(OUTPUT_DIR, f"{env_name}_per_seed_final_best_eval.csv"),
             ["Condition","Seed","Final Eval","Best Eval"], per_seed_rows)

    # ── Per-seed full curves ───────────────────────────────────────────────────
    per_seed_curve_rows = []
    for cond, data in results.items():
        r_arr  = data["reward"]
        ev_arr = data["eval_reward"]
        for s in range(r_arr.shape[0]):
            for ep_i, v in enumerate(r_arr[s]):
                per_seed_curve_rows.append([cond, s, "train", ep_i + 1, v])
        for s in range(ev_arr.shape[0]):
            for ev_i, v in enumerate(ev_arr[s]):
                per_seed_curve_rows.append([cond, s, "eval", eval_x[ev_i], v])

    save_csv(os.path.join(OUTPUT_DIR, f"{env_name}_per_seed_curves.csv"),
             ["Condition","Seed","CurveType","X","Value"], per_seed_curve_rows)

    # ── Signal curve value CSVs ────────────────────────────────────────────────
    # One CSV per signal with mean ± std at every episode.
    if PRINT_CURVE_VALUES or SAVE_CURVE_VALUES_CSV:
        for key, title in [
            ("reward",      f"{env_name} - Training Reward"),
            ("eval_reward", f"{env_name} - Eval Reward"),
            ("td",          f"{env_name} - TD Error"),
            ("nov",         f"{env_name} - Novelty"),
            ("unc",         f"{env_name} - Uncertainty"),
            ("bio",         f"{env_name} - Bio-RPE"),
            ("eps",         f"{env_name} - Epsilon"),
        ]:
            x_vals = np.arange(1, episodes + 1) if key != "eval_reward" else eval_x
            header = ["X"]
            for c in results:
                header += [f"{c}_mean", f"{c}_std"]

            csv_rows = []
            for i, xv in enumerate(x_vals):
                row = [xv]
                for c, data in results.items():
                    arr  = data[key]
                    row += [float(np.nanmean(arr, axis=0)[i]),
                            float(np.nanstd(arr,  axis=0)[i])]
                csv_rows.append(row)

            if PRINT_CURVE_VALUES:
                print(f"\n{title}\n" + ",".join(header))
                for row in csv_rows:
                    print(",".join(str(v) for v in row))

            if SAVE_CURVE_VALUES_CSV:
                save_csv(os.path.join(OUTPUT_DIR, f"{env_name}_{key}_curve_values.csv"),
                         header, csv_rows)


# =============================================================================
# ANALYSIS PLOTS
# =============================================================================

# ── Plot 1: Novelty vs. visit-count scatter — all novelty-enabled conds ──────
def plot_novelty_vs_visits(env_name: str, results: dict) -> None:
    NOV_CONDS = [c for c in ["TD+NOV", "FULL", "MULT", "GATE"] if c in results]
    if not NOV_CONDS:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    def _one(ax, cond):
        nov_arr    = results[cond]["nov"]
        states_all = results[cond]["visited_states"]
        seed_nov    = nov_arr[0]
        seed_states = np.array(states_all[0], dtype=np.float32)
        if len(seed_states) == 0:
            ax.set_visible(False)
            return

        x_raw  = seed_states[:, 0]
        y_raw  = seed_states[:, 1] if seed_states.shape[1] > 1 else seed_states[:, 0]
        x_bins = np.linspace(x_raw.min(), x_raw.max(), 21)
        y_bins = np.linspace(y_raw.min(), y_raw.max(), 21)
        x_idx  = np.clip(np.digitize(x_raw, x_bins) - 1, 0, 19)
        y_idx  = np.clip(np.digitize(y_raw, y_bins) - 1, 0, 19)

        vg = np.zeros((20, 20), dtype=int)
        for xi, yi in zip(x_idx, y_idx):
            vg[yi, xi] += 1

        step_vc = vg[y_idx, x_idx].astype(float)
        n_eps   = len(seed_nov)
        bounds  = np.linspace(0, len(step_vc), n_eps + 1, dtype=int)
        ep_vc   = np.array([
            step_vc[bounds[i]:bounds[i+1]].mean() if bounds[i+1] > bounds[i] else 0.0
            for i in range(n_eps)
        ])

        sc = ax.scatter(ep_vc, seed_nov, c=np.arange(n_eps),
                        cmap="plasma", alpha=0.55, s=14, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Episode →")

        mask = ~np.isnan(seed_nov)
        if mask.sum() > 5:
            z  = np.polyfit(ep_vc[mask], seed_nov[mask], 1)
            xs = np.linspace(ep_vc.min(), ep_vc.max(), 200)
            ax.plot(xs, np.poly1d(z)(xs), "r--", lw=1.4)
            r, _ = pearsonr(ep_vc[mask], seed_nov[mask])
            ax.set_title(f"{cond}  (r = {r:.3f})", fontsize=11, fontweight="bold")
        else:
            ax.set_title(cond, fontsize=11, fontweight="bold")

        ax.set_xlabel("Mean state-bin visit count", fontsize=9)
        ax.set_ylabel("δ_nov", fontsize=9)
        ax.grid(True, alpha=0.3)

    for i, cond in enumerate(NOV_CONDS):
        _one(axes[i], cond)
    for j in range(len(NOV_CONDS), 4):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{env_name}: Novelty vs. state familiarity — all novelty-enabled conditions (seed 0)\n"
        f"Negative trend confirms habituation: δ_nov falls as states become familiar.",
        fontsize=12
    )
    plt.tight_layout()
    _save_show(f"{env_name}_plot1_novelty_vs_visits.png")


# ── Plot 2: α(t) and β(t) over training ──────────────────────────────────────
def plot_alpha_beta(env_name: str, results: dict, episodes: int) -> None:
    if "alpha" not in next(iter(results.values())):
        print("  [plot_alpha_beta] 'alpha' key missing — add to train_step() return dict.")
        return

    x      = np.arange(1, episodes + 1)
    colors = _cond_colors()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for ax, key, ylabel, greek in zip(
        axes, ["alpha", "beta"], ["α(t)", "β(t)"], ["α", "β"]
    ):
        for cond, data in results.items():
            arr  = data[key]
            mean = np.nanmean(arr, axis=0)
            std  = np.nanstd(arr,  axis=0)
            ax.plot(x, mean, label=cond, color=colors[cond])
            ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=colors[cond])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"{env_name}: {greek}(t) = {greek}_max · tanh(|δ_TD| / 10)", fontsize=10
        )
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Episode", fontsize=11)
    fig.suptitle(
        f"{env_name}: Adaptive weighting coefficients α(t) and β(t)\n"
        f"(mean ± std, {len(SEEDS)} seeds)",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    _save_show(f"{env_name}_plot2_alpha_beta.png")


# ── Plot 3: lr_scale over training ───────────────────────────────────────────
def plot_lr_scale(env_name: str, results: dict, episodes: int) -> None:
    if "lr_scale" not in next(iter(results.values())):
        print("  [plot_lr_scale] 'lr_scale' key missing — add to train_step() return dict.")
        return

    x      = np.arange(1, episodes + 1)
    colors = _cond_colors()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for cond, data in results.items():
        for ax, key in zip(axes, ["lr_scale", "bio"]):
            arr  = data[key]
            mean = np.nanmean(arr, axis=0)
            std  = np.nanstd(arr,  axis=0)
            ax.plot(x, mean, label=cond, color=colors[cond])
            ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=colors[cond])

    axes[0].set_ylabel("lr_scale  [0.5, 1.0]", fontsize=11)
    axes[0].set_ylim(0.45, 1.05)
    axes[0].axhline(0.5, color="gray", lw=0.8, ls="--", label="floor (0.5)")
    axes[0].axhline(1.0, color="gray", lw=0.8, ls=":",  label="ceiling (1.0)")
    axes[0].set_title(
        f"{env_name}: lr_scale = clip(0.5 + 0.5·tanh(|δ_bio|/10), 0.5, 1.0)", fontsize=10
    )
    axes[0].legend(fontsize=8, ncol=4)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("|δ_bio|", fontsize=11)
    axes[1].set_title(f"{env_name}: Bio-RPE magnitude |δ_bio| (for reference)", fontsize=10)
    axes[1].legend(fontsize=8, ncol=3)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Episode", fontsize=11)

    fig.suptitle(
        f"{env_name}: Neuromodulation of learning rate  lr(t) = lr_base × lr_scale\n"
        f"(mean ± std, {len(SEEDS)} seeds). High lr_scale = full plasticity.",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    _save_show(f"{env_name}_plot3_lr_scale.png")


# ── Plot 4: Bio-RPE decomposition stacked area ───────────────────────────────
def plot_bio_rpe_decomposition(env_name: str, results: dict, episodes: int) -> None:
    x      = np.arange(1, episodes + 1)
    ncols  = 3
    nrows  = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, (cond, data) in enumerate(results.items()):
        ax = axes[idx]
        td_m  = np.nanmean(data["td"],  axis=0)
        nov_m = np.nanmean(data["nov"], axis=0)
        unc_m = np.nanmean(data["unc"], axis=0)

        if "alpha" in data:
            a_m = np.nanmean(data["alpha"], axis=0)
            b_m = np.nanmean(data["beta"],  axis=0)
        else:
            a_m = b_m = ALPHA_MAX * np.tanh(td_m / 10.0)

        td_s  = _smooth(np.abs(td_m),          20)
        nov_s = _smooth(np.abs(a_m * nov_m),   20)
        unc_s = _smooth(np.abs(b_m * unc_m),   20)

        ax.stackplot(x, td_s, nov_s, unc_s,
                     labels=["|δ_TD|", "|α·δ_nov|", "|β·δ_unc|"],
                     colors=["#4299E1", "#F6AD55", "#68D391"], alpha=0.82)
        ax.set_title(cond, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("|Component|", fontsize=9)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{env_name}: Bio-RPE decomposition — |δ_TD| vs |α·δ_nov| vs |β·δ_unc|\n"
        f"Stacked area (smoothed, mean across {len(SEEDS)} seeds).",
        fontsize=12
    )
    plt.tight_layout()
    _save_show(f"{env_name}_plot4_bio_rpe_decomposition.png")


# ── Plot 5: Rolling correlation bio-RPE ↔ reward improvement ─────────────────
def plot_bio_rpe_reward_correlation(env_name: str, results: dict,
                                     episodes: int, window: int = 30) -> None:
    x      = np.arange(window, episodes)
    colors = _cond_colors()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color="gray", lw=0.8, ls="--")

    for cond, data in results.items():
        bio_arr = data["bio"]
        rew_arr = data["reward"]
        all_c   = []

        for s in range(bio_arr.shape[0]):
            bio_s  = bio_arr[s]
            imp    = np.diff(rew_arr[s], prepend=rew_arr[s, 0])
            corrs  = []
            for t in range(window, episodes):
                b_w = bio_s[t - window: t]
                r_w = imp[t - window + 1: t + 1]
                ok  = ~(np.isnan(b_w) | np.isnan(r_w))
                if ok.sum() > 5:
                    r, _ = pearsonr(b_w[ok], r_w[ok])
                    corrs.append(r)
                else:
                    corrs.append(np.nan)
            all_c.append(corrs)

        m = np.nanmean(all_c, axis=0)
        s = np.nanstd(all_c,  axis=0)
        ax.plot(x, m, label=cond, color=colors[cond])
        ax.fill_between(x, m - s, m + s, alpha=0.12, color=colors[cond])

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(f"Rolling Pearson r  (window={window})", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(
        f"{env_name}: Rolling correlation  |δ_bio|(t) ↔ Δreward(t+1)\n"
        f"Positive = bio-RPE tracks learning progress. Mean ± std, {len(SEEDS)} seeds.",
        fontsize=11
    )
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_show(f"{env_name}_plot5_bio_reward_correlation.png")


# ── Plot 6: Per-seed novelty spaghetti — all novelty-enabled conditions ───────
def plot_novelty_spaghetti(env_name: str, results: dict, episodes: int) -> None:
    threshold = SPAGHETTI_THRESHOLDS.get(env_name, 0.0)
    NOV_CONDS = [c for c in ["TD+NOV", "FULL", "MULT", "GATE"] if c in results]
    if not NOV_CONDS:
        return

    x = np.arange(1, episodes + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for idx, cond in enumerate(NOV_CONDS):
        ax       = axes[idx]
        nov_arr  = results[cond]["nov"]
        eval_arr = results[cond]["eval_reward"]
        n_solved = n_stuck = 0

        for s in range(nov_arr.shape[0]):
            solved = eval_arr[s, -1] > threshold
            color  = "#2E8B57" if solved else "#C0392B"
            ax.plot(x, nov_arr[s],
                    color=color, alpha=0.75 if solved else 0.50,
                    lw=1.6 if solved else 1.0,
                    ls="-"  if solved else "--")
            if solved: n_solved += 1
            else:      n_stuck  += 1

        proxy = [
            mlines.Line2D([], [], color="#2E8B57", lw=2,
                          label=f"Solved (> {threshold:.0f})  n={n_solved}"),
            mlines.Line2D([], [], color="#C0392B", lw=1.5, ls="--",
                          label=f"Below threshold  n={n_stuck}"),
        ]
        ax.legend(handles=proxy, fontsize=8, loc="upper right")
        ax.set_title(cond, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("δ_nov", fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(NOV_CONDS), 4):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{env_name}: Per-seed novelty trajectories — all novelty-enabled conditions\n"
        f"Green = solved (final eval > {threshold:.0f});  Red = failed/stuck.",
        fontsize=12
    )
    plt.tight_layout()
    _save_show(f"{env_name}_plot6_novelty_spaghetti.png")


# ── Plot 7: Uncertainty heatmap over state space — all 6 conditions ──────────
def plot_uncertainty_state_heatmap(env_name: str, results: dict) -> None:
    if env_name != "MountainCar-v0":
        return

    BINS        = 30
    SHOW_CONDS  = [c for c in ["TD","TD+NOV","TD+UNC","FULL","MULT","GATE"]
                   if c in results]
    if not SHOW_CONDS:
        return

    # Global position/velocity range
    all_pos, all_vel = [], []
    for c in SHOW_CONDS:
        for ss in results[c]["visited_states"]:
            s = np.array(ss, dtype=np.float32)
            if len(s):
                all_pos.extend(s[:, 0].tolist())
                all_vel.extend(s[:, 1].tolist())

    pos_e = np.linspace(min(all_pos) if all_pos else -1.2,
                        max(all_pos) if all_pos else  0.6, BINS + 1)
    vel_e = np.linspace(min(all_vel) if all_vel else -0.07,
                        max(all_vel) if all_vel else  0.07, BINS + 1)

    # Pre-compute grids
    grids, all_unc_vals = {}, []
    for cond in SHOW_CONDS:
        unc_arr    = results[cond]["unc"]
        states_all = results[cond]["visited_states"]
        vg = np.zeros((BINS, BINS))
        ug = np.zeros((BINS, BINS))
        cg = np.zeros((BINS, BINS))

        for si, ss in enumerate(states_all):
            s_arr  = np.array(ss, dtype=np.float32)
            unc_ep = unc_arr[si]
            if len(s_arr) == 0:
                continue
            n_steps = len(s_arr)
            n_eps   = len(unc_ep)
            bounds  = np.linspace(0, n_steps, n_eps + 1, dtype=int)
            step_u  = np.zeros(n_steps)
            for ei in range(n_eps):
                lo, hi = bounds[ei], bounds[ei+1]
                step_u[lo:hi] = 0.0 if np.isnan(unc_ep[ei]) else unc_ep[ei]

            xi = np.clip(np.digitize(s_arr[:, 0], pos_e) - 1, 0, BINS-1)
            yi = np.clip(np.digitize(s_arr[:, 1], vel_e) - 1, 0, BINS-1)
            for t_, (xi_, yi_) in enumerate(zip(xi, yi)):
                vg[yi_, xi_] += 1
                ug[yi_, xi_] += step_u[t_]
                cg[yi_, xi_] += 1

        mu = np.where(cg > 0, ug / cg, np.nan)
        grids[cond] = (vg, mu)
        v = mu[~np.isnan(mu)]
        if len(v):
            all_unc_vals.extend(v.tolist())

    unc_vmin = np.nanpercentile(all_unc_vals,  2) if all_unc_vals else 0
    unc_vmax = np.nanpercentile(all_unc_vals, 98) if all_unc_vals else 1

    n = len(SHOW_CONDS)
    fig, axes = plt.subplots(2, n, figsize=(3.8 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, cond in enumerate(SHOW_CONDS):
        vg, mu = grids[cond]

        # Visit count (top)
        ax_t = axes[0, col]
        im0  = ax_t.pcolormesh(pos_e, vel_e,
                                np.ma.masked_where(vg == 0, vg),
                                cmap="Blues", shading="auto")
        plt.colorbar(im0, ax=ax_t, label="Visits")
        ax_t.set_title(cond, fontsize=10, fontweight="bold")
        ax_t.set_ylabel("Velocity" if col == 0 else "", fontsize=8)
        ax_t.set_xlabel("Position", fontsize=8)
        ax_t.tick_params(labelsize=7)

        # Mean uncertainty (bottom)
        ax_b = axes[1, col]
        im1  = ax_b.pcolormesh(pos_e, vel_e,
                                np.ma.masked_where(np.isnan(mu), mu),
                                cmap="Reds", shading="auto",
                                vmin=unc_vmin, vmax=unc_vmax)
        plt.colorbar(im1, ax=ax_b, label="Mean δ_unc")
        ax_b.set_ylabel("Velocity" if col == 0 else "", fontsize=8)
        ax_b.set_xlabel("Position", fontsize=8)
        ax_b.tick_params(labelsize=7)

    axes[0, 0].annotate("Visit count", xy=(-0.28, 0.5), xycoords="axes fraction",
                         fontsize=9, fontweight="bold", rotation=90, va="center")
    axes[1, 0].annotate("Mean δ_unc",  xy=(-0.28, 0.5), xycoords="axes fraction",
                         fontsize=9, fontweight="bold", rotation=90, va="center")

    fig.suptitle(
        f"{env_name}: State visitation (top) vs mean δ_unc (bottom) — all conditions\n"
        f"Shared colour scale on bottom row. "
        f"Uncertainty is uniformly high in unvisited regions — not selectively near goal (right).",
        fontsize=11
    )
    plt.tight_layout()
    _save_show(f"{env_name}_plot7_uncertainty_state_heatmap.png")


# ── Master Call for Plotting the 7 analysis plots ──────────
def plot_analysis_figures(env_name: str, results: dict, episodes: int) -> None:
    print(f"\n  [Analysis plots] {env_name}...")
    for label, fn, kwargs in [
        ("1 novelty_vs_visits",         plot_novelty_vs_visits,          {"env_name": env_name, "results": results}),
        ("2 alpha_beta",                 plot_alpha_beta,                 {"env_name": env_name, "results": results, "episodes": episodes}),
        ("3 lr_scale",                   plot_lr_scale,                   {"env_name": env_name, "results": results, "episodes": episodes}),
        ("4 bio_rpe_decomposition",      plot_bio_rpe_decomposition,      {"env_name": env_name, "results": results, "episodes": episodes}),
        ("5 bio_reward_correlation",     plot_bio_rpe_reward_correlation, {"env_name": env_name, "results": results, "episodes": episodes}),
        ("6 novelty_spaghetti",          plot_novelty_spaghetti,          {"env_name": env_name, "results": results, "episodes": episodes}),
        ("7 uncertainty_state_heatmap",  plot_uncertainty_state_heatmap,  {"env_name": env_name, "results": results}),
    ]:
        try:
            fn(**kwargs)
        except Exception as e:
            print(f"    [Plot {label}] Error: {e}")

    print(f"  [Analysis plots] {env_name} done.")