#!/usr/bin/env python3
"""
Plot sinusoidal trajectory tracking results from arm_sysid_traj.

Usage:
  python3 plot_sysid_traj.py --csv arm_sysid_traj.csv
  python3 plot_sysid_traj.py --csv run_old.csv --csv2 run_new.csv   # compare two URDFs
  e.g. python3 plot_sysid_traj.py --csv traj_following/arm_traj_following_id_without_FT.csv --csv2 traj_following/arm_traj_following_id_with_FT.csv --label1 "id-without-FT" --label2 "id-with-FT"

Outputs (saved next to the CSV):
  *_tracking.png        — desired vs actual joint positions
  *_tracking_error.png  — position tracking error per joint
  *_torque.png          — feedforward torques & measured torques
  *_summary.txt         — RMS / max error table
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JOINT_NAMES_SHORT = [
    "shoulder_pitch",
    "shoulder_roll",
    "arm_yaw",
    "arm_roll",
    "wrist_yaw",
]

NUM_JOINTS = 5
RAD2DEG = 180.0 / np.pi

# Must match the C++ RAMP_IN_TIME — used as a safety trim
RAMP_IN_TIME = 2.0


def load_csv(path):
    df = pd.read_csv(path)
    # Safety trim: drop any rows that leaked from the ramp-in period
    if "time" in df.columns:
        df = df[df["time"] >= RAMP_IN_TIME].reset_index(drop=True)
    return df


def get_joint_columns(df):
    """Auto-detect column names from the CSV header."""
    cols = list(df.columns)
    joints = []
    # Find unique joint prefixes (everything before _q_des)
    for c in cols:
        if c.endswith("_q_des"):
            prefix = c[: -len("_q_des")]
            joints.append(prefix)
    return joints


def plot_tracking(df, joints, out_prefix, label=""):
    """Plot desired vs actual position for each joint."""
    t = df["time"].values

    fig, axes = plt.subplots(NUM_JOINTS, 1, figsize=(14, 3.0 * NUM_JOINTS), sharex=True)
    if NUM_JOINTS == 1:
        axes = [axes]

    for i, jname in enumerate(joints):
        ax = axes[i]
        q_des = df[f"{jname}_q_des"].values * RAD2DEG
        q_act = df[f"{jname}_q_act"].values * RAD2DEG

        ax.plot(t, q_des, "b-", linewidth=1.2, label="desired")
        ax.plot(t, q_act, "r-", linewidth=1.0, alpha=0.8, label="actual")
        ax.set_ylabel(f"{JOINT_NAMES_SHORT[i]}\n(deg)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-180, 180)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Trajectory Tracking — Desired vs Actual{' (' + label + ')' if label else ''}", fontsize=13)
    fig.tight_layout()
    path = f"{out_prefix}_tracking.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_error(df, joints, out_prefix, label=""):
    """Plot tracking error for each joint."""
    t = df["time"].values

    fig, axes = plt.subplots(NUM_JOINTS, 1, figsize=(14, 3.0 * NUM_JOINTS), sharex=True)
    if NUM_JOINTS == 1:
        axes = [axes]

    for i, jname in enumerate(joints):
        ax = axes[i]
        err = (df[f"{jname}_q_des"].values - df[f"{jname}_q_act"].values) * RAD2DEG
        ax.plot(t, err, "k-", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel(f"{JOINT_NAMES_SHORT[i]}\nerr (deg)", fontsize=9)
        rms = np.sqrt(np.mean(err**2))
        mx  = np.max(np.abs(err))
        ax.set_title(f"RMS={rms:.2f}°  Max={mx:.2f}°", fontsize=9, loc="right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Tracking Error{' (' + label + ')' if label else ''}", fontsize=13)
    fig.tight_layout()
    path = f"{out_prefix}_tracking_error.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_torques(df, joints, out_prefix, label=""):
    """Plot feedforward and measured torques."""
    t = df["time"].values

    fig, axes = plt.subplots(NUM_JOINTS, 1, figsize=(14, 3.0 * NUM_JOINTS), sharex=True)
    if NUM_JOINTS == 1:
        axes = [axes]

    for i, jname in enumerate(joints):
        ax = axes[i]
        tau_ff   = df[f"{jname}_tau_ff"].values
        tau_meas = df[f"{jname}_tau_meas"].values

        ax.plot(t, tau_ff,   "b-", linewidth=1.0, label="tau_ff (ID)")
        ax.plot(t, tau_meas, "r-", linewidth=0.8, alpha=0.7, label="tau_meas")
        ax.set_ylabel(f"{JOINT_NAMES_SHORT[i]}\n(Nm)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-4, 4)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Torques{' (' + label + ')' if label else ''}", fontsize=13)
    fig.tight_layout()
    path = f"{out_prefix}_torque.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_comparison(df1, df2, joints1, joints2, out_prefix, label1="old URDF", label2="new URDF"):
    """Compare tracking error between two runs (two URDFs)."""
    t1 = df1["time"].values
    t2 = df2["time"].values

    fig, axes = plt.subplots(NUM_JOINTS, 1, figsize=(14, 3.0 * NUM_JOINTS), sharex=True)
    if NUM_JOINTS == 1:
        axes = [axes]

    for i in range(NUM_JOINTS):
        ax = axes[i]
        err1 = (df1[f"{joints1[i]}_q_des"].values - df1[f"{joints1[i]}_q_act"].values) * RAD2DEG
        err2 = (df2[f"{joints2[i]}_q_des"].values - df2[f"{joints2[i]}_q_act"].values) * RAD2DEG

        ax.plot(t1, err1, "r-", linewidth=0.8, alpha=0.7, label=f"{label1} (RMS={np.sqrt(np.mean(err1**2)):.2f}°)")
        ax.plot(t2, err2, "b-", linewidth=0.8, alpha=0.7, label=f"{label2} (RMS={np.sqrt(np.mean(err2**2)):.2f}°)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel(f"{JOINT_NAMES_SHORT[i]}\nerr (deg)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Tracking Error Comparison: {label1} vs {label2}", fontsize=13)
    fig.tight_layout()
    path = f"{out_prefix}_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def write_summary(df, joints, out_prefix, label=""):
    """Write RMS and max error summary to a text file."""
    path = f"{out_prefix}_summary.txt"
    lines = []
    lines.append(f"Trajectory Tracking Summary{' (' + label + ')' if label else ''}")
    lines.append("=" * 60)
    lines.append(f"{'Joint':<20} {'RMS err (deg)':>14} {'Max err (deg)':>14} {'RMS tau_ff (Nm)':>16}")
    lines.append("-" * 60)

    total_rms = 0.0
    for i, jname in enumerate(joints):
        err = (df[f"{jname}_q_des"].values - df[f"{jname}_q_act"].values) * RAD2DEG
        tau = df[f"{jname}_tau_ff"].values
        rms_err = np.sqrt(np.mean(err**2))
        max_err = np.max(np.abs(err))
        rms_tau = np.sqrt(np.mean(tau**2))
        total_rms += rms_err**2
        lines.append(f"{JOINT_NAMES_SHORT[i]:<20} {rms_err:>14.3f} {max_err:>14.3f} {rms_tau:>16.3f}")

    avg_rms = np.sqrt(total_rms / NUM_JOINTS)
    lines.append("-" * 60)
    lines.append(f"{'Avg RMS':<20} {avg_rms:>14.3f}")
    lines.append("")

    txt = "\n".join(lines)
    with open(path, "w") as f:
        f.write(txt)
    print(f"Saved: {path}")
    print(txt)


def main():
    parser = argparse.ArgumentParser(description="Plot sysid trajectory tracking results")
    parser.add_argument("--csv",  required=True, help="Path to CSV from arm_sysid_traj")
    parser.add_argument("--csv2", default=None,  help="Optional second CSV for comparison (new URDF)")
    parser.add_argument("--label1", default="old URDF", help="Label for first CSV")
    parser.add_argument("--label2", default="new URDF", help="Label for second CSV")
    args = parser.parse_args()

    # ---- Load and plot first CSV ----
    df1 = load_csv(args.csv)
    joints1 = get_joint_columns(df1)
    assert len(joints1) == NUM_JOINTS, f"Expected {NUM_JOINTS} joints, got {len(joints1)}: {joints1}"

    results_dir = os.path.join(os.path.dirname(args.csv), "results")
    os.makedirs(results_dir, exist_ok=True)

    out1 = os.path.join(results_dir, os.path.splitext(os.path.basename(args.csv))[0])
    plot_tracking(df1, joints1, out1, label=args.label1 if args.csv2 else "")
    plot_error(df1, joints1, out1, label=args.label1 if args.csv2 else "")
    plot_torques(df1, joints1, out1, label=args.label1 if args.csv2 else "")
    write_summary(df1, joints1, out1, label=args.label1 if args.csv2 else "")

    # ---- Optionally load and compare second CSV ----
    if args.csv2:
        df2 = load_csv(args.csv2)
        joints2 = get_joint_columns(df2)
        assert len(joints2) == NUM_JOINTS

        out2 = os.path.join(results_dir, os.path.splitext(os.path.basename(args.csv2))[0])
        plot_tracking(df2, joints2, out2, label=args.label2)
        plot_error(df2, joints2, out2, label=args.label2)
        plot_torques(df2, joints2, out2, label=args.label2)
        write_summary(df2, joints2, out2, label=args.label2)

        # Comparison plot
        plot_comparison(df1, df2, joints1, joints2, out1, args.label1, args.label2)


if __name__ == "__main__":
    main()
