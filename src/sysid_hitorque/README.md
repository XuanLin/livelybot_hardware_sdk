# System Identification Pipeline for HiTorque Right Arm

This repository contains the full pipeline for system identification (sysid) of the right arm kinematic chain: trajectory optimization, data collection, parameter identification, URDF generation, and validation.

## Directory Structure

```
sysid_hitorque/
├── optimize_sysid_trajectory/     # Trajectory optimization outputs
├── sysid_data/                    # Raw data collected from hardware
├── bounding_ellipsoids/           # Physical feasibility constraints
├── identified_results/            # Identification outputs
├── traj_following/                # Trajectory following experiment data & results
├── optimize_trajectory.py         # Step 1: trajectory optimization
├── solve_sysid.py                 # Step 3: solve identification
├── generate_identified_urdf.py    # Step 4: generate identified URDF
└── plot_sysid_traj.py             # Step 6: plot trajectory following results
```

All scripts are run from the project root (`sysid_hitorque/`).

---

## Step 1: Optimize Excitation Trajectory

Generate a Fourier-based excitation trajectory that minimizes the condition number of the regressor matrix, subject to joint position/velocity/acceleration limits.

```bash
python3 optimize_trajectory.py --urdf /path/to/robot.urdf
```

**Outputs** (saved to `optimize_sysid_trajectory/`):
- `optimal_fourier_coeffs.npz` — optimized Fourier coefficients

---

## Step 2: Run Excitation Trajectory on Hardware

Deploy the trajectory from `optimize_sysid_trajectory/sysid_trajectory.csv` on the real robot using the hardware controller (e.g. `arm_sysid_trajectory.sh`). The controller logs joint positions, velocities, and torques during execution.

Save the recorded CSV to:

```
sysid_data/sysid_data.csv
```

---

## Step 3: Solve System Identification

Using the recorded data and bounding ellipsoid constraints for physical feasibility, solve for the inertial and friction parameters.

```bash
python3 solve_sysid.py \
    --urdf /path/to/robot.urdf \
    --csv sysid_data/sysid_data.csv \
    --ellipsoids bounding_ellipsoids/ellipsoids.npz
```

**Outputs** (saved to `identified_results/`):
- `identified_params.npz` — identified inertial parameters and friction parameters

---

## Step 4: Generate Identified URDF

Convert the identified parameters back into URDF format (COM-frame inertias, mass, friction) and produce a new URDF.

```bash
python3 generate_identified_urdf.py \
    --urdf_in /path/to/robot.urdf \
    --params identified_results/identified_params.npz
```

**Output** (saved to `generated_urdf/`):
- `hi_identified.urdf` — URDF with updated inertial and friction parameters for the right arm links

---

## Step 5: Run Trajectory Following on Hardware

Use the identified URDF to compute feedforward torques on the robot and run a trajectory following experiment (e.g. `arm_traj_following.sh`). You can also run the same trajectory with the nominal (unidentified) URDF as a baseline for comparison.

Save the recorded CSVs to `traj_following/`, e.g.:

```
traj_following/arm_traj_following_id.csv             # identified URDF
traj_following/arm_traj_following_without_id.csv     # nominal URDF
```

---

## Step 6: Plot and Compare Results

Plot tracking performance comparing two runs (e.g. identified vs. without id):

```bash
python3 plot_sysid_traj.py \
    --csv traj_following/arm_traj_following_without_id.csv \
    --csv2 traj_following/arm_traj_following_id.csv \
    --label1 "nominal" \
    --label2 "identified"
```

**Outputs** (saved to `traj_following/results/`):
- `*_tracking.png` — desired vs. actual joint positions
- `*_tracking_error.png` — per-joint tracking error
- `*_torque.png` — feedforward and measured torques
- `*_summary.txt` — RMS and max error table
- `*_comparison.png` — side-by-side error comparison (when using `--csv2`)
