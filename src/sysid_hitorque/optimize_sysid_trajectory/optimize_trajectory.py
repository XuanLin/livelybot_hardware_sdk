#!/usr/bin/env python3
"""
Optimal excitation trajectory for right arm system identification.
Solves Problem (3.10): minimize cond(Γ) over Fourier coefficients a, b
subject to joint position/velocity/acceleration limits and boundary conditions.

Usage:
    python3 optimize_trajectory.py --urdf /path/to/robot.urdf
"""

import argparse
import numpy as np
import pinocchio as pin
import cyipopt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---- Right arm joints (same as gravity comp code) ----
ARM_JOINTS = [
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint",
    "r_arm_yaw_joint", "r_arm_roll_joint", "r_wrist_yaw_joint",
]

ARM_FRAME_NAMES = [
    "base_link", "r_shoulder_pitch_link", "r_shoulder_roll_link",
    "r_arm_yaw_link", "r_arm_roll_link", "r_wrist_yaw_link", "r_hand_box_link",
]

NJ = 5  # number of joints

# ---- Trajectory parameters (from thesis Section 3.3.1) ----
M  = 5          # Fourier harmonics per joint (thesis uses 10, start smaller)
WF = 0.1 * np.pi  # fundamental frequency
TF = 2 * np.pi / WF  # period = 20s
N  = 100        # time samples
DT = TF / N     # 0.2s per sample
T_VEC = np.arange(N) * DT

# ---- Joint limits (conservative defaults, change to match your robot) ----
# Q_MAX   = np.array([3.14, 3.14, 3.14, 3.14, 3.14])  # rad
Q_MAX   = np.array([3.14/2, 3.14/2, 3.14/2, 3.14/6, 3.14/2])  # rad
DQ_MAX  = np.array([10.0/10, 10.0/10, 10.0/10, 10.0/10, 10.0/10])  # rad/s
DDQ_MAX = np.array([20.0/10, 20.0/10, 20.0/10, 20.0/10, 20.0/10])  # rad/s^2

# Centre of oscillation per joint (Fourier trajectory oscillates around this)
Q_CENTER = np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.0])  # rad

# ===========================================================================
# Fourier trajectory: x = [a_{1,1}..a_{1,M}, b_{1,1}..b_{1,M}, ..., a_{5,M}, b_{5,M}]
# Total decision variables: NJ * 2 * M
# ===========================================================================

def unpack(x):
    """Unpack flat x into a(NJ, M) and b(NJ, M)."""
    x = x.reshape(NJ, 2 * M)
    return x[:, :M], x[:, M:]


def fourier_traj(x):
    """
    Compute q, dq, ddq from Fourier coefficients.
    Returns arrays of shape (N, NJ).
    Eq. (3.10a-c) from thesis.
    """
    a, b = unpack(x)
    k = np.arange(1, M + 1)  # harmonic indices
    wk = WF * k              # (M,)

    q   = np.zeros((N, NJ))
    dq  = np.zeros((N, NJ))
    ddq = np.zeros((N, NJ))

    for i in range(NJ):
        for j, t in enumerate(T_VEC):
            s = np.sin(wk * t)
            c = np.cos(wk * t)
            q[j, i]   = np.sum(a[i] / wk * s - b[i] / wk * c)
            dq[j, i]  = np.sum(a[i] * c + b[i] * s)
            ddq[j, i] = np.sum(-a[i] * wk * s + b[i] * wk * c)

    # q_offset to ensure q(0) = 0: q(0) = sum(-b/(wk)) + q0 = 0
    # so q0 = sum(b/wk) per joint, already embedded in the formula
    # since at t=0: sin=0, cos=1, q = sum(-b/wk) + q0
    # We define q0_i = sum(b_i / wk) so q(0) = 0
    q0 = np.sum(b / wk[None, :], axis=1)  # (NJ,)
    q += q0[None, :]

    # Shift oscillation centre (e.g. shoulder roll starts at -90°)
    q += Q_CENTER[None, :]

    return q, dq, ddq


# ===========================================================================
# Regressor builder (reuses logic from build_regressor_arm.py)
# ===========================================================================

def setup_model(urdf_path):
    """Load model, find arm indices."""
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    arm_jids = [model.getJointId(n) for n in ARM_JOINTS]
    arm_v = [model.joints[j].idx_v for j in arm_jids]
    arm_q = [model.joints[j].idx_q for j in arm_jids]
    arm_cols = [c for j in arm_jids for c in range((j - 1) * 10, j * 10)]
    return model, data, arm_q, arm_v, arm_cols


def build_regressor(model, data, arm_q_idx, arm_v_idx, arm_cols, q_traj, dq_traj, ddq_traj):
    """Build arm-only regressor Γ from trajectory. Returns (N*5, 50) matrix."""
    q_full = pin.neutral(model)
    Gamma = np.zeros((N * NJ, NJ * 10))
    for k in range(N):
        for i in range(NJ):
            q_full[arm_q_idx[i]] = q_traj[k, i]
        dq_full = np.zeros(model.nv)
        ddq_full = np.zeros(model.nv)
        for i in range(NJ):
            dq_full[arm_v_idx[i]] = dq_traj[k, i]
            ddq_full[arm_v_idx[i]] = ddq_traj[k, i]
        Y = pin.computeJointTorqueRegressor(model, data, q_full, dq_full, ddq_full)
        Gamma[k * NJ:(k + 1) * NJ, :] = Y[np.ix_(arm_v_idx, arm_cols)]
    return Gamma


def cond_number(Gamma):
    """Condition number of Γ, using only nonzero singular values."""
    S = np.linalg.svd(Gamma, compute_uv=False)
    tol = max(Gamma.shape) * S[0] * np.finfo(float).eps
    S_nz = S[S > tol]
    if len(S_nz) < 2:
        return 1e12
    return S_nz[0] / S_nz[-1]


# ===========================================================================
# Upper bounds on |q|, |dq|, |ddq| from Fourier coefficients (Eq. 3.10g-i)
# These are analytical worst-case bounds, no need to check every timestep.
# ===========================================================================

def fourier_bounds(x):
    """
    Returns (q_bound, dq_bound, ddq_bound), each shape (NJ,).
    These are the maximum possible |q|, |dq|, |ddq| over all time.
    """
    a, b = unpack(x)
    k = np.arange(1, M + 1)
    wk = WF * k
    amp = np.sqrt(a**2 + b**2)  # (NJ, M)

    q0 = np.sum(np.abs(b) / wk[None, :], axis=1)  # offset magnitude bound
    q_bound   = np.sum(amp / wk[None, :], axis=1) + q0
    dq_bound  = np.sum(amp, axis=1)
    ddq_bound = np.sum(amp * wk[None, :], axis=1)
    return q_bound, dq_bound, ddq_bound

# ===========================================================================
# IPOPT problem
# ===========================================================================

class ExcitationProblem:
    def __init__(self, model, data, arm_q_idx, arm_v_idx, arm_cols):
        self.model = model
        self.data = data
        self.arm_q_idx = arm_q_idx
        self.arm_v_idx = arm_v_idx
        self.arm_cols = arm_cols
        self.n_vars = NJ * 2 * M
        self.eval_count = 0

    def objective(self, x):
        self.eval_count += 1
        q, dq, ddq = fourier_traj(x)
        Gamma = build_regressor(
            self.model, self.data,
            self.arm_q_idx, self.arm_v_idx, self.arm_cols,
            q, dq, ddq,
        )
        c = cond_number(Gamma)
        if self.eval_count % 20 == 0:
            print(f"  eval {self.eval_count}: cond = {c:.1f}")
        return c

    def gradient(self, x):
        """Finite-difference gradient."""
        eps = 1e-6
        f0 = self.objective(x)
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x[i] += eps
            grad[i] = (self.objective(x) - f0) / eps
            x[i] -= eps
        return grad

    def constraints(self, x):
        """
        Inequality constraints (must be >= 0 for IPOPT with cl=0, cu=inf):
          Q_MAX  - q_bound  >= 0   (5 constraints)
          DQ_MAX - dq_bound >= 0   (5 constraints)
          DDQ_MAX- ddq_bound>= 0   (5 constraints)

        Equality constraints (cl=cu=0):
          sum_k a_{i,k} = 0        (5 constraints, dq(0)=0)
          sum_k k*wf*b_{i,k} = 0   (5 constraints, ddq(0)=0)
        """
        a, b = unpack(x)
        k = np.arange(1, M + 1)
        wk = WF * k

        q_b, dq_b, ddq_b = fourier_bounds(x)

        ineq = np.concatenate([
            Q_MAX - q_b,
            DQ_MAX - dq_b,
            DDQ_MAX - ddq_b,
        ])

        # Boundary: dq(0)=0 and ddq(0)=0
        eq_dq  = np.sum(a, axis=1)           # sum_k a_{i,k} = 0
        eq_ddq = np.sum(b * wk[None, :], axis=1)  # sum_k k*wf*b_{i,k} = 0

        return np.concatenate([ineq, eq_dq, eq_ddq])

    def jacobian(self, x):
        """Finite-difference Jacobian of constraints."""
        eps = 1e-6
        c0 = self.constraints(x)
        jac = np.zeros((len(c0), len(x)))
        for i in range(len(x)):
            x[i] += eps
            jac[:, i] = (self.constraints(x) - c0) / eps
            x[i] -= eps
        return jac.flatten()
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        # STOP CONDITION:
        print(f"  [Callback] Iter {iter_count}: Cond = {obj_value:.4f}, Infeas = {inf_pr:.2e}")
        # 1. Objective (cond number) is good enough (< 6.0)
        # 2. Feasibility (inf_pr) is acceptable (we aren't breaking physics)
        # if obj_value < 7.7 and inf_pr < 1e-2:
        if obj_value < 80.0 and inf_pr < 1e-4:
            print(f"\n[EARLY STOPPING] Reached target cond={obj_value:.2f} with feas={inf_pr:.2e}")
            return False  # Returning False tells IPOPT to terminate immediately
        
        return True  # Returning True tells IPOPT to continue


def solve(urdf_path, rate_hz=200.0):
    model, data, arm_q, arm_v, arm_cols = setup_model(urdf_path)
    prob = ExcitationProblem(model, data, arm_q, arm_v, arm_cols)

    n_vars = prob.n_vars
    n_ineq = 3 * NJ  # q, dq, ddq bounds
    n_eq   = 2 * NJ  # dq(0)=0, ddq(0)=0
    n_con  = n_ineq + n_eq

    # Constraint bounds: inequalities >= 0, equalities = 0
    cl = np.concatenate([np.zeros(n_ineq), -1e-4 * np.ones(n_eq)])
    cu = np.concatenate([np.full(n_ineq, 1e20),  1e-4 * np.ones(n_eq)])

    # Variable bounds (loose box on Fourier coefficients)
    lb = -10.0 * np.ones(n_vars)
    ub =  10.0 * np.ones(n_vars)

    # Initial guess: small random
    np.random.seed(42)
    x0 = 0.5 * np.random.randn(n_vars)

    # Check initial condition number
    q, dq, ddq = fourier_traj(x0)
    Gamma = build_regressor(model, data, arm_q, arm_v, arm_cols, q, dq, ddq)
    print(f"Initial cond(Γ): {cond_number(Gamma):.1f}")

    # Dense Jacobian structure
    jac_structure = (
        np.repeat(np.arange(n_con), n_vars),
        np.tile(np.arange(n_vars), n_con),
    )

    nlp = cyipopt.Problem(
        n=n_vars,
        m=n_con,
        problem_obj=prob,
        lb=lb, ub=ub,
        cl=cl, cu=cu,
    )

    nlp.add_option("print_level", 5)
    nlp.add_option("jacobian_approximation", "finite-difference-values")

    print("\nSolving with IPOPT...")
    x_opt, info = nlp.solve(x0)

    # Evaluate result
    q_opt, dq_opt, ddq_opt = fourier_traj(x_opt)
    Gamma_opt = build_regressor(model, data, arm_q, arm_v, arm_cols,
                                q_opt, dq_opt, ddq_opt)
    S = np.linalg.svd(Gamma_opt, compute_uv=False)
    tol = max(Gamma_opt.shape) * S[0] * np.finfo(float).eps
    rank = int(np.sum(S > tol))
    cond = cond_number(Gamma_opt)

    print(f"\n{'=' * 50}")
    print(f"  Optimized cond(Γ): {cond:.1f}")
    print(f"  Rank: {rank}/{Gamma_opt.shape[1]}")
    print(f"  σ_max={S[0]:.4e}, σ_min={S[S > tol][-1]:.4e}")
    print(f"  q  bounds used: {np.max(np.abs(q_opt), axis=0)}")
    print(f"  dq bounds used: {np.max(np.abs(dq_opt), axis=0)}")
    print(f"{'=' * 50}")

    # Save optimal coefficients
    a_opt, b_opt = unpack(x_opt)
    np.savez("optimal_fourier_coeffs.npz", a=a_opt, b=b_opt, wf=WF, M=M)
    print("Saved optimal_fourier_coeffs.npz")

    # Plot
    plot_trajectory(q_opt, dq_opt, ddq_opt)
    save_trajectory_csv(x_opt, rate_hz=200.0)
    save_animation_gif(model, data, arm_q, x_opt)

    return x_opt, q_opt, dq_opt, ddq_opt

# ===========================================================================
# Plot, save trajectory and generate animation
# ===========================================================================

def plot_trajectory(q, dq, ddq):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = [n.replace("r_", "").replace("_joint", "") for n in ARM_JOINTS]

    for i in range(NJ):
        axes[0].plot(T_VEC, q[:, i], label=labels[i])
        axes[1].plot(T_VEC, dq[:, i])
        axes[2].plot(T_VEC, ddq[:, i])

    axes[0].set_ylabel("q (rad)")
    axes[1].set_ylabel("dq (rad/s)")
    axes[2].set_ylabel("ddq (rad/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Optimal Excitation Trajectory")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("optimal_trajectory.png", dpi=150)
    print("Saved optimal_trajectory.png")

def save_trajectory_csv(x, rate_hz=200.0, out_path="sysid_trajectory.csv"):
    """Generate high-rate CSV from Fourier coefficients for hardware playback."""
    a, b = unpack(x)
    k = np.arange(1, M + 1)
    wk = WF * k
    dt = 1.0 / rate_hz
    t_vec = np.arange(0, TF + dt, dt)
    t_vec = t_vec[t_vec <= TF]
    N_csv = len(t_vec)

    q = np.zeros((N_csv, NJ))
    dq = np.zeros((N_csv, NJ))
    ddq = np.zeros((N_csv, NJ))

    for i in range(NJ):
        for j, t in enumerate(t_vec):
            s = np.sin(wk * t)
            c = np.cos(wk * t)
            q[j, i]   = np.sum(a[i] / wk * s - b[i] / wk * c)
            dq[j, i]  = np.sum(a[i] * c + b[i] * s)
            ddq[j, i] = np.sum(-a[i] * wk * s + b[i] * wk * c)

    q0 = np.sum(b / wk[None, :], axis=1)
    q += q0[None, :]
    q += Q_CENTER[None, :]

    header = ["t"] + [f"q{i}" for i in range(NJ)] + [f"dq{i}" for i in range(NJ)] + [f"ddq{i}" for i in range(NJ)]
    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for j in range(N_csv):
            row = [f"{t_vec[j]:.6f}"]
            row += [f"{q[j,i]:.8f}" for i in range(NJ)]
            row += [f"{dq[j,i]:.8f}" for i in range(NJ)]
            row += [f"{ddq[j,i]:.8f}" for i in range(NJ)]
            f.write(",".join(row) + "\n")
    print(f"Saved {out_path} ({N_csv} samples at {rate_hz} Hz)")

def save_animation_gif(model, data, arm_q_idx, x, out_path="trajectory_preview.gif"):
    """Compute FK and save a 3D stick-figure GIF."""
    N_ANIM = 300
    a, b = unpack(x)
    k = np.arange(1, M + 1)
    wk = WF * k
    t_vec = np.linspace(0, TF, N_ANIM)
    dt = t_vec[1] - t_vec[0]

    # Reconstruct q
    q_traj = np.zeros((N_ANIM, NJ))
    for i in range(NJ):
        for j, t in enumerate(t_vec):
            s = np.sin(wk * t)
            c = np.cos(wk * t)
            q_traj[j, i] = np.sum(a[i] / wk * s - b[i] / wk * c)
    q0 = np.sum(b / wk[None, :], axis=1)
    q_traj += q0[None, :]
    q_traj += Q_CENTER[None, :]

    # Get frame IDs
    frame_ids = [model.getFrameId(n) for n in ARM_FRAME_NAMES if model.existFrame(n)]

    # FK for all timesteps
    q_full = pin.neutral(model)
    all_pts = []
    for i in range(N_ANIM):
        for j, idx in enumerate(arm_q_idx):
            q_full[idx] = q_traj[i, j]
        pin.forwardKinematics(model, data, q_full)
        pin.updateFramePlacements(model, data)
        all_pts.append([data.oMf[fid].translation.copy() for fid in frame_ids])
    all_pts = np.array(all_pts)

    # Animate
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Optimal Trajectory - 3D Skeleton")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    line, = ax.plot([], [], [], "o-", color="steelblue", linewidth=3, markersize=8)

    mins, maxs = all_pts.min(axis=(0,1)), all_pts.max(axis=(0,1))
    center = (maxs + mins) / 2.0
    r = max((maxs - mins).max() / 2.0, 0.2)
    for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(c - r, c + r)

    def update(idx):
        pts = all_pts[idx]
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=N_ANIM, interval=dt*1000, blit=False)
    ani.save(out_path, writer="pillow", fps=int(1.0/dt))
    print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--rate", type=float, default=200.0)
    args = parser.parse_args()
    solve(args.urdf, rate_hz=args.rate)
