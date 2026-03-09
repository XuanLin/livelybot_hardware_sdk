#!/usr/bin/env python3
"""
System Identification with Convex Geometric (Entropic) Regularization.

Solves Problem (3.15) from the thesis:
    min_Ψ  ||Γ(q,dq,ddq)Ψ - τ||² + λ * d_F(Ψ, Ψ₀)²

    s.t.  tr(P(ψ_i) Q_i) >= 0,   i = 1,...,5

where d_F is the entropic measurement (Eq. 3.14), an approximation
of the Riemannian distance on the space of inertia parameters.

The prior Ψ₀ is extracted directly from the URDF file.

Usage:
    python3 solve_sysid.py \
        --urdf /path/to/hi.urdf \
        --csv /path/to/sysid_data.csv \
        --lambda_reg 0.1 \
        --filter_order 4 \
        --filter_window 101

    e.g. python3 solve_sysid.py --urdf urdf/hi.urdf --csv sysid_data/sysid_data_with_FT.csv --lambda_reg 100 --ellipsoids bounding_ellipsoids/ellipsoids.npz --com_scale 0.8
    
Dependencies:
    pip install pinocchio numpy scipy matplotlib
"""

import argparse
import numpy as np
import pinocchio as pin
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---- Right arm joints ----
ARM_JOINTS = [
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint",
    "r_arm_yaw_joint", "r_arm_roll_joint", "r_wrist_yaw_joint",
]
NJ = 5

# ---- Reflected inertia (armature) from MuJoCo XML ----
# These are I_rotor * N^2 values, keyed by joint name.
# From hi_25dof.xml:  arm_motor (5047-36) = 0.044277,
#                      wrist_motor (4438-30) = 0.008419
ARMATURE = {
    "r_shoulder_pitch_joint": 0.044277,
    "r_shoulder_roll_joint":  0.044277,
    "r_arm_yaw_joint":        0.044277,
    "r_arm_roll_joint":       0.044277,
    "r_wrist_yaw_joint":      0.008419,
}

# =========================================================================
# 1. Load and preprocess hardware data
# =========================================================================

def load_sysid_data(csv_path, t_start_skip=1.0, t_end_skip=5.0,
                    sg_order=4, sg_window=101):
    """
    Load the CSV recorded by arm_sysid_trajectory, keep only Phase 2 (EXEC),
    and apply Savitzky-Golay filtering.

    Returns: t, q, dq, ddq, tau_meas  (each shape (N_used, NJ))
    """
    import csv as csvmod

    with open(csv_path, "r") as f:
        reader = csvmod.DictReader(f)
        rows = list(reader)

    # Extract Phase 2 rows only
    phase2 = [r for r in rows if int(r["phase"]) == 2]
    if len(phase2) == 0:
        raise ValueError("No Phase 2 (EXEC) data found in CSV!")

    N = len(phase2)
    t       = np.zeros(N)
    q_meas  = np.zeros((N, NJ))
    dq_meas = np.zeros((N, NJ))
    tau_meas = np.zeros((N, NJ))
    # We also have desired ddq for reference
    ddq_des  = np.zeros((N, NJ))

    for k, row in enumerate(phase2):
        t[k] = float(row["time"])
        for i in range(NJ):
            q_meas[k, i]   = float(row[f"q_meas_{i}"])
            dq_meas[k, i]  = float(row[f"dq_meas_{i}"])
            tau_meas[k, i] = float(row[f"tau_meas_{i}"])
            ddq_des[k, i]  = float(row[f"ddq_des_{i}"])

    print(f"Loaded {N} Phase-2 samples from {csv_path}")
    print(f"  Time range: [{t[0]:.2f}, {t[-1]:.2f}] s")

    # Skip first t_start_skip seconds (static friction) and last t_end_skip
    t_rel = t - t[0]
    mask = (t_rel >= t_start_skip) & (t_rel <= t_rel[-1] - t_end_skip)
    t       = t[mask]
    q_meas  = q_meas[mask]
    dq_meas = dq_meas[mask]
    tau_meas = tau_meas[mask]
    ddq_des  = ddq_des[mask]

    N_used = q_meas.shape[0]
    print(f"  After trimming: {N_used} samples")

    # Apply Savitzky-Golay filter (thesis uses 4th order, window 501 at 800Hz)
    # Adjust window for your sample rate
    if sg_window % 2 == 0:
        sg_window += 1  # must be odd

    for i in range(NJ):
        q_meas[:, i]   = savgol_filter(q_meas[:, i],   sg_window, sg_order)
        dq_meas[:, i]  = savgol_filter(dq_meas[:, i],  sg_window, sg_order)
        tau_meas[:, i] = savgol_filter(tau_meas[:, i],  sg_window, sg_order)

    # Compute acceleration from filtered velocity via numerical differentiation
    dt = np.median(np.diff(t))
    ddq_meas = np.zeros_like(dq_meas)
    for i in range(NJ):
        ddq_meas[:, i] = savgol_filter(dq_meas[:, i], sg_window, sg_order, deriv=1, delta=dt)

    return t, q_meas, dq_meas, ddq_meas, tau_meas


# =========================================================================
# 2. Build observation matrix (regressor) Γ
# =========================================================================

def setup_pinocchio(urdf_path):
    """Load model, find arm joint/column indices."""
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    arm_jids = [model.getJointId(n) for n in ARM_JOINTS]
    arm_v    = [model.joints[j].idx_v for j in arm_jids]
    arm_q    = [model.joints[j].idx_q for j in arm_jids]
    arm_cols = [c for j in arm_jids for c in range((j - 1) * 10, j * 10)]

    return model, data, arm_q, arm_v, arm_cols


def build_regressor(model, data, arm_q_idx, arm_v_idx, arm_cols,
                    q_traj, dq_traj, ddq_traj):
    """
    Build the observation matrix Γ from trajectory data.
    Returns Γ of shape (N*NJ, NJ*10).
    """
    N = q_traj.shape[0]
    q_full = pin.neutral(model)
    Gamma = np.zeros((N * NJ, NJ * 10))
    Gamma_f = np.zeros((N * NJ, 2 * NJ))

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

        # --- friction regressor for this timestep ---
        dq_arm = dq_traj[k, :]  # (NJ,)
        Gamma_f[k * NJ:(k + 1) * NJ, :NJ]  = np.diag(dq_arm)          # viscous
        Gamma_f[k * NJ:(k + 1) * NJ, NJ:]  = np.diag(np.sign(dq_arm)) # Coulomb

    return Gamma, Gamma_f


# =========================================================================
# 3. Extract nominal inertial parameters Ψ₀ from URDF
# =========================================================================

def extract_nominal_params(model):
    """
    Extract the 10 inertial parameters per arm link from the Pinocchio model.
    For each link i (rigid body):
        ψ_i = [m, h_x, h_y, h_z, Ixx, Iyy, Izz, Ixy, Iyz, Ixz]
    where h = m*c (first mass moment) and I is about the joint origin frame.

    Returns Ψ₀ of shape (NJ*10,)
    """
    psi0 = np.zeros(NJ * 10)

    arm_jids = [model.getJointId(n) for n in ARM_JOINTS]

    for idx, jid in enumerate(arm_jids):
        # Pinocchio stores inertia as model.inertias[jid]
        inertia = model.inertias[jid]

        m = inertia.mass
        c = inertia.lever  # COM position relative to joint frame
        h = m * c          # first mass moment

        # Inertia tensor at COM frame from Pinocchio
        I_com = inertia.inertia  # 3x3 symmetric matrix at COM

        # We need inertia at the joint origin frame using parallel axis theorem:
        # I_O = I_com + m * (c^T c I_3 - c c^T)
        I_O = I_com + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))

        jname = ARM_JOINTS[idx]
        armature_val = ARMATURE.get(jname, 0.0)

        # Determine rotation axis from Pinocchio joint type
        # JointModelRX -> 0, JointModelRY -> 1, JointModelRZ -> 2
        shortname = model.joints[jid].shortname()
        axis_map = {'JointModelRX': 0, 'JointModelRY': 1, 'JointModelRZ': 2}
        aa = axis_map[shortname]
        I_O[aa, aa] += armature_val

        psi0[idx * 10 + 0] = m
        psi0[idx * 10 + 1] = h[0]
        psi0[idx * 10 + 2] = h[1]
        psi0[idx * 10 + 3] = h[2]
        psi0[idx * 10 + 4] = I_O[0, 0]  # Ixx
        psi0[idx * 10 + 5] = I_O[1, 1]  # Iyy
        psi0[idx * 10 + 6] = I_O[2, 2]  # Izz
        psi0[idx * 10 + 7] = I_O[0, 1]  # Ixy
        psi0[idx * 10 + 8] = I_O[1, 2]  # Iyz
        psi0[idx * 10 + 9] = I_O[0, 2]  # Ixz

        axis_labels = ['X→Ixx', 'Y→Iyy', 'Z→Izz']
        print(f"  Link {idx} ({ARM_JOINTS[idx]}): m={m:.4f}  "
              f"h=[{h[0]:.6f},{h[1]:.6f},{h[2]:.6f}]  "
              f"Ixx={I_O[0,0]:.6e}  "
              f"armature={armature_val:.6f} on {axis_labels[aa]}")

    return psi0


# =========================================================================
# 4. Pseudo-inertia matrix P(ψ) and entropic distance  (Eqs. 3.13-3.14)
# =========================================================================

def pseudo_inertia_matrix(psi_i):
    """
    Build 4×4 symmetric pseudo-inertia matrix P(ψ) from Eq. (3.13):
        P = [ 0.5*tr(I)*I_3 - I,  h^T ]
            [       h,             m   ]
    """
    m  = psi_i[0]
    h  = psi_i[1:4]
    Ixx, Iyy, Izz = psi_i[4], psi_i[5], psi_i[6]
    Ixy, Iyz, Ixz = psi_i[7], psi_i[8], psi_i[9]

    I = np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz],
    ])

    tr_I = np.trace(I)
    P = np.zeros((4, 4))
    P[:3, :3] = 0.5 * tr_I * np.eye(3) - I
    P[:3, 3]  = h
    P[3, :3]  = h
    P[3, 3]   = m

    return P


# def entropic_distance_sq(psi_all, psi0_all):
#     """
#     Entropic measurement d_F(Ψ, Ψ₀)² from Eq. (3.14):
#         d_F² = Σᵢ ( -log|P(ψᵢ)| + tr(P(ψᵢ₀)⁻¹ P(ψᵢ)) )
#     """
#     d = 0.0
#     for i in range(NJ):
#         psi_i  = psi_all[i*10:(i+1)*10]
#         psi0_i = psi0_all[i*10:(i+1)*10]

#         P  = pseudo_inertia_matrix(psi_i)
#         P0 = pseudo_inertia_matrix(psi0_i)

#         # Check positive definiteness
#         try:
#             det_P = np.linalg.det(P)
#             if det_P <= 0:
#                 return 1e12  # Infeasible

#             P0_inv = np.linalg.inv(P0)
#             d += -np.log(det_P) + np.trace(P0_inv @ P)
#         except np.linalg.LinAlgError:
#             return 1e12

#     return d

# def entropic_distance_sq_with_grad(psi_all, psi0_all):
#     """
#     Compute d_F² and its analytical gradient simultaneously.

#     Gradient derivation:
#         ∂d_F²/∂ψ_k = tr(W · ∂P/∂ψ_k)   where  W = -P⁻¹ + P₀⁻¹

#     Since W is symmetric (both P, P₀ are symmetric) and ∂P/∂ψ_k
#     are very sparse, tr(W · ∂P/∂ψ_k) reduces to element lookups:

#         ∂P/∂m    → (3,3) only             →  W[3,3]
#         ∂P/∂hx   → (0,3)+(3,0)            →  2·W[0,3]
#         ∂P/∂hy   → (1,3)+(3,1)            →  2·W[1,3]
#         ∂P/∂hz   → (2,3)+(3,2)            →  2·W[2,3]
#         ∂P/∂Ixx  → diag(-½, ½, ½, 0)     → -½W[0,0]+½W[1,1]+½W[2,2]
#         ∂P/∂Iyy  → diag( ½,-½, ½, 0)     →  ½W[0,0]-½W[1,1]+½W[2,2]
#         ∂P/∂Izz  → diag( ½, ½,-½, 0)     →  ½W[0,0]+½W[1,1]-½W[2,2]
#         ∂P/∂Ixy  → (0,1)+(1,0) with -1    → -2·W[0,1]
#         ∂P/∂Iyz  → (1,2)+(2,1) with -1    → -2·W[1,2]
#         ∂P/∂Ixz  → (0,2)+(2,0) with -1    → -2·W[0,2]
#     """
#     d = 0.0
#     grad = np.zeros_like(psi_all)

#     for i in range(NJ):
#         psi_i  = psi_all[i*10:(i+1)*10]
#         psi0_i = psi0_all[i*10:(i+1)*10]

#         P  = pseudo_inertia_matrix(psi_i)
#         P0 = pseudo_inertia_matrix(psi0_i)

#         try:
#             eigvals = np.linalg.eigvalsh(P)
#             if np.min(eigvals) <= 0:
#                 return 1e12, np.zeros_like(psi_all)

#             P_inv  = np.linalg.inv(P)
#             P0_inv = np.linalg.inv(P0)
#         except np.linalg.LinAlgError:
#             return 1e12, np.zeros_like(psi_all)

#         d += -np.log(np.linalg.det(P)) + np.trace(P0_inv @ P)

#         W = -P_inv + P0_inv

#         g = np.zeros(10)
#         g[0] = W[3, 3]                                        # ∂/∂m
#         g[1] = 2.0 * W[0, 3]                                  # ∂/∂hx
#         g[2] = 2.0 * W[1, 3]                                  # ∂/∂hy
#         g[3] = 2.0 * W[2, 3]                                  # ∂/∂hz
#         g[4] = -0.5*W[0,0] + 0.5*W[1,1] + 0.5*W[2,2]         # ∂/∂Ixx
#         g[5] =  0.5*W[0,0] - 0.5*W[1,1] + 0.5*W[2,2]         # ∂/∂Iyy
#         g[6] =  0.5*W[0,0] + 0.5*W[1,1] - 0.5*W[2,2]         # ∂/∂Izz
#         g[7] = -2.0 * W[0, 1]                                 # ∂/∂Ixy
#         g[8] = -2.0 * W[1, 2]                                 # ∂/∂Iyz
#         g[9] = -2.0 * W[0, 2]                                 # ∂/∂Ixz

#         grad[i*10:(i+1)*10] = g

#     return d, grad

def _safe_neg_log(x, eps=1e-4):
    """
    Smooth extension of -log(x):
      x > eps:  -log(x)
      x <= eps: -log(eps) + (eps-x)/eps + 0.5*((eps-x)/eps)^2
    Matches value and first derivative at x=eps, goes to +inf as x->-inf.
    """
    if x > eps:
        return -np.log(x)
    else:
        t = (eps - x) / eps
        return -np.log(eps) + t + 0.5 * t * t


def _safe_neg_log_deriv(x, eps=1e-4):
    """
    Derivative of _safe_neg_log w.r.t. x:
      x > eps:  -1/x
      x <= eps: -1/eps * (1 + (eps-x)/eps)
    """
    if x > eps:
        return -1.0 / x
    else:
        t = (eps - x) / eps
        return -1.0 / eps * (1.0 + t)


def entropic_distance_sq(psi_all, psi0_all):
    """
    Entropic measurement with smooth barrier (no hard 1e12 cutoff).
    """
    d = 0.0
    for i in range(NJ):
        psi_i  = psi_all[i*10:(i+1)*10]
        psi0_i = psi0_all[i*10:(i+1)*10]

        P  = pseudo_inertia_matrix(psi_i)
        P0 = pseudo_inertia_matrix(psi0_i)

        try:
            P0_inv = np.linalg.inv(P0)
        except np.linalg.LinAlgError:
            return 1e12

        eigvals = np.linalg.eigvalsh(P)
        # Smooth -log|P| = -Σ log(λ_k) with safe extension
        neg_log_det = sum(_safe_neg_log(e) for e in eigvals)
        d += neg_log_det + np.trace(P0_inv @ P)

    return d


def entropic_distance_sq_with_grad(psi_all, psi0_all):
    """
    Compute d_F² and analytical gradient with smooth barrier.

    Uses eigendecomposition for -log|P| to handle near-singular P smoothly.
    The gradient still uses W = ∂(-log|P|)/∂P + P₀⁻¹, but computes
    P⁻¹ via the eigendecomposition with safe reciprocals.
    """
    d = 0.0
    grad = np.zeros_like(psi_all)

    for i in range(NJ):
        psi_i  = psi_all[i*10:(i+1)*10]
        psi0_i = psi0_all[i*10:(i+1)*10]

        P  = pseudo_inertia_matrix(psi_i)
        P0 = pseudo_inertia_matrix(psi0_i)

        try:
            P0_inv = np.linalg.inv(P0)
        except np.linalg.LinAlgError:
            return 1e12, np.zeros_like(psi_all)

        # Eigendecomposition of P
        eigvals, eigvecs = np.linalg.eigh(P)

        # Smooth -log|P|
        neg_log_det = sum(_safe_neg_log(e) for e in eigvals)
        d += neg_log_det + np.trace(P0_inv @ P)

        # Smooth "P_inv" via eigendecomposition: P_inv_smooth = V diag(d(-log)/de) V^T
        # Note: d(-log(e))/dP = -P^{-1} when all e>0, but we use smooth extension
        neg_log_derivs = np.array([_safe_neg_log_deriv(e) for e in eigvals])
        # ∂(-log|P|)/∂P = V diag(-1/λ_k or smooth ext) V^T  (but with negative sign absorbed)
        # Actually: ∂(-Σlog(λ_k))/∂P = -P^{-1} when PD, so W = -P^{-1} + P0^{-1}
        # With smooth extension: "P_inv_smooth" replaces P^{-1}
        P_inv_smooth = eigvecs @ np.diag(-neg_log_derivs) @ eigvecs.T

        W = -P_inv_smooth + P0_inv

        g = np.zeros(10)
        g[0] = W[3, 3]
        g[1] = 2.0 * W[0, 3]
        g[2] = 2.0 * W[1, 3]
        g[3] = 2.0 * W[2, 3]
        g[4] = -0.5*W[0,0] + 0.5*W[1,1] + 0.5*W[2,2]
        g[5] =  0.5*W[0,0] - 0.5*W[1,1] + 0.5*W[2,2]
        g[6] =  0.5*W[0,0] + 0.5*W[1,1] - 0.5*W[2,2]
        g[7] = -2.0 * W[0, 1]
        g[8] = -2.0 * W[1, 2]
        g[9] = -2.0 * W[0, 2]

        grad[i*10:(i+1)*10] = g

    return d, grad

# =========================================================================
# 5. Solve the regularized least-squares problem (Problem 3.15)
# =========================================================================

# def solve_sysid(Gamma, tau_vec, psi0, lambda_reg=0.1):
#     """
#     Solve:
#         min_Ψ  ||ΓΨ - τ||² + λ * d_F(Ψ, Ψ₀)²

#     Using scipy.optimize.minimize with L-BFGS-B.
#     """
#     N_params = len(psi0)

#     def objective(psi):
#         residual = Gamma @ psi - tau_vec
#         data_term = np.dot(residual, residual)
#         reg_term = entropic_distance_sq(psi, psi0)
#         return data_term + lambda_reg * reg_term

#     def gradient(psi):
#         eps = 1e-7
#         grad = np.zeros_like(psi)
#         f0 = objective(psi)
#         for i in range(len(psi)):
#             psi[i] += eps
#             grad[i] = (objective(psi) - f0) / eps
#             psi[i] -= eps
#         return grad

#     print(f"\nSolving system identification (λ={lambda_reg})...")
#     print(f"  Params: {N_params}, Data points: {Gamma.shape[0]}")
#     print(f"  Γ shape: {Gamma.shape}")

#     # Initial guess: nominal parameters from URDF
#     x0 = psi0.copy()
#     f0 = objective(x0)
#     print(f"  Initial cost: {f0:.4f}")

#     # Set bounds: mass > 0, inertias can vary
#     bounds = []
#     for i in range(NJ):
#         bounds.append((1e-4, None))     # mass > 0
#         bounds.extend([(None, None)] * 3)  # first mass moment
#         bounds.extend([(1e-8, None)] * 3)  # Ixx, Iyy, Izz > 0
#         bounds.extend([(None, None)] * 3)  # Ixy, Iyz, Ixz

#     result = minimize(
#         objective,
#         x0,
#         jac=gradient,
#         method='L-BFGS-B',
#         bounds=bounds,
#         options={
#             'maxiter': 500,
#             'ftol': 1e-10,
#             'disp': True,
#         }
#     )

#     psi_opt = result.x
#     print(f"\n  Final cost: {result.fun:.4f}")
#     print(f"  Converged: {result.success}")
#     print(f"  Message: {result.message}")

#     return psi_opt

# def solve_sysid(Gamma, tau_vec, psi0, lambda_reg=0.1):
#     """
#     Solve:
#         min_Ψ  ||ΓΨ - τ||² + λ * d_F(Ψ, Ψ₀)²

#     Using scipy.optimize.minimize with L-BFGS-B.
#     Analytical gradient:
#         ∇ = 2·ΓᵀΓψ - 2·Γᵀτ + λ·∇d_F²
#     """
#     N_params = len(psi0)

#     # Precompute for the data term gradient (constant across iterations)
#     GtG = Gamma.T @ Gamma        # (50, 50)
#     Gttau = Gamma.T @ tau_vec    # (50,)

#     def objective_and_gradient(psi):
#         # --- Data term ---
#         residual = Gamma @ psi - tau_vec
#         data_cost = np.dot(residual, residual)
#         data_grad = 2.0 * (GtG @ psi - Gttau)

#         if lambda_reg == 0.0:
#             return data_cost, data_grad

#         # --- Entropic regularization term ---
#         reg_cost, reg_grad = entropic_distance_sq_with_grad(psi, psi0)

#         total_cost = data_cost + lambda_reg * reg_cost
#         total_grad = data_grad + lambda_reg * reg_grad

#         return total_cost, total_grad

#     print(f"\nSolving system identification (λ={lambda_reg})...")
#     print(f"  Params: {N_params}, Data points: {Gamma.shape[0]}")
#     print(f"  Γ shape: {Gamma.shape}")

#     # Initial guess: nominal parameters from URDF
#     x0 = psi0.copy()
#     f0, _ = objective_and_gradient(x0)
#     print(f"  Initial cost: {f0:.4f}")

#     # Set bounds: mass > 0, inertias can vary
#     bounds = []
#     for i in range(NJ):
#         bounds.append((1e-4, None))     # mass > 0
#         bounds.extend([(None, None)] * 3)  # first mass moment
#         bounds.extend([(1e-8, None)] * 3)  # Ixx, Iyy, Izz > 0
#         bounds.extend([(None, None)] * 3)  # Ixy, Iyz, Ixz

#     result = minimize(
#         objective_and_gradient,
#         x0,
#         jac=True,    # objective_and_gradient returns (f, grad) together
#         method='L-BFGS-B',
#         bounds=bounds,
#         options={
#             'maxiter': 100000,
#             'ftol': 1e-12,
#             'gtol': 1e-8,
#             'disp': True,
#         }
#     )

#     psi_opt = result.x
#     print(f"\n  Final cost: {result.fun:.4f}")
#     print(f"  Converged: {result.success}")
#     print(f"  Message: {result.message}")

#     return psi_opt

def solve_sysid_sdp(Gamma, Gamma_f, tau_vec, psi0, lambda_reg=0.0, ellipsoids_file=None, com_scale=1.0):
    """
    Solve system identification as SDP with physical consistency constraints.
    ...
    """
    import cvxpy as cp
    import os

    print(f"\nSolving SDP system identification (λ={lambda_reg})...")
    print(f"  Params: {len(psi0)}, Data points: {Gamma.shape[0]}")

    # Load COM ellipsoids if provided
    E_list, c_list = None, None
    if ellipsoids_file is not None and os.path.exists(ellipsoids_file):
        try:
            data = np.load(ellipsoids_file)
            E_list = [data[f'E_{i}'] for i in range(NJ)]
            c_list = [data[f'c_{i}'] for i in range(NJ)]
            print(f"  [+] Loaded bounding ellipsoids from: {ellipsoids_file}")
            print(f"  [+] Enforcing COM SOCP constraint (scale factor = {com_scale})")
        except Exception as e:
            print(f"  [-] Warning: Could not parse {ellipsoids_file}: {e}")

    # Decision variable
    psi = cp.Variable(len(psi0))
    n_fric = 2 * NJ  # 5 viscous + 5 Coulomb
    f_bar = cp.Variable(n_fric)

    # Data fidelity term
    # data_term = cp.sum_squares(Gamma @ psi - tau_vec)
    data_term = cp.sum_squares(Gamma @ psi + Gamma_f @ f_bar - tau_vec)

    if lambda_reg > 0:
        
        lambda_per_joint = np.array([
            lambda_reg,    # joint 0: shoulder_pitch
            lambda_reg,    # joint 1: shoulder_roll
            lambda_reg,    # joint 2: arm_yaw
            lambda_reg,    # joint 3: arm_roll
            lambda_reg * 0.01,  # joint 4: wrist_yaw (FT sensor → much lower)
        ])
        # Expand to per-parameter (10 params per joint)
        w = np.repeat(lambda_per_joint, 10)  # shape (50,)
        W_sqrt = np.diag(np.sqrt(w))

        reg_term = cp.sum_squares(W_sqrt @ (psi - psi0))

        objective = cp.Minimize(data_term + reg_term)
    else:
        objective = cp.Minimize(data_term)

    # Build 4x4 basis matrices for P(ψ) = Σ_k A_k · ψ_k
    A = np.zeros((10, 4, 4))
    A[0, 3, 3] = 1.0                                      # m
    A[1, 0, 3] = 1.0; A[1, 3, 0] = 1.0                    # hx
    A[2, 1, 3] = 1.0; A[2, 3, 1] = 1.0                    # hy
    A[3, 2, 3] = 1.0; A[3, 3, 2] = 1.0                    # hz
    A[4, 0, 0] = -0.5; A[4, 1, 1] = 0.5; A[4, 2, 2] = 0.5 # Ixx
    A[5, 0, 0] = 0.5; A[5, 1, 1] = -0.5; A[5, 2, 2] = 0.5 # Iyy
    A[6, 0, 0] = 0.5; A[6, 1, 1] = 0.5; A[6, 2, 2] = -0.5 # Izz
    A[7, 0, 1] = -1.0; A[7, 1, 0] = -1.0                  # Ixy
    A[8, 1, 2] = -1.0; A[8, 2, 1] = -1.0                  # Iyz
    A[9, 0, 2] = -1.0; A[9, 2, 0] = -1.0                  # Ixz

    # SDP constraints: P(ψ_i) ≽ 0 for each link
    constraints = []

    # Friction nonnegativity: f̄ ≥ 0
    constraints.append(f_bar >= 0)
    
    for i in range(NJ):
        # 1) LMI constraint (Positive Semidefinite pseudo-inertia)
        P_i = cp.Constant(np.zeros((4, 4)))
        for k in range(10):
            P_i = P_i + A[k] * psi[i * 10 + k]
        constraints.append(P_i >> 0)
        
        # 2) COM-in-ellipsoid SOCP constraint
        if E_list is not None and c_list is not None:
            m_i  = psi[i * 10 + 0]
            hx_i = psi[i * 10 + 1]
            hy_i = psi[i * 10 + 2]
            hz_i = psi[i * 10 + 3]
            h_i  = cp.hstack([hx_i, hy_i, hz_i])  # Affine expression (3,)
            
            # Fetch the geometric parameters
            E_i = E_list[i]
            c_i = c_list[i]
            
            # Ensure perfect symmetry to prevent Cholesky numerical errors
            E_i = (E_i + E_i.T) / 2.0 
            
            # Decompose E_i = L_i^T L_i
            # np.linalg.cholesky returns lower triangular L_lower, so L_i is the Transpose
            L_lower = np.linalg.cholesky(E_i)
            L_i = L_lower.T
            
            # Formulate the SOC expression: || L_i @ (h_i - m_i * c_i) ||_2 <= com_scale * m_i
            soc_expr = L_i @ (h_i - m_i * c_i)
            constraints.append(cp.norm(soc_expr, 2) <= com_scale * m_i)

    # Additional regularization constraints ======================================================
    # ---- Identical-assembly constraint: shoulder_roll and arm_roll ----
    # Same physical part mounted with 180° rotation about z on the actual hardware.
    SR, AR = 1, 3  # indices in ARM_JOINTS

    # mass and hz identical
    constraints.append(psi[AR*10 + 0] == psi[SR*10 + 0])   # m
    constraints.append(psi[AR*10 + 3] == psi[SR*10 + 3])   # hz

    # hx, hy negated  (180° about z flips x and y)
    constraints.append(psi[AR*10 + 1] == -psi[SR*10 + 1])  # hx
    constraints.append(psi[AR*10 + 2] == -psi[SR*10 + 2])  # hy

    # diagonal inertias and Ixy unchanged
    constraints.append(psi[AR*10 + 4] == psi[SR*10 + 4])   # Ixx
    constraints.append(psi[AR*10 + 5] == psi[SR*10 + 5])   # Iyy
    constraints.append(psi[AR*10 + 6] == psi[SR*10 + 6])   # Izz
    constraints.append(psi[AR*10 + 7] == psi[SR*10 + 7])   # Ixy

    # Ixz, Iyz negated
    constraints.append(psi[AR*10 + 8] == -psi[SR*10 + 8])  # Iyz
    constraints.append(psi[AR*10 + 9] == -psi[SR*10 + 9])  # Ixz

    m_motor_large = 0.300
    m_motor_small = 0.190
    m_ft_sensor = 0.160

    # m_1 ~ m_4 has weight >= m_motor_large
    constraints.append(psi[0*10 + 0] >= m_motor_large)
    constraints.append(psi[1*10 + 0] >= m_motor_large)
    constraints.append(psi[2*10 + 0] >= m_motor_large)
    constraints.append(psi[3*10 + 0] >= m_motor_large)

    constraints.append(psi[0*10 + 0] <= 2*m_motor_large)
    constraints.append(psi[1*10 + 0] <= 2*m_motor_large)
    constraints.append(psi[2*10 + 0] <= 2*m_motor_large)
    constraints.append(psi[3*10 + 0] <= 2*m_motor_large)

    # m_5 >= m_motor_small + 0.070 kg (Pinocchio merges hand_box_link via fixed joint)
    # constraints.append(psi[4*10 + 0] >= 0.07) #(m_motor_small + 0.07)) - was this too heavy??? There may be some bug for this mass in pinocchio!
    constraints.append(psi[4*10 + 0] >= (0.07 + m_ft_sensor))

    # m_1 + m_2 + m_3 + m_4 + m_5 <= 1.650 kg
    total_mass = sum(psi[i*10 + 0] for i in range(NJ))
    # constraints.append(total_mass <= 1.650 + 4*(m_motor_large-m_motor_small+0.1))
    constraints.append(total_mass <= 1.650 + 4*(m_motor_large-m_motor_small+0.1+m_ft_sensor))

    # Viscous (Nm·s/rad) and Coulomb (Nm) upper bounds
    for i in range(4):       # joints 0-3: big motor
        constraints.append(f_bar[i]     <= 1.0)   # fv
        constraints.append(f_bar[NJ+i]  <= 0.5)   # fc
    constraints.append(f_bar[4]     <= 0.5)       # fv wrist
    constraints.append(f_bar[NJ+4]  <= 0.3)       # fc wrist

    # Same motor type, friction shouldn't differ by more than
    # some absolute tolerance (loading differs, but hardware is similar)
    fv_spread = 0.3   # Nm·s/rad
    fc_spread = 0.1   # Nm
    for i in range(4):
        for j in range(i+1, 4):
            constraints.append(f_bar[i] - f_bar[j] <=  fv_spread)
            constraints.append(f_bar[i] - f_bar[j] >= -fv_spread)
            constraints.append(f_bar[NJ+i] - f_bar[NJ+j] <=  fc_spread)
            constraints.append(f_bar[NJ+i] - f_bar[NJ+j] >= -fc_spread)

    # End: Additional regularization constraints =================================================

    prob = cp.Problem(objective, constraints)

    print("Solve with Mosek ...")
    prob.solve(solver=cp.MOSEK, verbose=True)
    solver_used = "Mosek"

    print(f"\n  Solver: {solver_used}")
    print(f"  Status: {prob.status}")
    print(f"  Optimal cost: {prob.value:.4f}")

    psi_opt = psi.value
    f_opt = f_bar.value
    residual = Gamma @ psi_opt + Gamma_f @ f_opt - tau_vec
    rms = np.sqrt(np.mean(residual**2))
    print(f"  RMS: {rms:.4f} Nm")

    return psi_opt, f_opt

# =========================================================================
# 6. Validate and visualize
# =========================================================================

def validate(Gamma, Gamma_f, tau_vec, psi_opt, f_opt, psi0, t, plot_path="sysid_validation.png"):
    """
    Compare predicted torques from identified vs nominal parameters.
    """
    tau_pred_opt = Gamma @ psi_opt + Gamma_f @ f_opt
    tau_pred_nom = Gamma @ psi0

    N_steps = len(t)

    fig, axes = plt.subplots(NJ, 1, figsize=(14, 12), sharex=True)
    joint_names = [n.replace("r_", "").replace("_joint", "") for n in ARM_JOINTS]

    for i in range(NJ):
        tau_m = tau_vec[i::NJ][:N_steps]
        tau_o = tau_pred_opt[i::NJ][:N_steps]
        tau_n = tau_pred_nom[i::NJ][:N_steps]

        axes[i].plot(t[:len(tau_m)], tau_m, 'b-', alpha=0.6, label='Measured', linewidth=0.8)
        axes[i].plot(t[:len(tau_o)], tau_o, 'r-', alpha=0.8, label='Identified', linewidth=1.0)
        axes[i].plot(t[:len(tau_n)], tau_n, 'g--', alpha=0.5, label='URDF nominal', linewidth=0.8)
        axes[i].set_ylabel(f"{joint_names[i]}\nτ (Nm)")
        axes[i].legend(loc='upper right', fontsize=7)
        axes[i].grid(True, alpha=0.3)

        # RMS errors
        rms_opt = np.sqrt(np.mean((tau_m - tau_o[:len(tau_m)])**2))
        rms_nom = np.sqrt(np.mean((tau_m - tau_n[:len(tau_m)])**2))
        axes[i].set_title(f"RMS: identified={rms_opt:.4f}  nominal={rms_nom:.4f}", fontsize=9)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("System Identification Validation", fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved validation plot: {plot_path}")


def print_comparison(psi_opt, psi0):
    """Print identified vs nominal parameters side by side."""
    print(f"\n{'='*80}")
    print(f"{'Joint':<25} {'Param':<8} {'URDF (nom)':>12} {'Identified':>12} {'Change%':>10}")
    print(f"{'='*80}")

    param_names = ['m', 'hx', 'hy', 'hz', 'Ixx', 'Iyy', 'Izz', 'Ixy', 'Iyz', 'Ixz']
    joint_names = [n.replace("_joint", "") for n in ARM_JOINTS]

    for i in range(NJ):
        for j, pn in enumerate(param_names):
            idx = i * 10 + j
            v0 = psi0[idx]
            v1 = psi_opt[idx]
            pct = ((v1 - v0) / max(abs(v0), 1e-10)) * 100
            jname = joint_names[i] if j == 0 else ""
            print(f"{jname:<25} {pn:<8} {v0:>12.6f} {v1:>12.6f} {pct:>9.1f}%")

            # After hz (j==3), print derived COM in cm
            if j == 3:
                m0 = psi0[i * 10]
                m1 = psi_opt[i * 10]
                for k, axis in enumerate(["cx (cm)", "cy (cm)", "cz (cm)"]):
                    h0 = psi0[i * 10 + 1 + k]
                    h1 = psi_opt[i * 10 + 1 + k]
                    c0 = (h0 / m0) * 100  # to cm
                    c1 = (h1 / m1) * 100
                    pct_c = ((c1 - c0) / max(abs(c0), 1e-10)) * 100
                    print(f"{'':25} {axis:<8} {c0:>12.4f} {c1:>12.4f} {pct_c:>9.1f}%")

        print(f"{'-'*80}")

def verify_physical_consistency(psi_all, label=""):
    """Check triangle inequality and positive definiteness for each link."""
    print(f"\n{'='*70}")
    print(f"Physical Consistency Check: {label}")
    print(f"{'='*70}")
    param_names = ['Ixx', 'Iyy', 'Izz', 'Ixy', 'Iyz', 'Ixz']
    joint_names = [n.replace("_joint", "") for n in ARM_JOINTS]

    for i in range(NJ):
        psi_i = psi_all[i*10:(i+1)*10]
        m = psi_i[0]
        h = psi_i[1:4]
        Ixx, Iyy, Izz = psi_i[4], psi_i[5], psi_i[6]
        Ixy, Iyz, Ixz = psi_i[7], psi_i[8], psi_i[9]

        I_O = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz],
        ])

        # Eigenvalues of inertia tensor at joint origin
        eigs_O = np.linalg.eigvalsh(I_O)

        # Inertia at COM (remove parallel axis): I_com = I_O - m*(c^Tc I - cc^T)
        c = h / m if m > 1e-10 else np.zeros(3)
        I_com = I_O - m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
        eigs_com = np.linalg.eigvalsh(I_com)

        # Triangle inequality on principal moments (eigenvalues at COM)
        e = np.sort(eigs_com)  # ascending
        tri_ok = e[0] + e[1] >= e[2] - 1e-6

        # Positive semi-definiteness
        psd_O   = np.all(eigs_O > -1e-6)
        psd_com = np.all(eigs_com > -1e-6)

        # Pseudo-inertia P(ψ) positive definiteness
        P = pseudo_inertia_matrix(psi_i)
        eigs_P = np.linalg.eigvalsh(P)
        pd_P = np.all(eigs_P > -1e-6)

        status = "OK" if (tri_ok and psd_com and pd_P) else "FAIL"
        print(f"\n  {joint_names[i]:30s} [{status}]")
        print(f"    I_com eigenvalues:  [{eigs_com[0]:+.6e}, {eigs_com[1]:+.6e}, {eigs_com[2]:+.6e}]")
        print(f"    PSD (I at COM):     {psd_com}")
        print(f"    Triangle ineq:      {e[0]:.6e} + {e[1]:.6e} = {e[0]+e[1]:.6e} >= {e[2]:.6e}  {tri_ok}")
        print(f"    P(ψ) eigenvalues:   [{eigs_P[0]:+.6e}, {eigs_P[1]:+.6e}, {eigs_P[2]:+.6e}, {eigs_P[3]:+.6e}]")
        print(f"    P(ψ) pos. definite: {pd_P}")

# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="System identification with entropic regularization")
    parser.add_argument("--urdf", required=True, help="Path to robot URDF")
    parser.add_argument("--csv", required=True, help="Path to sysid_data.csv from hardware")
    parser.add_argument("--lambda_reg", type=float, default=0.1,
                        help="Regularization weight λ (default: 0.1)")
    parser.add_argument("--filter_order", type=int, default=4,
                        help="Savitzky-Golay filter order (default: 4)")
    parser.add_argument("--filter_window", type=int, default=101,
                        help="Savitzky-Golay filter window (default: 101, must be odd)")
    parser.add_argument("--t_skip_start", type=float, default=1.0,
                        help="Skip first N seconds of data (static friction)")
    parser.add_argument("--t_skip_end", type=float, default=5.0,
                        help="Skip last N seconds of data")
    parser.add_argument("--out", default="identified_results/identified_params.npz",
                        help="Output file for identified parameters")
    parser.add_argument("--ellipsoids", type=str, default=None,
                        help="Path to ellipsoids.npz to enforce strict COM SOCP constraints")
    parser.add_argument("--com_scale", type=float, default=1.0,
                        help="Scale factor for the COM ellipsoid. E.g., pass ~0.909 to completely strip a 1.1 margin.")
    args = parser.parse_args()

    # 1. Load and filter hardware data
    print("=" * 60)
    print("Step 1: Loading and filtering hardware data")
    print("=" * 60)
    t, q, dq, ddq, tau_meas = load_sysid_data(
        args.csv,
        t_start_skip=args.t_skip_start,
        t_end_skip=args.t_skip_end,
        sg_order=args.filter_order,
        sg_window=args.filter_window,
    )

    # 2. Setup Pinocchio and build regressor
    print("\n" + "=" * 60)
    print("Step 2: Building observation matrix Γ")
    print("=" * 60)
    model, data, arm_q, arm_v, arm_cols = setup_pinocchio(args.urdf)

    Gamma, Gamma_f = build_regressor(model, data, arm_q, arm_v, arm_cols, q, dq, ddq)

    # Flatten measured torques to match regressor layout
    N_steps = q.shape[0]
    tau_vec = np.zeros(N_steps * NJ)
    for k in range(N_steps):
        for i in range(NJ):
            tau_vec[k * NJ + i] = tau_meas[k, i]

    # Check regressor conditioning
    S = np.linalg.svd(Gamma, compute_uv=False)
    tol = max(Gamma.shape) * S[0] * np.finfo(float).eps
    rank = int(np.sum(S > tol))
    cond = S[0] / S[S > tol][-1] if np.any(S > tol) else float('inf')
    print(f"  Γ shape: {Gamma.shape}")
    print(f"  Rank:    {rank}/{Gamma.shape[1]}")
    print(f"  Cond(Γ): {cond:.1f}")

    # 3. Extract nominal parameters from URDF
    print("\n" + "=" * 60)
    print("Step 3: Extracting nominal Ψ₀ from URDF")
    print("=" * 60)
    psi0 = extract_nominal_params(model)

    # Baseline: least-squares without regularization
    residual_nom = Gamma @ psi0 - tau_vec
    rms_nom = np.sqrt(np.mean(residual_nom**2))
    print(f"\n  Nominal URDF RMS error: {rms_nom:.4f} Nm")

    # 4. Solve with entropic regularization
    # print("\n" + "=" * 60)
    # print("Step 4: Solving regularized least-squares (entropic)")
    # print("=" * 60)
    # psi_opt = solve_sysid(Gamma, tau_vec, psi0, lambda_reg=args.lambda_reg)

    # 4. Solve with SDP constraints
    print("\n" + "=" * 60)
    print("Step 4: Solving constrained least-squares (SDP)")
    print("=" * 60)
    psi_opt, f_opt = solve_sysid_sdp(Gamma, Gamma_f, tau_vec, psi0, 
                                     lambda_reg=args.lambda_reg,
                                     ellipsoids_file=args.ellipsoids,
                                     com_scale=args.com_scale)

    residual_opt = Gamma @ psi_opt + Gamma_f @ f_opt - tau_vec
    print(f"\n  RMS errors:")
    print(f"    URDF nominal:       {rms_nom:.4f} Nm")
    print(f"    Entropic regularized: {np.sqrt(np.mean(residual_opt**2)):.4f} Nm")

    # Print friction results
    fv_opt = f_opt[:NJ]
    fc_opt = f_opt[NJ:]
    print(f"\n{'='*80}")
    print(f"{'Joint':<30} {'fv (viscous)':>14} {'fc (Coulomb)':>14}")
    print(f"{'='*80}")
    for i, jname in enumerate(ARM_JOINTS):
        print(f"{jname:<30} {fv_opt[i]:>14.6f} {fc_opt[i]:>14.6f}")

    # 5. Print comparison
    print_comparison(psi_opt, psi0)
    verify_physical_consistency(psi0, "URDF nominal")
    verify_physical_consistency(psi_opt, "Identified")

    # 6. Validate and plot
    print("\n" + "=" * 60)
    print("Step 5: Validation plots")
    print("=" * 60)
    validate(Gamma, Gamma_f, tau_vec, psi_opt, f_opt, psi0, t, plot_path="identified_results/sysid_validation.png")

    # 7. Save results
    np.savez(args.out,
             psi_opt=psi_opt,
             f_opt=f_opt,
             psi0=psi0,
             joint_names=ARM_JOINTS,
             lambda_reg=args.lambda_reg)
    print(f"\nSaved identified parameters: {args.out}")
    print("Done!")


if __name__ == "__main__":
    main()
