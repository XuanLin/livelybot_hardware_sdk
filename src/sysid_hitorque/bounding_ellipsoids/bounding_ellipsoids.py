#!/usr/bin/env python3
"""
Bounding Ellipsoid Generator & Visualizer for Robot System Identification.

Automatically computes Minimum Volume Enclosing Ellipsoids (MVEE) from
STL link meshes and generates the Q_i matrices for the constraint:

    tr(P(ψ_i) Q_i) >= 0       (thesis Eq. 3.15a)
    COM_i ∈ ellipsoid_i        (direct SOCP constraint)

Features:
  - Parses URDF to find STL mesh for each joint's child link
  - Computes MVEE via Khachiyan's algorithm
  - Per-link visualization: mesh + ellipsoid + URDF COM
  - Full arm assembly visualization with forward kinematics
  - Saves Q, E, c matrices for use in solve_sysid.py

Usage:
    python3 bounding_ellipsoids.py \
        --urdf hi.urdf \
        --mesh_dir ./meshes \
        --scale 1.1 \
        --save_npz ellipsoids.npz

Dependencies:
    pip install numpy trimesh matplotlib scipy
"""

import argparse
import numpy as np
import xml.etree.ElementTree as ET
import os
import sys

# ── Configuration ──────────────────────────────────────────────────────────

ARM_JOINTS = [
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_arm_yaw_joint",
    "r_arm_roll_joint",
    "r_wrist_yaw_joint",
]

JOINT_DISPLAY_NAMES = [
    "Shoulder Pitch",
    "Shoulder Roll",
    "Arm Yaw",
    "Arm Roll",
    "Wrist Yaw",
]

# Colors for each link
LINK_COLORS = [
    (0.85, 0.33, 0.33, 0.35),   # red-ish
    (0.33, 0.70, 0.33, 0.35),   # green-ish
    (0.33, 0.50, 0.85, 0.35),   # blue-ish
    (0.80, 0.60, 0.20, 0.35),   # gold
    (0.60, 0.33, 0.75, 0.35),   # purple
]

ELLIPSOID_COLORS = [
    (0.85, 0.33, 0.33, 0.15),
    (0.33, 0.70, 0.33, 0.15),
    (0.33, 0.50, 0.85, 0.15),
    (0.80, 0.60, 0.20, 0.15),
    (0.60, 0.33, 0.75, 0.15),
]

NJ = 5


# ═══════════════════════════════════════════════════════════════════════════
#  1.  MVEE — Minimum Volume Enclosing Ellipsoid
# ═══════════════════════════════════════════════════════════════════════════

def mvee(points, tol=1e-4, max_iter=10000):
    """
    Minimum Volume Enclosing Ellipsoid (Khachiyan / Todd-Yıldırım).

    Returns
    -------
    E : (3,3) positive-definite shape matrix
        Ellipsoid = { x : (x - c)^T E (x - c) <= 1 }
    c : (3,) center
    """
    pts = np.asarray(points, dtype=np.float64)
    M, d = pts.shape
    Q = np.column_stack([pts, np.ones(M)]).T          # (d+1, M)
    u = np.ones(M) / M

    for _ in range(max_iter):
        S = Q @ np.diag(u) @ Q.T
        try:
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Si = np.linalg.inv(S + 1e-10 * np.eye(d + 1))

        g = np.einsum('ij,jk,ik->i', Q.T, Si, Q.T)
        jm = np.argmax(g)
        gm = g[jm]
        step = (gm - (d + 1)) / ((d + 1) * (gm - 1))
        if step < tol:
            break
        u *= (1 - step)
        u[jm] += step

    c = pts.T @ u
    diff = pts - c
    Sx = (diff.T * u) @ diff
    try:
        Sxi = np.linalg.inv(Sx)
    except np.linalg.LinAlgError:
        Sxi = np.linalg.pinv(Sx)

    return Sxi / d, c


# ═══════════════════════════════════════════════════════════════════════════
#  2.  STL loading
# ═══════════════════════════════════════════════════════════════════════════

def load_stl_vertices_and_faces(stl_path):
    """Return (vertices (N,3), faces (F,3)) from an STL file."""
    try:
        import trimesh
        mesh = trimesh.load(stl_path)
        return np.array(mesh.vertices), np.array(mesh.faces)
    except ImportError:
        pass

    # Fallback: binary STL manual parse (vertices only, faces as sequential triples)
    verts, faces = _parse_binary_stl_full(stl_path)
    return verts, faces


def _parse_binary_stl_full(path):
    """Parse binary STL → unique vertices + face indices."""
    with open(path, 'rb') as f:
        f.read(80)
        n_tri = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        raw_verts = []
        for _ in range(n_tri):
            data = np.frombuffer(f.read(48), dtype=np.float32)
            f.read(2)  # attribute byte count
            raw_verts.extend([data[3:6], data[6:9], data[9:12]])

    raw_verts = np.array(raw_verts, dtype=np.float64)
    # Deduplicate vertices and build face index array
    unique, inverse = np.unique(np.round(raw_verts, 8), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return unique, faces


# ═══════════════════════════════════════════════════════════════════════════
#  3.  URDF parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_urdf(urdf_path, joint_names, mesh_dir=None):
    """
    Parse URDF and return per-joint info:
      - STL path
      - Joint origin (xyz, rpy)
      - Link inertial (mass, COM, inertia)
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Build maps
    joint_elems = {j.get('name'): j for j in root.findall('joint')}
    link_elems  = {l.get('name'): l for l in root.findall('link')}

    results = []
    for jname in joint_names:
        info = {'joint_name': jname}

        j_el = joint_elems.get(jname)
        if j_el is None:
            results.append(info)
            continue

        # Joint origin
        origin_el = j_el.find('origin')
        if origin_el is not None:
            xyz = np.array([float(x) for x in origin_el.get('xyz', '0 0 0').split()])
            rpy = np.array([float(x) for x in origin_el.get('rpy', '0 0 0').split()])
        else:
            xyz, rpy = np.zeros(3), np.zeros(3)
        info['origin_xyz'] = xyz
        info['origin_rpy'] = rpy

        # Axis
        axis_el = j_el.find('axis')
        if axis_el is not None:
            info['axis'] = np.array([float(x) for x in axis_el.get('xyz', '0 0 1').split()])
        else:
            info['axis'] = np.array([0, 0, 1.0])

        # Child link
        child_name = j_el.find('child').get('link')
        info['child_link'] = child_name

        l_el = link_elems.get(child_name)
        if l_el is not None:
            # Inertial
            inertial = l_el.find('inertial')
            if inertial is not None:
                mass_el = inertial.find('mass')
                info['mass'] = float(mass_el.get('value')) if mass_el is not None else 0

                com_el = inertial.find('origin')
                if com_el is not None:
                    info['com'] = np.array([float(x) for x in com_el.get('xyz', '0 0 0').split()])
                else:
                    info['com'] = np.zeros(3)

            # Mesh file
            vis = l_el.find('visual')
            if vis is not None:
                geom = vis.find('geometry')
                if geom is not None:
                    mesh_el = geom.find('mesh')
                    if mesh_el is not None:
                        fn = mesh_el.get('filename')
                        if mesh_dir:
                            info['stl_path'] = os.path.join(mesh_dir, os.path.basename(fn))
                        elif fn.startswith('file://'):
                            info['stl_path'] = fn[7:]
                        else:
                            info['stl_path'] = os.path.basename(fn)

        results.append(info)

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  4.  Homogeneous transforms for forward kinematics
# ═══════════════════════════════════════════════════════════════════════════

def rpy_to_rotation(rpy):
    """Roll-pitch-yaw (XYZ intrinsic) to 3×3 rotation matrix."""
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_transform(xyz, rpy):
    """Build 4×4 homogeneous transform from xyz translation + rpy rotation."""
    T = np.eye(4)
    T[:3, :3] = rpy_to_rotation(rpy)
    T[:3, 3] = xyz
    return T


def transform_points(T, pts):
    """Apply 4×4 transform to (N,3) points → (N,3)."""
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])            # (N, 4)
    return (T @ pts_h.T).T[:, :3]


# ═══════════════════════════════════════════════════════════════════════════
#  5.  Q matrix & constraint helpers
# ═══════════════════════════════════════════════════════════════════════════

def ellipsoid_to_Q(E, c):
    """
    Ellipsoid { x : (x-c)^T E (x-c) <= 1 }
    → Q such that { x : [x;1]^T Q [x;1] >= 0 }
    """
    Ec = E @ c
    Q = np.zeros((4, 4))
    Q[:3, :3] = -E
    Q[:3, 3] = Ec
    Q[3, :3] = Ec
    Q[3, 3] = 1.0 - c @ Ec
    return Q


def build_Q_matrices_from_urdf(urdf_path, mesh_dir, joint_names, scale=1.1):
    """
    Convenience function: parse URDF + load meshes + compute Q, E, c.

    This is the main entry point for library usage from solve_sysid.py.

    Args:
        urdf_path:   path to URDF
        mesh_dir:    directory containing STL files
        joint_names: ordered list of joint names
        scale:       inflate ellipsoid (1.1 = 10% margin, recommended)
    Returns:
        Q_list: list of (4,4) Q matrices (for tr(P*Q) constraint)
        E_list: list of (3,3) E matrices (for COM constraint)
        c_list: list of (3,) ellipsoid centers
    """
    joint_infos = parse_urdf(urdf_path, joint_names, mesh_dir)

    Q_list, E_list, c_list = [], [], []
    for idx, info in enumerate(joint_infos):
        stl_path = info.get('stl_path')
        if stl_path and os.path.exists(stl_path):
            verts, _ = load_stl_vertices_and_faces(stl_path)
            E_raw, c = mvee(verts, tol=1e-5)
            E = E_raw / (scale ** 2)
            semi = 1.0 / np.sqrt(np.linalg.eigvalsh(E))
            print(f"  [{joint_names[idx]:35s}] {len(verts):>5} verts, "
                  f"semi=[{semi[0]:.4f},{semi[1]:.4f},{semi[2]:.4f}]")
        else:
            print(f"  WARNING: STL not found for '{joint_names[idx]}': {stl_path}")
            print(f"           Using URDF COM as center with 5cm radius fallback.")
            print(f"           For proper constraints, fix --mesh_dir path.")
            # Fallback: small sphere (5cm radius) centered at URDF COM
            com = info.get('com', np.zeros(3))
            r_fallback = 0.05  # 5 cm
            E = np.eye(3) / (r_fallback ** 2)
            c = com.copy()

        Q_list.append(ellipsoid_to_Q(E, c))
        E_list.append(E)
        c_list.append(c)

    return Q_list, E_list, c_list


# ═══════════════════════════════════════════════════════════════════════════
#  6.  Ellipsoid surface mesh (for visualization)
# ═══════════════════════════════════════════════════════════════════════════

def ellipsoid_surface(E, c, n_u=40, n_v=25):
    """Generate surface points of ellipsoid defined by E, c."""
    eigvals, eigvecs = np.linalg.eigh(E)
    semi = 1.0 / np.sqrt(np.clip(eigvals, 1e-12, None))
    T = eigvecs @ np.diag(semi)

    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([xs.ravel(), ys.ravel(), zs.ravel()])   # (3, N)
    pts = (T @ sphere).T + c                                   # (N, 3)

    return (pts[:, 0].reshape(xs.shape),
            pts[:, 1].reshape(ys.shape),
            pts[:, 2].reshape(zs.shape))


def ellipsoid_wireframe(E, c, n_rings=12, n_pts=60):
    """Generate wireframe rings of the ellipsoid (for cleaner overlay on mesh)."""
    eigvals, eigvecs = np.linalg.eigh(E)
    semi = 1.0 / np.sqrt(np.clip(eigvals, 1e-12, None))
    T = eigvecs @ np.diag(semi)

    lines = []
    theta = np.linspace(0, 2 * np.pi, n_pts)

    # Rings in XY, XZ, YZ planes of the ellipsoid
    for phi in np.linspace(0, np.pi, n_rings // 3 + 1)[:-1]:
        # Latitude ring
        ring = np.array([np.cos(theta) * np.sin(phi),
                         np.sin(theta) * np.sin(phi),
                         np.full_like(theta, np.cos(phi))])
        lines.append((T @ ring).T + c)

    for phi in np.linspace(0, np.pi, n_rings // 3 + 1)[:-1]:
        # Longitude ring (XZ)
        ring = np.array([np.cos(theta) * np.cos(phi) + np.sin(theta) * np.sin(phi) * 0,
                         np.full_like(theta, 0),
                         np.sin(theta)])
        # Simpler: great circles
        ring = np.array([np.cos(theta),
                         np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi)])
        lines.append((T @ ring).T + c)

    return lines


# ═══════════════════════════════════════════════════════════════════════════
#  7.  CVXPY constraint builders (for use in solve_sysid.py)
# ═══════════════════════════════════════════════════════════════════════════

def _basis_matrices_P():
    """10 basis matrices A_k such that P(psi) = sum_k A_k * psi_k."""
    A = np.zeros((10, 4, 4))
    A[0, 3, 3] = 1.0
    A[1, 0, 3] = 1.0; A[1, 3, 0] = 1.0
    A[2, 1, 3] = 1.0; A[2, 3, 1] = 1.0
    A[3, 2, 3] = 1.0; A[3, 3, 2] = 1.0
    A[4, 0, 0] = -0.5; A[4, 1, 1] = 0.5; A[4, 2, 2] = 0.5
    A[5, 0, 0] = 0.5; A[5, 1, 1] = -0.5; A[5, 2, 2] = 0.5
    A[6, 0, 0] = 0.5; A[6, 1, 1] = 0.5; A[6, 2, 2] = -0.5
    A[7, 0, 1] = -1.0; A[7, 1, 0] = -1.0
    A[8, 1, 2] = -1.0; A[8, 2, 1] = -1.0
    A[9, 0, 2] = -1.0; A[9, 2, 0] = -1.0
    return A


def add_trace_PQ_constraints(psi_var, Q_list, NJ=5):
    """
    Thesis Eq. (3.15a):  tr(P(psi_i) Q_i) >= 0
    Linear in psi. One scalar constraint per link.
    """
    import cvxpy as cp
    A = _basis_matrices_P()
    constraints = []
    for i in range(NJ):
        coeffs = np.array([np.trace(A[k] @ Q_list[i]) for k in range(10)])
        constraints.append(coeffs @ psi_var[i*10:(i+1)*10] >= 0)
    return constraints


def add_com_in_ellipsoid_constraints(psi_var, E_list, c_list, NJ=5):
    """
    Direct COM-in-ellipsoid (SOCP):  || L(h - m*c_ell) ||_2  <=  m
    One second-order cone constraint per link.
    """
    import cvxpy as cp
    constraints = []
    for i in range(NJ):
        try:
            L = np.linalg.cholesky(E_list[i])
            L_up = L.T
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(E_list[i])
            eigvals = np.clip(eigvals, 1e-10, None)
            L_up = np.diag(np.sqrt(eigvals)) @ eigvecs.T

        m_i = psi_var[i * 10]
        h_i = psi_var[i * 10 + 1: i * 10 + 4]
        constraints.append(cp.SOC(m_i, L_up @ (h_i - m_i * c_list[i])))
    return constraints


# ═══════════════════════════════════════════════════════════════════════════
#  8. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def visualize_per_link(joint_infos, E_list, c_list, semi_list, save_path=None):
    """
    Per-link subplot: mesh triangles + ellipsoid wireframe + COM markers.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    n = len(joint_infos)
    fig = plt.figure(figsize=(4.5 * n, 6))

    for idx, (info, E_i, c_i, semi_i) in enumerate(zip(joint_infos, E_list, c_list, semi_list)):
        ax = fig.add_subplot(1, n, idx + 1, projection='3d')

        stl_path = info.get('stl_path')
        has_mesh = stl_path and os.path.exists(stl_path)

        if has_mesh:
            verts, faces = load_stl_vertices_and_faces(stl_path)

            # Subsample faces for performance if too many
            if len(faces) > 3000:
                idx_sub = np.random.choice(len(faces), 3000, replace=False)
                faces_sub = faces[idx_sub]
            else:
                faces_sub = faces

            # Plot mesh as semi-transparent triangles
            tri_verts = verts[faces_sub]
            mesh_col = Poly3DCollection(tri_verts,
                                        alpha=LINK_COLORS[idx][3],
                                        facecolor=LINK_COLORS[idx][:3],
                                        edgecolor=(0.3, 0.3, 0.3, 0.08),
                                        linewidth=0.2)
            ax.add_collection3d(mesh_col)

        # Plot ellipsoid as wireframe
        xe, ye, ze = ellipsoid_surface(E_i, c_i, n_u=30, n_v=18)
        ax.plot_surface(xe, ye, ze,
                        alpha=ELLIPSOID_COLORS[idx][3],
                        color=ELLIPSOID_COLORS[idx][:3],
                        edgecolor=(*LINK_COLORS[idx][:3], 0.3),
                        linewidth=0.3)

        # Plot URDF COM
        com = info.get('com', np.zeros(3))
        ax.scatter(*com, s=80, c='red', marker='x', linewidths=2.5,
                   zorder=10, label=f'URDF COM')

        # Plot ellipsoid center
        ax.scatter(*c_i, s=60, c='blue', marker='+', linewidths=2.0,
                   zorder=10, label='Ellipsoid center')

        # Axis labels and title
        ax.set_xlabel('X (m)', fontsize=7, labelpad=1)
        ax.set_ylabel('Y (m)', fontsize=7, labelpad=1)
        ax.set_zlabel('Z (m)', fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6)

        mass = info.get('mass', 0)
        ax.set_title(f"{JOINT_DISPLAY_NAMES[idx]}\n"
                     f"m={mass:.3f} kg  semi=[{semi_i[0]:.3f},{semi_i[1]:.3f},{semi_i[2]:.3f}]",
                     fontsize=8, pad=2)
        ax.legend(fontsize=6, loc='upper left', markerscale=0.7)

        # Equal aspect ratio
        if has_mesh:
            all_pts = np.vstack([verts, c_i.reshape(1, 3)])
        else:
            all_pts = c_i.reshape(1, 3)
        max_range = max(semi_i) * 1.5
        mid = c_i
        ax.set_xlim([mid[0] - max_range, mid[0] + max_range])
        ax.set_ylim([mid[1] - max_range, mid[1] + max_range])
        ax.set_zlim([mid[2] - max_range, mid[2] + max_range])

        ax.view_init(elev=20, azim=-60)

    plt.suptitle('Per-Link Bounding Ellipsoids', fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight', pad_inches=0.1)
        print(f"  Saved: {save_path}")
    return fig


def visualize_assembly(joint_infos, E_list, c_list, semi_list, save_path=None):
    """
    Full arm assembly: forward kinematics to place all links in world frame.
    Shows meshes + ellipsoids + joint frames in a single 3D view.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Forward kinematics: chain of transforms
    # Start from base_link frame
    T_world = np.eye(4)

    all_joint_origins = [T_world[:3, 3].copy()]
    all_points = []

    for idx, (info, E_i, c_i, semi_i) in enumerate(zip(joint_infos, E_list, c_list, semi_list)):
        # Apply joint transform
        xyz = info.get('origin_xyz', np.zeros(3))
        rpy = info.get('origin_rpy', np.zeros(3))
        T_joint = make_transform(xyz, rpy)
        T_world = T_world @ T_joint

        joint_pos = T_world[:3, 3].copy()
        all_joint_origins.append(joint_pos)

        stl_path = info.get('stl_path')
        has_mesh = stl_path and os.path.exists(stl_path)

        if has_mesh:
            verts, faces = load_stl_vertices_and_faces(stl_path)

            # Transform mesh vertices to world frame
            verts_w = transform_points(T_world, verts)

            if len(faces) > 2000:
                face_sub = faces[np.random.choice(len(faces), 2000, replace=False)]
            else:
                face_sub = faces

            tri_verts = verts_w[face_sub]
            mesh_col = Poly3DCollection(tri_verts,
                                        alpha=LINK_COLORS[idx][3],
                                        facecolor=LINK_COLORS[idx][:3],
                                        edgecolor=(0.2, 0.2, 0.2, 0.05),
                                        linewidth=0.15)
            ax.add_collection3d(mesh_col)
            all_points.append(verts_w)

        # Transform ellipsoid to world frame
        R_w = T_world[:3, :3]
        t_w = T_world[:3, 3]
        c_w = R_w @ c_i + t_w
        # E transforms as: E_w = R^{-T} E R^{-1} = (R E^{-1} R^T)^{-1}
        # For surface generation, transform the surface points instead
        xe_l, ye_l, ze_l = ellipsoid_surface(E_i, c_i, n_u=25, n_v=15)
        shape = xe_l.shape
        ell_pts = np.stack([xe_l.ravel(), ye_l.ravel(), ze_l.ravel()], axis=1)
        ell_pts_w = transform_points(T_world, ell_pts)

        xe_w = ell_pts_w[:, 0].reshape(shape)
        ye_w = ell_pts_w[:, 1].reshape(shape)
        ze_w = ell_pts_w[:, 2].reshape(shape)

        ax.plot_surface(xe_w, ye_w, ze_w,
                        alpha=0.10,
                        color=LINK_COLORS[idx][:3],
                        edgecolor=(*LINK_COLORS[idx][:3], 0.15),
                        linewidth=0.2)

        # COM in world frame
        com_local = info.get('com', np.zeros(3))
        com_w = R_w @ com_local + t_w
        ax.scatter(*com_w, s=50, c=[LINK_COLORS[idx][:3]], marker='x',
                   linewidths=2, zorder=10)

        # Ellipsoid center in world frame
        ax.scatter(*c_w, s=35, c=[LINK_COLORS[idx][:3]], marker='+',
                   linewidths=1.5, zorder=10)

        # Label
        ax.text(c_w[0], c_w[1], c_w[2] + max(semi_i) * 0.5,
                JOINT_DISPLAY_NAMES[idx], fontsize=6, ha='center',
                color=LINK_COLORS[idx][:3], fontweight='bold')

    # Draw kinematic chain (joint-to-joint lines)
    origins = np.array(all_joint_origins)
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2],
            'k-o', linewidth=2, markersize=5, zorder=20, label='Joint chain')

    # Draw small coordinate frames at each joint
    frame_len = 0.015
    T_accum = np.eye(4)
    for idx, info in enumerate(joint_infos):
        xyz = info.get('origin_xyz', np.zeros(3))
        rpy = info.get('origin_rpy', np.zeros(3))
        T_accum = T_accum @ make_transform(xyz, rpy)
        o = T_accum[:3, 3]
        R = T_accum[:3, :3]
        for k, color in enumerate(['r', 'g', 'b']):
            tip = o + R[:, k] * frame_len
            ax.plot([o[0], tip[0]], [o[1], tip[1]], [o[2], tip[2]],
                    color=color, linewidth=1.5, alpha=0.7)

    # Aesthetics
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Z (m)', fontsize=9)
    ax.set_title('Right Arm Assembly — Links + Bounding Ellipsoids', fontsize=13, fontweight='bold')

    # Legend patches
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', marker='o', linewidth=2, markersize=5, label='Joint chain'),
    ]
    for idx in range(NJ):
        legend_elements.append(
            Patch(facecolor=LINK_COLORS[idx][:3], alpha=0.4, label=JOINT_DISPLAY_NAMES[idx])
        )
    legend_elements += [
        Line2D([0], [0], marker='x', color='gray', linewidth=0, markersize=8, label='URDF COM'),
        Line2D([0], [0], marker='+', color='gray', linewidth=0, markersize=8, label='Ellipsoid center'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

    # Auto-scale
    if all_points:
        all_pts = np.vstack(all_points + [origins])
    else:
        all_pts = origins
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    span = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2
    ax.set_xlim([mid[0] - span, mid[0] + span])
    ax.set_ylim([mid[1] - span, mid[1] + span])
    ax.set_zlim([mid[2] - span, mid[2] + span])

    ax.view_init(elev=15, azim=-55)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight', pad_inches=0.1)
        print(f"  Saved: {save_path}")
    return fig


def print_report(joint_infos, E_list, c_list, semi_list, scale):
    """Print a summary table."""
    print(f"\n{'═' * 85}")
    print(f"  Bounding Ellipsoid Report  (scale = {scale})")
    print(f"{'═' * 85}")
    print(f"  {'Joint':<22} {'Mass':>6} {'URDF COM (cm)':>26} "
          f"{'Ell. Center (cm)':>26}")
    print(f"  {'':<22} {'(kg)':>6} {'x':>8}{'y':>8}{'z':>8}  "
          f"{'x':>8}{'y':>8}{'z':>8}")
    print(f"  {'─' * 80}")

    for idx, (info, c_i, semi_i) in enumerate(zip(joint_infos, c_list, semi_list)):
        mass = info.get('mass', 0)
        com = info.get('com', np.zeros(3)) * 100   # to cm
        cc = c_i * 100

        name = JOINT_DISPLAY_NAMES[idx]
        print(f"  {name:<22} {mass:>6.3f} "
              f"{com[0]:>8.3f}{com[1]:>8.3f}{com[2]:>8.3f}  "
              f"{cc[0]:>8.3f}{cc[1]:>8.3f}{cc[2]:>8.3f}")

    print()
    print(f"  {'Joint':<22} {'Semi-axes (cm)':>30}  {'Verts':>6}  {'All inside?':>12}")
    print(f"  {'':<22} {'a':>9}{'b':>9}{'c':>9}  {'':<6}  {'':<12}")
    print(f"  {'─' * 80}")

    for idx, (info, E_i, c_i, semi_i) in enumerate(zip(joint_infos, E_list, c_list, semi_list)):
        sc = np.sort(semi_i) * 100
        stl_path = info.get('stl_path')
        if stl_path and os.path.exists(stl_path):
            verts = load_stl_vertices_and_faces(stl_path)[0]
            n_v = len(verts)
            Q = ellipsoid_to_Q(E_i, c_i)
            ones = np.ones((n_v, 1))
            V = np.hstack([verts, ones])
            scores = np.einsum('ij,jk,ik->i', V, Q, V)
            n_out = np.sum(scores < -1e-8)
            inside = f"{'YES ✓' if n_out == 0 else f'NO ({n_out} out)'}"
        else:
            n_v = 0
            inside = "no mesh"

        name = JOINT_DISPLAY_NAMES[idx]
        print(f"  {name:<22} {sc[0]:>9.3f}{sc[1]:>9.3f}{sc[2]:>9.3f}"
              f"  {n_v:>6}  {inside:>12}")

    print(f"{'═' * 85}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  10.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Bounding Ellipsoid Generator & Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 bounding_ellipsoids.py --urdf hi.urdf --mesh_dir ./meshes
  python3 bounding_ellipsoids.py --urdf hi.urdf --mesh_dir ./meshes --scale 1.2 --save_npz ell.npz
  python3 bounding_ellipsoids.py --urdf hi.urdf --mesh_dir ./meshes --no_show --save_fig output.png
        """)
    parser.add_argument('--urdf', required=True, help='Path to URDF file')
    parser.add_argument('--mesh_dir', required=True, help='Directory containing STL files')
    parser.add_argument('--scale', type=float, default=1.1,
                        help='Ellipsoid inflation factor (default: 1.1 = 10%% margin)')
    parser.add_argument('--joints', nargs='+', default=None,
                        help='Joint names (default: right arm joints)')
    parser.add_argument('--save_npz', default=None,
                        help='Save Q, E, c matrices to .npz file')
    parser.add_argument('--save_fig', default=None,
                        help='Save figures to file (prefix, e.g. "output" → output_links.png, output_assembly.png)')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display interactive plots')
    args = parser.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    joint_names = args.joints or ARM_JOINTS

    # ── Validate mesh directory ──
    mesh_dir = os.path.expanduser(args.mesh_dir)   # expand ~/...
    mesh_dir = os.path.abspath(mesh_dir)
    if not os.path.isdir(mesh_dir):
        print(f"\n  ERROR: Mesh directory does not exist: {mesh_dir}")
        print(f"         You passed: --mesh_dir {args.mesh_dir}")
        print(f"\n  Please provide the full path to the directory containing your STL files.")
        print(f"  Example:")
        print(f"    --mesh_dir ~/xuan_ws/src/.../hi_description/meshes")
        sys.exit(1)

    stl_files = [f for f in os.listdir(mesh_dir) if f.lower().endswith('.stl')]
    if len(stl_files) == 0:
        print(f"\n  ERROR: No .STL files found in: {mesh_dir}")
        print(f"  Contents: {os.listdir(mesh_dir)[:10]}")
        sys.exit(1)

    print(f"  Mesh directory: {mesh_dir}  ({len(stl_files)} STL files found)")

    # ── Parse URDF ──
    print(f"Parsing URDF: {args.urdf}")
    joint_infos = parse_urdf(args.urdf, joint_names, mesh_dir)

    # ── Compute ellipsoids ──
    print(f"\nComputing MVEE bounding ellipsoids (scale={args.scale}):")
    Q_list, E_list, c_list, semi_list = [], [], [], []

    for idx, info in enumerate(joint_infos):
        stl_path = info.get('stl_path')
        jname = joint_names[idx]

        if stl_path and os.path.exists(stl_path):
            verts, faces = load_stl_vertices_and_faces(stl_path)
            E_raw, c = mvee(verts, tol=1e-5)
            E = E_raw / (args.scale ** 2)

            semi = 1.0 / np.sqrt(np.linalg.eigvalsh(E))
            print(f"  [{JOINT_DISPLAY_NAMES[idx]:20s}] {len(verts):>5} verts | "
                  f"center=[{c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f}] | "
                  f"semi=[{semi[0]:.4f},{semi[1]:.4f},{semi[2]:.4f}]")
        else:
            print(f"\n  ERROR: STL not found for joint '{jname}'")
            print(f"         Expected file: {stl_path}")
            print(f"         Available STL files in {mesh_dir}:")
            for f in sorted(stl_files):
                print(f"           {f}")
            print(f"\n  Check that --mesh_dir points to the correct directory")
            print(f"  and that the URDF mesh filenames match the files on disk.")
            sys.exit(1)

        Q = ellipsoid_to_Q(E, c)
        Q_list.append(Q)
        E_list.append(E)
        c_list.append(c)
        semi_list.append(semi)

    # ── Report ──
    print_report(joint_infos, E_list, c_list, semi_list, args.scale)

    # ── Save ──
    if args.save_npz:
        np.savez(args.save_npz,
                 **{f'Q_{i}': Q for i, Q in enumerate(Q_list)},
                 **{f'E_{i}': E for i, E in enumerate(E_list)},
                 **{f'c_{i}': c for i, c in enumerate(c_list)},
                 joint_names=joint_names,
                 scale=args.scale)
        print(f"Saved matrices: {args.save_npz}")
        print(f"  Load with: data = np.load('{args.save_npz}')")
        print(f"             Q_0 = data['Q_0'], E_0 = data['E_0'], c_0 = data['c_0']")

    # ── Visualize ──
    fig_prefix = args.save_fig

    print("\nGenerating per-link visualization...")
    fig1 = visualize_per_link(
        joint_infos, E_list, c_list, semi_list,
        save_path=f"{fig_prefix}_links.png" if fig_prefix else None
    )

    print("Generating assembly visualization...")
    fig2 = visualize_assembly(
        joint_infos, E_list, c_list, semi_list,
        save_path=f"{fig_prefix}_assembly.png" if fig_prefix else None
    )

    if not args.no_show:
        print("\nShowing interactive plots (close windows to exit)...")
        plt.show()
    else:
        # If not showing, save to default paths if no prefix given
        if not fig_prefix:
            fig1.savefig('ellipsoid_links.png', dpi=180, bbox_inches='tight')
            fig2.savefig('ellipsoid_assembly.png', dpi=180, bbox_inches='tight')
            print("  Saved: ellipsoid_links.png, ellipsoid_assembly.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
