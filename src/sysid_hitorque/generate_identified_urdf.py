#!/usr/bin/env python3
"""
Generate a new URDF with identified inertial parameters.

Converts from the identification parameterization:
    ψ_i = [m, hx, hy, hz, Ixx_O, Iyy_O, Izz_O, Ixy_O, Iyz_O, Ixz_O]
    (where I_O is inertia about the joint origin frame, h = m*c)

Back to URDF format:
    <origin xyz="cx cy cz"/>
    <mass value="m"/>
    <inertia ixx iyy izz ixy ixz iyz/>  (at COM frame)

Usage:
    python3 generate_identified_urdf.py \
        --urdf_in hi.urdf \
        --params identified_params.npz \
        --urdf_out hi_identified.urdf
    
    e.g. python3 generate_identified_urdf.py --urdf_in generated_urdf/hi.urdf --params identified_results/identified_params.npz
    
"""

import argparse
import numpy as np
import xml.etree.ElementTree as ET
import copy
import re

ARM_JOINTS = [
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint",
    "r_arm_yaw_joint", "r_arm_roll_joint", "r_wrist_yaw_joint",
]

# Map from joint name -> link name (child link of that joint)
JOINT_TO_LINK = {
    "r_shoulder_pitch_joint": "r_shoulder_pitch_link",
    "r_shoulder_roll_joint":  "r_shoulder_roll_link",
    "r_arm_yaw_joint":        "r_arm_yaw_link",
    "r_arm_roll_joint":       "r_arm_roll_link",
    "r_wrist_yaw_joint":      "r_wrist_yaw_link",
}

NJ = 5


def psi_to_urdf_inertial(psi_i):
    """
    Convert identified parameters (joint-origin frame) to URDF format (COM frame).

    Input:  ψ_i = [m, hx, hy, hz, Ixx_O, Iyy_O, Izz_O, Ixy_O, Iyz_O, Ixz_O]
    Output: (m, [cx, cy, cz], I_com_dict)

    where I_com is the inertia tensor at the COM frame, computed via
    reverse parallel axis theorem:
        I_com = I_O - m * (c·c * I_3 - c ⊗ c)
    """
    m  = psi_i[0]
    hx, hy, hz = psi_i[1], psi_i[2], psi_i[3]
    Ixx_O, Iyy_O, Izz_O = psi_i[4], psi_i[5], psi_i[6]
    Ixy_O, Iyz_O, Ixz_O = psi_i[7], psi_i[8], psi_i[9]

    # COM position
    if m > 1e-10:
        cx, cy, cz = hx / m, hy / m, hz / m
    else:
        cx, cy, cz = 0.0, 0.0, 0.0

    c = np.array([cx, cy, cz])

    # Inertia at joint origin
    I_O = np.array([
        [Ixx_O, Ixy_O, Ixz_O],
        [Ixy_O, Iyy_O, Iyz_O],
        [Ixz_O, Iyz_O, Izz_O],
    ])

    # Reverse parallel axis theorem: I_com = I_O - m*(c·c * I - c⊗c)
    I_com = I_O - m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))

    return m, (cx, cy, cz), {
        'ixx': I_com[0, 0],
        'iyy': I_com[1, 1],
        'izz': I_com[2, 2],
        'ixy': I_com[0, 1],
        'ixz': I_com[0, 2],
        'iyz': I_com[1, 2],
    }


def update_urdf(urdf_in_path, psi_opt, urdf_out_path, f_opt=None):
    """
    Parse the input URDF, update inertial parameters for arm links,
    and write the new URDF.
    """
    # Parse preserving comments and formatting as much as possible
    tree = ET.parse(urdf_in_path)
    root = tree.getroot()

    for idx, joint_name in enumerate(ARM_JOINTS):
        link_name = JOINT_TO_LINK[joint_name]
        psi_i = psi_opt[idx * 10:(idx + 1) * 10]

        m, (cx, cy, cz), I_com = psi_to_urdf_inertial(psi_i)

        # Find the link element
        link_elem = None
        for link in root.iter('link'):
            if link.get('name') == link_name:
                link_elem = link
                break

        if link_elem is None:
            print(f"WARNING: Link '{link_name}' not found in URDF!")
            continue

        inertial = link_elem.find('inertial')
        if inertial is None:
            print(f"WARNING: No <inertial> in link '{link_name}'!")
            continue

        # Update origin (COM position)
        origin = inertial.find('origin')
        if origin is not None:
            origin.set('xyz', f"{cx:.8g} {cy:.8g} {cz:.8g}")

        # Update mass
        mass_elem = inertial.find('mass')
        if mass_elem is not None:
            mass_elem.set('value', f"{m:.8g}")

        # Update inertia
        inertia_elem = inertial.find('inertia')
        if inertia_elem is not None:
            inertia_elem.set('ixx', f"{I_com['ixx']:.8g}")
            inertia_elem.set('iyy', f"{I_com['iyy']:.8g}")
            inertia_elem.set('izz', f"{I_com['izz']:.8g}")
            inertia_elem.set('ixy', f"{I_com['ixy']:.8g}")
            inertia_elem.set('ixz', f"{I_com['ixz']:.8g}")
            inertia_elem.set('iyz', f"{I_com['iyz']:.8g}")

        print(f"Updated {link_name}:")
        print(f"  mass = {m:.6f}")
        print(f"  COM  = [{cx:.6f}, {cy:.6f}, {cz:.6f}]")
        print(f"  I_com: ixx={I_com['ixx']:.6e}  iyy={I_com['iyy']:.6e}  izz={I_com['izz']:.6e}")
        print(f"         ixy={I_com['ixy']:.6e}  ixz={I_com['ixz']:.6e}  iyz={I_com['iyz']:.6e}")

        # Sanity check: eigenvalues of I_com
        I_mat = np.array([
            [I_com['ixx'], I_com['ixy'], I_com['ixz']],
            [I_com['ixy'], I_com['iyy'], I_com['iyz']],
            [I_com['ixz'], I_com['iyz'], I_com['izz']],
        ])
        eigs = np.linalg.eigvalsh(I_mat)
        status = "OK" if np.all(eigs > -1e-6) else "WARN: negative eigenvalues!"
        print(f"  I_com eigenvalues: [{eigs[0]:.4e}, {eigs[1]:.4e}, {eigs[2]:.4e}] {status}")

    # Update joint friction parameters
    for idx, joint_name in enumerate(ARM_JOINTS):
        joint_elem = None
        for joint in root.iter('joint'):
            if joint.get('name') == joint_name:
                joint_elem = joint
                break
        if joint_elem is None:
            continue

        dynamics = joint_elem.find('dynamics')

        if f_opt is not None:
            fv = f_opt[idx]       # viscous
            fc = f_opt[NJ + idx]  # Coulomb
            if dynamics is None:
                dynamics = ET.SubElement(joint_elem, 'dynamics')
            dynamics.set('damping', f"{fv:.8g}")
            dynamics.set('friction', f"{fc:.8g}")
            print(f"Updated {joint_name}: damping={fv:.6f}, friction={fc:.6f}")
        else:
            # Keep whatever is already in the URDF
            if dynamics is not None:
                d = dynamics.get('damping', '0')
                f = dynamics.get('friction', '0')
                print(f"Kept {joint_name}: damping={d}, friction={f}")

    # Write output URDF
    if hasattr(ET, 'indent'):
        ET.indent(tree, space='\t', level=0)
        
    tree.write(urdf_out_path, encoding='utf-8', xml_declaration=True)
    print(f"\nSaved identified URDF: {urdf_out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate URDF with identified inertial parameters")
    parser.add_argument("--urdf_in", required=True,
                        help="Path to original URDF")
    parser.add_argument("--params", required=True,
                        help="Path to identified_params.npz from solve_sysid.py")
    parser.add_argument("--urdf_out", default="generated_urdf/hi_identified.urdf",
                        help="Output URDF path (default: hi_identified.urdf)")
    parser.add_argument("--use_ls", action="store_true",
                        help="Use plain least-squares params instead of regularized")
    args = parser.parse_args()

    # Load identified parameters
    data = np.load(args.params)
    psi_opt = data['psi_opt']
    f_opt = data['f_opt'] if 'f_opt' in data.files else None
    print(f"Using regularized parameters (λ={data['lambda_reg']})")

    psi0 = data['psi0']

    print(f"Loaded parameters from {args.params}\n")

    # Print summary of changes
    print("=" * 60)
    print("Converting identified parameters to URDF format")
    print("=" * 60)

    update_urdf(args.urdf_in, psi_opt, args.urdf_out, f_opt=f_opt)

    # Also show what the nominal URDF params would be for comparison
    print("\n" + "=" * 60)
    print("For reference - nominal URDF parameters (from npz):")
    print("=" * 60)
    for idx, jname in enumerate(ARM_JOINTS):
        m0, (cx0, cy0, cz0), I0 = psi_to_urdf_inertial(psi0[idx*10:(idx+1)*10])
        print(f"  {JOINT_TO_LINK[jname]}: m={m0:.4f}  COM=[{cx0:.6f},{cy0:.6f},{cz0:.6f}]")


if __name__ == "__main__":
    main()
