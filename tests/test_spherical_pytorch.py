import adam
import idyntree.bindings as idyntree
import numpy as np
import pytest
import torch
from typing import Any, Tuple


from adam.pytorch import KinDynComputations
from adam.model.conversions.idyntree import to_idyntree_model
from conftest import to_numpy


URDF_SPHERICAL = """<?xml version="1.0"?>
<robot name="spherical_bot">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>
  <link name="tip">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="ball" type="spherical">
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="tip"/>
  </joint>
</robot>
"""


def _rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    
    # R = Rx(roll) * Ry(pitch) * Rz(yaw)
    row0 = torch.stack([cp * cy, -cp * sy, sp])
    row1 = torch.stack([cr * sy + sr * sp * cy, cr * cy - sr * sp * sy, -sr * cp])
    row2 = torch.stack([sr * sy - cr * sp * cy, sr * cy + cr * sp * sy, cr * cp])
    return torch.stack([row0, row1, row2])


def _rpy_to_quaternion(rpy: np.ndarray) -> np.ndarray:
    """Convert ADAM's Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).
    
    ADAM uses XYZ convention: R = Rx(roll) * Ry(pitch) * Rz(yaw)
    We compute the rotation matrix first, then convert to quaternion.
    """
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # R = Rx * Ry * Rz (ADAM's XYZ convention)
    R = np.array([
        [cp * cy, -cp * sy, sp],
        [cr * sy + sr * sp * cy, cr * cy - sr * sp * sy, -sr * cp],
        [sr * sy - cr * sp * cy, sr * cy + cr * sp * sy, cr * cp]
    ])
    
    # Convert rotation matrix to quaternion using iDynTree
    rot = idyntree.Rotation()
    for i in range(3):
        for j in range(3):
            rot.setVal(i, j, float(R[i, j]))
    q = rot.asQuaternion()
    return np.array([q[0], q[1], q[2], q[3]])


def _load_models(urdf_folder, device) -> Tuple[KinDynComputations, Any]:
    urdf_path = urdf_folder / "spherical_bot.urdf"
    urdf_path.write_text(URDF_SPHERICAL)

    kdc = KinDynComputations(str(urdf_path), device=device, dtype=torch.float64)
    kdc.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    # NDoF is 3 (velocity DOFs), NPosDof is 4 (quaternion position coords)
    assert kdc.NDoF == 3
    assert kdc.NPosDof == 4

    # Convert Adam model to iDynTree model
    idyn_model = to_idyntree_model(kdc.rbdalgos.model)
    idyn = idyntree.KinDynComputations()
    assert idyn.loadRobotModel(idyn_model)
    idyn.setFrameVelocityRepresentation(idyntree.MIXED_REPRESENTATION)

    return kdc, idyn


def test_forward_kinematics_spherical_joint(tmp_path, device):
    kdc, idyn = _load_models(tmp_path, device)

    base = torch.eye(4, device=device, dtype=torch.float64)
    # Use Euler angles to create quaternion for testing
    rpy = np.array([0.2, -0.1, 0.3])
    q_quat_np = _rpy_to_quaternion(rpy)
    q = torch.tensor(q_quat_np, device=device, dtype=torch.float64)

    # evaluate the adam forward kinematics
    fk = kdc.forward_kinematics("tip", base, q)
    expected = base.clone()
    expected[:3, :3] = _rpy_to_matrix(torch.tensor(rpy, device=device, dtype=torch.float64))
    expected[:3, 3] = torch.tensor([0.0, 0.0, 0.1], device=device, dtype=torch.float64)
    assert torch.allclose(fk, expected, atol=1e-6)

    base_np = to_numpy(base)
    zero6 = np.zeros(6)
    zero3 = np.zeros(3)
    idyn.setRobotState(base_np, q_quat_np, zero6, zero3, np.array([0.0, 0.0, -9.80665]))
    idyn_fk = idyn.getRelativeTransform("base_link", "tip").asHomogeneousTransform().toNumPy()
    print("Adam FK:\n", to_numpy(fk))
    print("iDynTree FK:\n", idyn_fk)
    assert to_numpy(fk) - idyn_fk == pytest.approx(0.0, abs=1e-6)
    
  
def test_jacobian_spherical_joint(tmp_path, device):
    kdc, idyn = _load_models(tmp_path, device)

    base = torch.eye(4, device=device, dtype=torch.float64)
    # Use Euler angles to create quaternion for testing
    rpy = np.array([0.2, -0.1, 0.3])
    q_quat_np = _rpy_to_quaternion(rpy)
    q = torch.tensor(q_quat_np, device=device, dtype=torch.float64)

    jacobian = kdc.jacobian("tip", base, q)
    # Jacobian is 6 x (6 + NDoF) where NDoF=3 (velocity DOFs)
    assert jacobian.shape[-2:] == (6, 6 + kdc.NDoF)
    joint_block = jacobian[..., :, 6:]
    assert joint_block.shape[-2:] == (6, kdc.NDoF)
    assert torch.all(torch.isfinite(joint_block))

    # Compare with iDynTree - now both use same quaternion convention
    base_np = to_numpy(base)
    zero6 = np.zeros(6)
    zero3 = np.zeros(3)
    idyn.setRobotState(
        base_np, q_quat_np, zero6, zero3, np.array([0.0, 0.0, -9.80665])
    )
    idyn_jac = idyntree.MatrixDynSize(6, 6 + 3)  # 6 rows, 6+3 columns
    idyn.getFrameFreeFloatingJacobian("tip", idyn_jac)
    idyn_jac_np = idyn_jac.toNumPy()
    print("Adam Jacobian:\n", to_numpy(jacobian))
    print("iDynTree Jacobian:\n", idyn_jac_np)
    assert to_numpy(jacobian) - idyn_jac_np == pytest.approx(0.0, abs=1e-6)


def test_jacobian_dot_spherical_joint(tmp_path, device):
    kdc, _ = _load_models(tmp_path, device)

    base = torch.eye(4, device=device, dtype=torch.float64)
    # Use Euler angles to create quaternion for testing
    rpy = np.array([0.2, -0.1, 0.3])
    q_quat_np = _rpy_to_quaternion(rpy)
    q = torch.tensor(q_quat_np, device=device, dtype=torch.float64)
    base_vel = torch.tensor(
        [0.1, -0.2, 0.3, 0.05, -0.07, 0.09], device=device, dtype=torch.float64
    )
    # Angular velocity (3 DOFs)
    q_dot = torch.tensor([0.05, -0.02, 0.03], device=device, dtype=torch.float64)

    jacobian_dot = kdc.jacobian_dot("tip", base, q, base_vel, q_dot)
    jacobian_dot_times_v = jacobian_dot @ torch.concatenate([base_vel, q_dot])
    assert torch.all(torch.isfinite(jacobian_dot_times_v))


def test_mass_matrix_spherical_joint(tmp_path, device):
    kdc, idyn = _load_models(tmp_path, device)

    base = torch.eye(4, device=device, dtype=torch.float64)
    # Use Euler angles to create quaternion for testing
    rpy = np.array([0.2, -0.1, 0.3])
    q_quat_np = _rpy_to_quaternion(rpy)
    q = torch.tensor(q_quat_np, device=device, dtype=torch.float64)

    mass_matrix = kdc.mass_matrix(base, q)
    # Mass matrix is (6 + NDoF) x (6 + NDoF) where NDoF=3
    assert mass_matrix.shape[-2:] == (6 + kdc.NDoF, 6 + kdc.NDoF)
    assert torch.allclose(
        mass_matrix, mass_matrix.transpose(-1, -2), atol=1e-9
    )

    # Compare with iDynTree - now both use same quaternion convention
    base_np = to_numpy(base)
    zero6 = np.zeros(6)
    zero3 = np.zeros(3)
    idyn.setRobotState(
        base_np, q_quat_np, zero6, zero3, np.array([0.0, 0.0, -9.80665])
    )
    idyn_mass = idyntree.MatrixDynSize()
    idyn.getFreeFloatingMassMatrix(idyn_mass)
    idyn_mass_np = idyn_mass.toNumPy()
    print("Adam Mass Matrix:\n", to_numpy(mass_matrix))
    print("iDynTree Mass Matrix:\n", idyn_mass_np)
    assert to_numpy(mass_matrix) - idyn_mass_np == pytest.approx(0.0, abs=1e-6)


def test_gravity_term_spherical_joint(tmp_path, device):
    kdc, idyn = _load_models(tmp_path, device)

    base = torch.eye(4, device=device, dtype=torch.float64)
    # Use Euler angles to create quaternion for testing
    rpy = np.array([0.2, -0.1, 0.3])
    q_quat_np = _rpy_to_quaternion(rpy)
    q = torch.tensor(q_quat_np, device=device, dtype=torch.float64)

    gravity_term = kdc.gravity_term(base, q)
    print("kdc gravity term:\n", gravity_term)
    assert gravity_term.shape[-1] == 6 + kdc.NDoF
    assert torch.all(torch.isfinite(gravity_term))

    # Compare with iDynTree
    base_np = to_numpy(base)
    zero6 = np.zeros(6)
    zero3 = np.zeros(3)
    idyn.setRobotState(
        base_np, q_quat_np, zero6, zero3, np.array([0.0, 0.0, -9.80665])
    )
    idyn_gravity = idyntree.FreeFloatingGeneralizedTorques(idyn.model())
    print("idyn_gravity:", idyn_gravity)
    idyn.generalizedGravityForces(idyn_gravity)
    idyn_gravity_base_wrench = idyn_gravity.baseWrench().toNumPy()
    idyn_gravity_joint_torques = idyn_gravity.jointTorques().toNumPy()
    idyn_gravity_term = np.concatenate(
        (idyn_gravity_base_wrench, idyn_gravity_joint_torques)
    )
    print("idyn_gravity term:\n", idyn_gravity_term)
    assert (
        to_numpy(gravity_term) - idyn_gravity_term == pytest.approx(0.0, abs=1e-6)
    )
