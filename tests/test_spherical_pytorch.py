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


def _load_models(urdf_folder, device) -> Tuple[KinDynComputations, Any]:
    urdf_path = urdf_folder / "spherical_bot.urdf"
    urdf_path.write_text(URDF_SPHERICAL)

    kdc = KinDynComputations(str(urdf_path), device=device, dtype=torch.float64)
    kdc.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    assert kdc.NDoF == 3

    # Convert Adam model to iDynTree model
    idyn_model = to_idyntree_model(kdc.rbdalgos.model)
    idyn = idyntree.KinDynComputations()
    assert idyn.loadRobotModel(idyn_model)
    idyn.setFrameVelocityRepresentation(idyntree.MIXED_REPRESENTATION)

    return kdc, idyn


def test_forward_kinematics_spherical_joint(tmp_path, device):
    kdc, idyn = _load_models(tmp_path, device)

    base = torch.eye(4, device=device, dtype=torch.float64)
    q = torch.tensor([0.2, -0.1, 0.3], device=device, dtype=torch.float64)

    # evaluate the adam forward kinematics
    fk = kdc.forward_kinematics("tip", base, q)
    expected = base.clone()
    expected[:3, :3] = _rpy_to_matrix(q)
    expected[:3, 3] = torch.tensor([0.0, 0.0, 0.1], device=device, dtype=torch.float64)
    assert torch.allclose(fk, expected, atol=1e-6)

    base_np = to_numpy(base)
    q_np = to_numpy(q)
    zero6 = np.zeros(6)
    zero3 = np.zeros(3)
    idyn.setRobotState(base_np, q_np, zero6, zero3, np.array([0.0, 0.0, -9.80665]))
    idyn_fk = idyn.getRelativeTransform("base_link", "tip").asHomogeneousTransform().toNumPy()
    print("Adam FK:\n", to_numpy(fk))
    print("iDynTree FK:\n", idyn_fk)
    assert to_numpy(fk) - idyn_fk == pytest.approx(0.0, abs=1e-6)
    
  
def test_jacobian_spherical_joint(tmp_path, device):
    kdc, idyn = _load_models(tmp_path, device)

    base = torch.eye(4, device=device, dtype=torch.float64)
    q = torch.tensor([0.2, -0.1, 0.3], device=device, dtype=torch.float64)

    jacobian = kdc.jacobian("tip", base, q)
    assert jacobian.shape[-2:] == (6, 6 + kdc.NDoF)
    joint_block = jacobian[..., :, 6:]
    assert joint_block.shape[-2:] == (6, kdc.NDoF)
    assert torch.all(torch.isfinite(joint_block))

    # Compare with iDynTree
    base_np = to_numpy(base)
    q_np = to_numpy(q)
    zero6 = np.zeros(6)
    zero3 = np.zeros(3)
    idyn.setRobotState(base_np, q_np, zero6, zero3, np.array([0.0, 0.0, -9.80665]))
    idyn_jac = idyntree.MatrixDynSize(6, 6 + kdc.NDoF)
    idyn.getFrameFreeFloatingJacobian("tip", idyn_jac)
    idyn_jac = idyn_jac.toNumPy()
    assert to_numpy(jacobian) - idyn_jac == pytest.approx(0.0, abs=1e-6)


def test_jacobian_dot_spherical_joint(tmp_path, device):
    kdc, idyn = _load_models(tmp_path, device)

    base = torch.eye(4, device=device, dtype=torch.float64)
    q = torch.tensor([0.2, -0.1, 0.3], device=device, dtype=torch.float64)
    base_vel = torch.tensor([0.1, -0.2, 0.3, 0.05, -0.07, 0.09], device=device, dtype=torch.float64)
    q_dot = torch.tensor([0.05, -0.02, 0.03], device=device, dtype=torch.float64)

    jacobian_dot = kdc.jacobian_dot("tip", base, q, base_vel, q_dot) @ torch.concatenate([base_vel, q_dot])
    assert torch.all(torch.isfinite(jacobian_dot))

    # Compare with iDynTree
    base_np = to_numpy(base)
    q_np = to_numpy(q)
    base_vel_np = to_numpy(base_vel)
    q_dot_np = to_numpy(q_dot)
    zero6 = np.zeros(6)
    zero3 = np.zeros(3)
    idyn.setRobotState(base_np, q_np, base_vel_np, q_dot_np, np.array([0.0, 0.0, -9.80665]))
    idyn_bias_acc = idyn.getFrameBiasAcc("tip").toNumPy()
    assert to_numpy(jacobian_dot) - idyn_bias_acc == pytest.approx(0.0, abs=1e-6)


def test_mass_matrix_spherical_joint(tmp_path, device):
    kdc, idyn = _load_models(tmp_path, device)

    base = torch.eye(4, device=device, dtype=torch.float64)
    q = torch.tensor([0.2, -0.1, 0.3], device=device, dtype=torch.float64)

    mass_matrix = kdc.mass_matrix(base, q)
    assert mass_matrix.shape[-2:] == (6 + kdc.NDoF, 6 + kdc.NDoF)
    assert torch.allclose(mass_matrix, mass_matrix.transpose(-1, -2), atol=1e-9)

    # Compare with iDynTree
    base_np = to_numpy(base)
    q_np = to_numpy(q)
    zero6 = np.zeros(6)
    zero3 = np.zeros(3)
    idyn.setRobotState(base_np, q_np, zero6, zero3, np.array([0.0, 0.0, -9.80665]))
    idyn_mass = idyntree.MatrixDynSize(6 + kdc.NDoF, 6 + kdc.NDoF)
    idyn.getFreeFloatingMassMatrix(idyn_mass)
    idyn_mass = idyn_mass.toNumPy()
    assert to_numpy(mass_matrix) - idyn_mass == pytest.approx(0.0, abs=1e-6)


def test_gravity_term_spherical_joint(tmp_path, device):
    kdc, idyn = _load_models(tmp_path, device)

    base = torch.eye(4, device=device, dtype=torch.float64)
    q = torch.tensor([0.2, -0.1, 0.3], device=device, dtype=torch.float64)

    gravity_term = kdc.gravity_term(base, q)
    print("kdc gravity term:\n", gravity_term)
    assert gravity_term.shape[-1] == 6 + kdc.NDoF
    assert torch.all(torch.isfinite(gravity_term))

    # Compare with iDynTree
    base_np = to_numpy(base)
    q_np = to_numpy(q)
    zero6 = np.zeros(6)
    zero3 = np.zeros(3)
    idyn.setRobotState(base_np, q_np, zero6, zero3, np.array([0.0, 0.0, -9.80665]))
    idyn_gravity = idyntree.FreeFloatingGeneralizedTorques(idyn.model())
    print("idyn_gravity:", idyn_gravity)
    idyn.generalizedGravityForces(idyn_gravity)
    idyn_gravity_base_wrench = idyn_gravity.baseWrench().toNumPy()
    idyn_gravity_joint_torques = idyn_gravity.jointTorques().toNumPy()
    idyn_gravity_term = np.concatenate((idyn_gravity_base_wrench, idyn_gravity_joint_torques))
    print("idyn_gravity term:\n", idyn_gravity_term)
    assert to_numpy(gravity_term) - idyn_gravity_term == pytest.approx(0.0, abs=1e-6)