import adam
import idyntree.bindings as idyntree
import numpy as np
import pytest
import torch
from typing import Any, Tuple


from adam.pytorch import KinDynComputations
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
    row0 = torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr])
    row1 = torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr])
    row2 = torch.stack([-sp, cp * sr, cp * cr])
    return torch.stack([row0, row1, row2])


def _spherical_as_rpy_chain_urdf() -> str:
    """Return a URDF string where the spherical joint is expanded into 3 revolute joints (roll-pitch-yaw)."""
    return """<?xml version="1.0"?>
<robot name="spherical_bot_rpy">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>
  <link name="roll_link"/>
  <link name="pitch_link"/>
  <link name="tip">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="ball_roll" type="revolute">
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="roll_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="0" velocity="0"/>
  </joint>
  <joint name="ball_pitch" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="roll_link"/>
    <child link="pitch_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="0" velocity="0"/>
  </joint>
  <joint name="ball_yaw" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="pitch_link"/>
    <child link="tip"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="0" velocity="0"/>
  </joint>
</robot>
"""

def _load_models(urdf_folder, device) -> Tuple[KinDynComputations, Any]:
    urdf_path = urdf_folder / "spherical_bot.urdf"
    urdf_path.write_text(URDF_SPHERICAL)

    idyn_urdf_path = urdf_folder / "spherical_bot_rpy.urdf"
    idyn_urdf_path.write_text(_spherical_as_rpy_chain_urdf())
    idyn_joint_names = ["ball_roll", "ball_pitch", "ball_yaw"]

    kdc = KinDynComputations(str(urdf_path), device=device, dtype=torch.float64)
    kdc.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    assert kdc.NDoF == 3

    # Compare against iDynTree results using an equivalent RPY joint chain
    idyn_loader = idyntree.ModelLoader()
    assert idyn_loader.loadReducedModelFromFile(str(idyn_urdf_path), idyn_joint_names)
    idyn = idyntree.KinDynComputations()
    assert idyn.loadRobotModel(idyn_loader.model())
    idyn.setFrameVelocityRepresentation(idyntree.MIXED_REPRESENTATION)

    # Compare against iDynTree results using an equivalent RPY joint chain
    idyn_loader = idyntree.ModelLoader()
    assert idyn_loader.loadReducedModelFromFile(str(idyn_urdf_path), idyn_joint_names)
    idyn = idyntree.KinDynComputations()
    assert idyn.loadRobotModel(idyn_loader.model())
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
    print("type of idyn_fk:", type(idyn_fk))
    print("Adam FK:\n", to_numpy(fk))
    print("iDynTree FK:\n", idyn_fk)
    assert to_numpy(fk) - idyn_fk == pytest.approx(0.0, abs=1e-6)

    
# def test_jacobian_spherical_joint(tmp_path, device):
#     kdc, idyn = _load_models(tmp_path, device)

#     base = torch.eye(4, device=device, dtype=torch.float64)
#     q = torch.tensor([0.2, -0.1, 0.3], device=device, dtype=torch.float64)

#     jacobian = kdc.jacobian("tip", base, q)
#     assert jacobian.shape[-2:] == (6, 6 + kdc.NDoF)
#     joint_block = jacobian[..., :, 6:]
#     assert joint_block.shape[-2:] == (6, kdc.NDoF)
#     assert torch.all(torch.isfinite(joint_block))

# def test_jacobian_dot_spherical_joint(tmp_path, device):
#     kdc, idyn_mass = _load_models(tmp_path, device)

#     base = torch.eye(4, device=device, dtype=torch.float64)
#     q = torch.tensor([0.2, -0.1, 0.3], device=device, dtype=torch.float64)
#     base_vel = torch.tensor([0.1, -0.2, 0.3, 0.05, -0.07, 0.09], device=device, dtype=torch.float64)
#     q_dot = torch.tensor([0.05, -0.02, 0.03], device=device, dtype=torch.float64)

#     jacobian_dot = kdc.jacobian_dot("tip", base, q, base_vel, q_dot) @ torch.concatenate([base_vel, q_dot])
#     assert torch.all(torch.isfinite(jacobian_dot))

# def test_mass_matrix_spherical_joint(tmp_path, device):
#     kdc, idyn_mass = _load_models(tmp_path, device)

#     base = torch.eye(4, device=device, dtype=torch.float64)
#     q = torch.tensor([0.2, -0.1, 0.3], device=device, dtype=torch.float64)

#     mass_matrix = kdc.mass_matrix(base, q)
#     assert mass_matrix.shape[-2:] == (6 + kdc.NDoF, 6 + kdc.NDoF)
#     assert torch.allclose(mass_matrix, mass_matrix.transpose(-1, -2), atol=1e-9)

# def test_spherical_joint_support(tmp_path, device):
#     kdc, idyn_mass = _load_models(tmp_path, device)

#     base = torch.eye(4, device=device, dtype=torch.float64)
#     q = torch.tensor([0.2, -0.1, 0.3], device=device, dtype=torch.float64)

#     fk = kdc.forward_kinematics("tip", base, q)
#     expected = base.clone()
#     expected[:3, :3] = _rpy_to_matrix(q)
#     expected[:3, 3] = torch.tensor([0.0, 0.0, 0.1], device=device, dtype=torch.float64)

#     assert torch.allclose(fk, expected, atol=1e-6)

#     mass_matrix = kdc.mass_matrix(base, q)
#     assert mass_matrix.shape[-2:] == (6 + kdc.NDoF, 6 + kdc.NDoF)
#     assert torch.allclose(mass_matrix, mass_matrix.transpose(-1, -2), atol=1e-9)

#     jacobian = kdc.jacobian("tip", base, q)
#     assert jacobian.shape[-2:] == (6, 6 + kdc.NDoF)
#     joint_block = jacobian[..., :, 6:]
#     assert joint_block.shape[-2:] == (6, kdc.NDoF)
#     assert torch.all(torch.isfinite(joint_block))

#     base_vel = torch.tensor([0.1, -0.2, 0.3, 0.05, -0.07, 0.09], device=device, dtype=torch.float64)
#     q_dot = torch.tensor([0.05, -0.02, 0.03], device=device, dtype=torch.float64)
#     jacobian_dot = kdc.jacobian_dot("tip", base, q, base_vel, q_dot) @ torch.concatenate([base_vel, q_dot])

#     # Compare against iDynTree results using an equivalent RPY joint chain
#     idyn_loader = idyntree.ModelLoader()
#     assert idyn_loader.loadReducedModelFromFile(str(idyn_urdf_path), idyn_joint_names)
#     idyn = idyntree.KinDynComputations()
#     assert idyn.loadRobotModel(idyn_loader.model())
#     idyn.setFrameVelocityRepresentation(idyntree.MIXED_REPRESENTATION)

#     base_np = to_numpy(base)
#     q_np = to_numpy(q)
#     zero6 = np.zeros(6)
#     zero3 = np.zeros(3)
#     idyn.setRobotState(base_np, q_np, zero6, zero3, np.array([0.0, 0.0, -9.80665]))

#     idyn.getFreeFloatingMassMatrix(idyn_mass)
#     idyn_mass = idyn_mass.toNumPy()
#     assert to_numpy(mass_matrix) - idyn_mass == pytest.approx(0.0, abs=5e-3)

#     idyn_jac = idyntree.MatrixDynSize(6, 6 + len(idyn_joint_names))
#     idyn.getFrameFreeFloatingJacobian("tip", idyn_jac)
#     idyn_jac = idyn_jac.toNumPy()
#     assert to_numpy(jacobian) - idyn_jac == pytest.approx(0.0, abs=1e-6)

#     idyn.setRobotState(base_np, q_np, to_numpy(base_vel), to_numpy(q_dot), np.array([0.0, 0.0, -9.80665]))
#     idyn_bias_acc = idyn.getFrameBiasAcc("tip").toNumPy()
#     assert to_numpy(jacobian_dot) - idyn_bias_acc == pytest.approx(0.0, abs=1e-5)
