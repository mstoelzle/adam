import dataclasses
import logging
import os
from itertools import product

import icub_models
import idyntree.bindings as idyntree
import numpy as np
import pytest
import requests

from adam import Representations
from adam.model.conversions.idyntree import to_idyntree_model
from adam.numpy import KinDynComputations as KinDynComputationsNumpy
from scipy.spatial.transform import Rotation


def is_spherical_robot(robot_name: str) -> bool:
    """Check if a robot has spherical joints."""
    return robot_name == "SmplHumanoid"


def generate_random_quaternion() -> np.ndarray:
    """Generate a random unit quaternion (w, x, y, z)."""
    # Generate random rotation using Rotation class
    rot = Rotation.random()
    quat = rot.as_quat()  # Returns (x, y, z, w)
    # Convert to (w, x, y, z) format used by ADAM/iDynTree
    return np.array([quat[3], quat[0], quat[1], quat[2]])


@dataclasses.dataclass
class State:
    H: np.ndarray
    joints_pos: np.ndarray
    base_vel: np.ndarray
    joints_vel: np.ndarray
    gravity: np.ndarray


@dataclasses.dataclass
class IDynFunctionValues:
    mass_matrix: np.ndarray
    centroidal_momentum_matrix: np.ndarray
    CoM_position: np.ndarray
    CoM_jacobian: np.ndarray
    total_mass: float
    jacobian: np.ndarray
    jacobian_non_actuated: np.ndarray
    jacobian_dot_nu: np.ndarray
    relative_jacobian: np.ndarray
    forward_kinematics: np.ndarray
    forward_kinematics_non_actuated: np.ndarray
    bias_force: np.ndarray
    coriolis_term: np.ndarray
    gravity_term: np.ndarray


@dataclasses.dataclass
class RobotCfg:
    robot_name: str
    velocity_representation: Representations
    model_path: str
    joints_name_list: list
    n_dof: int
    kin_dyn: idyntree.KinDynComputations
    idyn_function_values: IDynFunctionValues
    frame: str
    frame_non_actuated: str


VELOCITY_REPRESENTATIONS = [
    Representations.MIXED_REPRESENTATION,
    Representations.BODY_FIXED_REPRESENTATION,
]

# Robots with standard revolute/prismatic joints (fully tested)
ROBOTS = [
    "iCubGenova04",
    "StickBot",
]

# Robots with spherical joints (partial support, some tests may fail)
ROBOTS_WITH_SPHERICAL = [
    "iCubGenova04",
    "StickBot",
    "SmplHumanoid",
]


def get_robot_model_path(robot_name: str) -> str:
    if robot_name == "StickBot":
        model_path = "stickbot.urdf"
        if not os.path.exists(model_path):
            url = "https://raw.githubusercontent.com/icub-tech-iit/ergocub-gazebo-simulations/master/models/stickBot/model.urdf"
            response = requests.get(url)
            with open(model_path, "wb") as file:
                file.write(response.content)
    elif robot_name == "SmplHumanoid":
        model_path = os.path.join(os.path.dirname(__file__), "..", "assets", "smpl_humanoid.urdf")
    else:
        model_path = str(icub_models.get_model_file(robot_name))
    return model_path


def get_robot_joints_list(robot_name: str) -> list:
    """Get the list of joints for a given robot."""
    if robot_name == "SmplHumanoid":
        return [
            "Pelvis_L_Hip",
            "Pelvis_R_Hip",
            "Pelvis_Torso",
            "smpl_humanoid_Torso_Spine",
            "smpl_humanoid_Spine_Chest",
            "smpl_humanoid_Chest_L_Thorax",
            "smpl_humanoid_Chest_R_Thorax",
            "smpl_humanoid_Chest_Neck",
            "smpl_humanoid_Neck_Head",
            "smpl_humanoid_L_Hip_L_Knee",
            "smpl_humanoid_L_Knee_L_Ankle",
            "smpl_humanoid_L_Ankle_L_Toe",
            "smpl_humanoid_R_Hip_R_Knee",
            "smpl_humanoid_R_Knee_R_Ankle",
            "smpl_humanoid_R_Ankle_R_Toe",
            "smpl_humanoid_L_Thorax_L_Shoulder",
            "smpl_humanoid_L_Shoulder_L_Elbow",
            "smpl_humanoid_L_Elbow_L_Wrist",
            "smpl_humanoid_L_Wrist_L_Hand",
            "smpl_humanoid_R_Thorax_R_Shoulder",
            "smpl_humanoid_R_Shoulder_R_Elbow",
            "smpl_humanoid_R_Elbow_R_Wrist",
            "smpl_humanoid_R_Wrist_R_Hand",
        ]
    else:
        return [
            "torso_pitch",
            "torso_roll",
            "torso_yaw",
            "l_shoulder_pitch",
            "l_shoulder_roll",
            "l_shoulder_yaw",
            "l_elbow",
            "r_shoulder_pitch",
            "r_shoulder_roll",
            "r_shoulder_yaw",
            "r_elbow",
            "l_hip_pitch",
            "l_hip_roll",
            "l_hip_yaw",
            "l_knee",
            "l_ankle_pitch",
            "l_ankle_roll",
            "r_hip_pitch",
            "r_hip_roll",
            "r_hip_yaw",
            "r_knee",
            "r_ankle_pitch",
            "r_ankle_roll",
        ]


def get_robot_frames(robot_name: str) -> dict:
    """Get the frame names for jacobian/FK tests for a given robot."""
    if robot_name == "SmplHumanoid":
        return {
            "frame": "smpl_humanoid_L_Ankle",
            "frame_non_actuated": "smpl_humanoid_Head",
        }
    else:
        return {
            "frame": "l_sole",
            "frame_non_actuated": "head",
        }


TEST_CONFIGURATIONS = list(product(VELOCITY_REPRESENTATIONS, ROBOTS))
TEST_CONFIGURATIONS_WITH_SPHERICAL = list(
    product(VELOCITY_REPRESENTATIONS, ROBOTS_WITH_SPHERICAL)
)


@pytest.fixture(scope="module", params=TEST_CONFIGURATIONS, ids=str)
def tests_setup(request) -> RobotCfg | State:
    """Fixture for robots with standard revolute/prismatic joints only."""
    velocity_representation, robot_name = request.param

    np.random.seed(42)

    model_path = get_robot_model_path(robot_name)
    joints_name_list = get_robot_joints_list(robot_name)
    frames = get_robot_frames(robot_name)

    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Showing the robot tree.")

    robot_iDyn = idyntree.ModelLoader()
    robot_iDyn.loadReducedModelFromFile(model_path, joints_name_list)

    kin_dyn = idyntree.KinDynComputations()
    kin_dyn.loadRobotModel(robot_iDyn.model())

    n_dof = len(joints_name_list)

    # joints quantities
    joints_val = (np.random.rand(n_dof) - 0.5) * 5
    joints_dot_val = (np.random.rand(n_dof) - 0.5) * 5

    if velocity_representation == Representations.BODY_FIXED_REPRESENTATION:
        idyn_representation = idyntree.BODY_FIXED_REPRESENTATION
    elif velocity_representation == Representations.MIXED_REPRESENTATION:
        idyn_representation = idyntree.MIXED_REPRESENTATION
    else:
        raise ValueError(
            f"Unknown velocity representation: {velocity_representation}"
        )
    kin_dyn.setFrameVelocityRepresentation(idyn_representation)

    # base quantities
    xyz = (np.random.rand(3) - 0.5) * 5
    rpy = (np.random.rand(3) - 0.5) * 5
    base_vel = (np.random.rand(6) - 0.5) * 5

    g = np.array([0, 0, -9.80665])
    R_b = Rotation.from_euler("xyz", rpy).as_matrix()
    H_b = np.eye(4)
    H_b[:3, :3] = R_b
    H_b[:3, 3] = xyz

    state = State(
        H=H_b,
        joints_pos=joints_val,
        base_vel=base_vel,
        joints_vel=joints_dot_val,
        gravity=g,
    )

    idyn_function_values = compute_idyntree_values(kin_dyn, state, frames)

    robot_cfg = RobotCfg(
        robot_name=robot_name,
        velocity_representation=velocity_representation,
        model_path=model_path,
        joints_name_list=joints_name_list,
        n_dof=n_dof,
        kin_dyn=kin_dyn,
        idyn_function_values=idyn_function_values,
        frame=frames["frame"],
        frame_non_actuated=frames["frame_non_actuated"],
    )

    yield robot_cfg, state


def compute_idyntree_values(
    kin_dyn: idyntree.KinDynComputations, state: State, frames: dict = None
) -> IDynFunctionValues:
    # Use default frame names if not provided (backward compatible with original)
    if frames is None:
        frames = {"frame": "l_sole", "frame_non_actuated": "head"}
    kin_dyn.setRobotState(
        state.H, state.joints_pos, state.base_vel, state.joints_vel, state.gravity
    )

    # mass matrix
    idyn_mass_matrix = idyntree.MatrixDynSize()
    kin_dyn.getFreeFloatingMassMatrix(idyn_mass_matrix)
    idyn_mass_matrix = idyn_mass_matrix.toNumPy()

    # centroidal momentum matrix
    idyn_cmm = idyntree.MatrixDynSize()
    kin_dyn.getCentroidalTotalMomentumJacobian(idyn_cmm)
    idyn_cmm = idyn_cmm.toNumPy()

    # CoM position
    idyn_com = kin_dyn.getCenterOfMassPosition().toNumPy()

    # Com jacobian
    idyn_com_jacobian = idyntree.MatrixDynSize(3, kin_dyn.model().getNrOfDOFs() + 6)
    kin_dyn.getCenterOfMassJacobian(idyn_com_jacobian)
    idyn_com_jacobian = idyn_com_jacobian.toNumPy()

    # total mass
    total_mass = kin_dyn.model().getTotalMass()

    # jacobian
    idyn_jacobian = idyntree.MatrixDynSize(6, kin_dyn.model().getNrOfDOFs() + 6)
    kin_dyn.getFrameFreeFloatingJacobian(frames["frame"], idyn_jacobian)
    idyn_jacobian = idyn_jacobian.toNumPy()

    # jacobian_non_actuated
    idyn_jacobian_non_actuated = idyntree.MatrixDynSize(
        6, kin_dyn.model().getNrOfDOFs() + 6
    )
    kin_dyn.getFrameFreeFloatingJacobian(
        frames["frame_non_actuated"], idyn_jacobian_non_actuated
    )
    idyn_jacobian_non_actuated = idyn_jacobian_non_actuated.toNumPy()

    # jacobian_dot_nu
    idyn_jacobian_dot_nu = kin_dyn.getFrameBiasAcc(frames["frame"]).toNumPy()

    # relative_jacobian
    # set the base pose to the identity to get the relative jacobian
    kin_dyn.setRobotState(
        np.eye(4),
        state.joints_pos,
        state.base_vel,
        state.joints_vel,
        state.gravity,
    )
    idyn_relative_jacobian = idyntree.MatrixDynSize(
        6, kin_dyn.model().getNrOfDOFs() + 6
    )
    kin_dyn.getFrameFreeFloatingJacobian(frames["frame"], idyn_relative_jacobian)
    idyn_relative_jacobian = idyn_relative_jacobian.toNumPy()[:, 6:]

    # forward_kinematics
    # set back the state to the original one
    kin_dyn.setRobotState(
        state.H, state.joints_pos, state.base_vel, state.joints_vel, state.gravity
    )
    idyn_forward_kinematics = (
        kin_dyn.getWorldTransform(frames["frame"]).asHomogeneousTransform().toNumPy()
    )

    # forward_kinematics_non_actuated
    idyn_forward_kinematics_non_actuated = (
        kin_dyn.getWorldTransform(
            frames["frame_non_actuated"]
        ).asHomogeneousTransform().toNumPy()
    )

    # bias_force
    idyn_bias_force = idyntree.FreeFloatingGeneralizedTorques(kin_dyn.model())
    assert kin_dyn.generalizedBiasForces(idyn_bias_force)
    idyn_bias_force = np.concatenate(
        (
            idyn_bias_force.baseWrench().toNumPy(),
            idyn_bias_force.jointTorques().toNumPy(),
        )
    )

    # coriolis_term
    # set gravity to zero to get only the coriolis term
    kin_dyn.setRobotState(
        state.H, state.joints_pos, state.base_vel, state.joints_vel, np.zeros(3)
    )
    idyn_coriolis_term = idyntree.FreeFloatingGeneralizedTorques(kin_dyn.model())
    assert kin_dyn.generalizedBiasForces(idyn_coriolis_term)
    idyn_coriolis_term = np.concatenate(
        (
            idyn_coriolis_term.baseWrench().toNumPy(),
            idyn_coriolis_term.jointTorques().toNumPy(),
        )
    )

    # gravity_term
    # set gravity to the actual value and velocities to zero to get only the gravity term
    kin_dyn.setRobotState(
        state.H,
        state.joints_pos,
        np.zeros(6),
        np.zeros(kin_dyn.model().getNrOfDOFs()),
        state.gravity,
    )
    idyn_gravity_term = idyntree.FreeFloatingGeneralizedTorques(kin_dyn.model())
    assert kin_dyn.generalizedBiasForces(idyn_gravity_term)
    idyn_gravity_term = np.concatenate(
        (
            idyn_gravity_term.baseWrench().toNumPy(),
            idyn_gravity_term.jointTorques().toNumPy(),
        )
    )

    return IDynFunctionValues(
        mass_matrix=idyn_mass_matrix,
        centroidal_momentum_matrix=idyn_cmm,
        CoM_position=idyn_com,
        CoM_jacobian=idyn_com_jacobian,
        total_mass=total_mass,
        jacobian=idyn_jacobian,
        jacobian_non_actuated=idyn_jacobian_non_actuated,
        jacobian_dot_nu=idyn_jacobian_dot_nu,
        relative_jacobian=idyn_relative_jacobian,
        forward_kinematics=idyn_forward_kinematics,
        forward_kinematics_non_actuated=idyn_forward_kinematics_non_actuated,
        bias_force=idyn_bias_force,
        coriolis_term=idyn_coriolis_term,
        gravity_term=idyn_gravity_term,
    )


@pytest.fixture(scope="session")
def device():
    import torch

    """Pick CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy(x):
    """Convert a torch tensor to a numpy array, handling gradients if present."""
    if x.device.type == "cuda":
        x = x.cpu()
    return x.detach().numpy()


@pytest.fixture(
    scope="module", params=TEST_CONFIGURATIONS_WITH_SPHERICAL, ids=str
)
def tests_setup_with_spherical(request) -> RobotCfg | State:
    """Fixture that includes robots with spherical joints.
    
    Note: Some tests may fail for spherical joint robots as support
    is still partial (e.g., jacobian_dot, coriolis_term).
    """
    velocity_representation, robot_name = request.param

    np.random.seed(42)

    model_path = get_robot_model_path(robot_name)
    joints_name_list = get_robot_joints_list(robot_name)
    frames = get_robot_frames(robot_name)

    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Showing the robot tree.")

    if is_spherical_robot(robot_name):
        # For robots with spherical joints, we need to load via ADAM first
        # and then convert to iDynTree model
        adam_numpy_kin_dyn = KinDynComputationsNumpy(
            model_path, joints_name_list
        )
        idyn_model = to_idyntree_model(adam_numpy_kin_dyn.rbdalgos.model)

        kin_dyn = idyntree.KinDynComputations()
        assert kin_dyn.loadRobotModel(idyn_model)

        n_dof = adam_numpy_kin_dyn.NDoF  # velocity DOFs
        n_pos_dof = adam_numpy_kin_dyn.rbdalgos.NPosDof  # position DOFs

        # Generate random joint positions (quaternions for spherical joints)
        n_spherical_joints = len(joints_name_list)
        joints_val = np.zeros(n_pos_dof)
        for i in range(n_spherical_joints):
            quat = generate_random_quaternion()
            joints_val[i * 4:(i + 1) * 4] = quat

        # Velocities are 3 DOFs per spherical joint
        joints_dot_val = (np.random.rand(n_dof) - 0.5) * 5
    else:
        robot_iDyn = idyntree.ModelLoader()
        robot_iDyn.loadReducedModelFromFile(model_path, joints_name_list)

        kin_dyn = idyntree.KinDynComputations()
        kin_dyn.loadRobotModel(robot_iDyn.model())

        n_dof = len(joints_name_list)

        # joints quantities
        joints_val = (np.random.rand(n_dof) - 0.5) * 5
        joints_dot_val = (np.random.rand(n_dof) - 0.5) * 5

    if velocity_representation == Representations.BODY_FIXED_REPRESENTATION:
        idyn_representation = idyntree.BODY_FIXED_REPRESENTATION
    elif velocity_representation == Representations.MIXED_REPRESENTATION:
        idyn_representation = idyntree.MIXED_REPRESENTATION
    else:
        raise ValueError(
            f"Unknown velocity representation: {velocity_representation}"
        )
    kin_dyn.setFrameVelocityRepresentation(idyn_representation)

    # base quantities
    xyz = (np.random.rand(3) - 0.5) * 5
    rpy = (np.random.rand(3) - 0.5) * 5
    base_vel = (np.random.rand(6) - 0.5) * 5

    g = np.array([0, 0, -9.80665])
    R_b = Rotation.from_euler("xyz", rpy).as_matrix()
    H_b = np.eye(4)
    H_b[:3, :3] = R_b
    H_b[:3, 3] = xyz

    state = State(
        H=H_b,
        joints_pos=joints_val,
        base_vel=base_vel,
        joints_vel=joints_dot_val,
        gravity=g,
    )

    idyn_function_values = compute_idyntree_values(kin_dyn, state, frames)

    robot_cfg = RobotCfg(
        robot_name=robot_name,
        velocity_representation=velocity_representation,
        model_path=model_path,
        joints_name_list=joints_name_list,
        n_dof=n_dof,
        kin_dyn=kin_dyn,
        idyn_function_values=idyn_function_values,
        frame=frames["frame"],
        frame_non_actuated=frames["frame_non_actuated"],
    )

    yield robot_cfg, state
