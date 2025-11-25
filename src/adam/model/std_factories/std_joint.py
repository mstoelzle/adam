from typing import Union

import numpy.typing as npt
import urdf_parser_py.urdf

from adam.core.spatial_math import SpatialMath
from adam.model import Joint, Limits, Pose
import math


class StdJoint(Joint):
    """Standard Joint class"""

    def __init__(
        self,
        joint: urdf_parser_py.urdf.Joint,
        math: SpatialMath,
        idx: Union[int, None] = None,
    ) -> None:
        self.math = math
        self.name = joint.name
        self.parent = joint.parent
        self.child = joint.child
        self.type = joint.joint_type
        self.axis = self._set_axis(joint.axis)
        self.origin = self._set_origin(joint.origin)
        self.limit = self._set_limits(joint.limit)
        self.idx = idx
        self.dofs = self._infer_dofs()
        self.pos_dofs = self._infer_pos_dofs()

    def _infer_dofs(self) -> int:
        """Number of velocity DOFs (generalized velocities)."""
        if self.type == "fixed":
            return 0
        if self.type == "spherical":
            return 3  # Angular velocity has 3 components
        return 1

    def _infer_pos_dofs(self) -> int:
        """Number of position coordinates (generalized positions).
        
        For most joints this equals dofs, but spherical joints use
        4 quaternion coordinates for position while having 3 velocity DOFs.
        """
        if self.type == "fixed":
            return 0
        if self.type == "spherical":
            return 4  # Quaternion (w, x, y, z) has 4 components
        return 1

    def _set_axis(self, axis: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            axis (npt.ArrayLike): axis

        Returns:
            npt.ArrayLike: set the axis
        """
        return None if axis is None else self.math.asarray(axis)

    def _set_origin(self, origin: Pose) -> Pose:
        """
        Args:
            origin (Pose): origin

        Returns:
            Pose: set the origin
        """
        return Pose.build(xyz=origin.xyz, rpy=origin.rpy, math=self.math)

    def _set_limits(self, limit: Limits) -> Limits:
        """
        Args:
            limit (Limits): limit

        Returns:
            Limits: set the limits
        """
        joint_lim = math.inf if self.type == "prismatic" else 2 * math.pi
        return Limits(
            lower=-joint_lim if limit is None else limit.lower,
            upper=joint_lim if limit is None else limit.upper,
            effort=math.inf if limit is None else limit.effort,
            velocity=math.inf if limit is None else limit.velocity,
        )

    def homogeneous(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint value

        Returns:
            npt.ArrayLike: the homogenous transform of a joint, given q
        """

        if self.type == "fixed":
            xyz = self.origin.xyz
            rpy = self.origin.rpy
            return self.math.H_from_Pos_RPY(xyz, rpy)
        elif self.type in ["revolute", "continuous"]:
            return self.math.H_revolute_joint(
                self.origin.xyz,
                self.origin.rpy,
                self.axis,
                q,
            )
        elif self.type in ["prismatic"]:
            return self.math.H_prismatic_joint(
                self.origin.xyz,
                self.origin.rpy,
                self.axis,
                q,
            )
        elif self.type in ["spherical"]:
            return self.math.H_spherical_joint(
                self.origin.xyz,
                self.origin.rpy,
                q,
            )

    def spatial_transform(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint motion

        Returns:
            npt.ArrayLike: spatial transform of the joint given q
        """
        if self.type == "fixed":
            return self.math.X_fixed_joint(self.origin.xyz, self.origin.rpy)
        elif self.type in ["revolute", "continuous"]:
            return self.math.X_revolute_joint(
                self.origin.xyz, self.origin.rpy, self.axis, q
            )
        elif self.type in ["prismatic"]:
            return self.math.X_prismatic_joint(
                self.origin.xyz,
                self.origin.rpy,
                self.axis,
                q,
            )
        elif self.type in ["spherical"]:
            return self.math.X_spherical_joint(
                self.origin.xyz,
                self.origin.rpy,
                q,
            )

    def motion_subspace(self, q: npt.ArrayLike | None = None) -> npt.ArrayLike:
        """
        Args:
            joint (Joint): Joint
            q (npt.ArrayLike): joint position (used for position-dependent S)

        Returns:
            npt.ArrayLike: motion subspace of the joint
        """
        if self.type == "fixed":
            return self.math.zeros(6, 0)
        elif self.type in ["revolute", "continuous"]:
            axis = self.axis
            z = self.math.zeros(1)
            return self.math.vertcat(z, z, z, axis[0], axis[1], axis[2])
        elif self.type in ["prismatic"]:
            axis = self.axis
            zero = self.math.zeros(
                3,
            )
            return self.math.vertcat(axis[0], axis[1], axis[2], zero, zero, zero)
        elif self.type in ["spherical"]:
            # For spherical joints using quaternion position + angular velocity:
            # The motion subspace S maps angular velocity (3 DOFs) to spatial velocity
            # S = [0; I_3x3] - angular velocity expressed in child frame
            # This is constant (does not depend on q) when using angular velocity
            # as the velocity coordinates (like iDynTree SphericalJoint)
            zeros = self.math.zeros(3, 3)
            eye = self.math.eye(3)
            return self.math.vertcat(zeros, eye)

    def motion_subspace_dot(
        self, q: npt.ArrayLike, q_dot: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint position
            q_dot (npt.ArrayLike): joint velocity

        Returns:
            npt.ArrayLike: time derivative of the motion subspace of the joint
        """
        # For spherical joints with quaternion position + angular velocity,
        # the motion subspace S = [0; I] is constant, so S_dot = 0
        # This is because angular velocity directly gives body angular velocity.
        S = self.motion_subspace(q)
        return self.math.zeros_like(S)
