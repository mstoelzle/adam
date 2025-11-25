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

    def _infer_dofs(self) -> int:
        if self.type == "fixed":
            return 0
        if self.type == "spherical":
            return 3
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
            if q is None:
                zeros = self.math.zeros(3, 3)
                eye = self.math.eye(3)
                return self.math.vertcat([zeros, eye])

            q1 = q[..., 1]
            q2 = q[..., 2]

            # S = [0; S_ang]
            # S_ang columns are the axes of rotation expressed in the child frame
            # Col 1: Rz(-q2) * Ry(-q1) * [1; 0; 0]
            # Col 2: Rz(-q2) * [0; 1; 0]
            # Col 3: [0; 0; 1]

            Rz_neg_q2 = self.math.Rz(-q2)
            Ry_neg_q1 = self.math.Ry(-q1)

            # We need to handle batching for the axes
            batch_shape = q.shape[:-1]

            # Helper to create batched axis
            def make_axis(v):
                return self.math.asarray(v)

            x_axis = make_axis([1.0, 0.0, 0.0])
            y_axis = make_axis([0.0, 1.0, 0.0])
            z_axis = make_axis([0.0, 0.0, 1.0])

            col3_ang = z_axis
            if len(batch_shape) > 0:
                col3_ang = self.math.tile(z_axis, batch_shape + (1,))
                x_axis = self.math.tile(x_axis, batch_shape + (1,))
                y_axis = self.math.tile(y_axis, batch_shape + (1,))

            col2_ang = self.math.mxv(Rz_neg_q2, y_axis)
            col1_ang = self.math.mxv(
                Rz_neg_q2, self.math.mxv(Ry_neg_q1, x_axis)
            )

            S_ang = self.math.stack([col1_ang, col2_ang, col3_ang], axis=-1)
            S_lin = self.math.zeros_like(S_ang)
            return self.math.concatenate([S_lin, S_ang], axis=-2)

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
        if self.type in ["fixed", "revolute", "continuous", "prismatic"]:
            # For these joints, S is constant, so S_dot is zero
            S = self.motion_subspace(q)
            return self.math.zeros_like(S)
        elif self.type in ["spherical"]:
            q1 = q[..., 1]
            q2 = q[..., 2]
            qd1 = q_dot[..., 1]
            qd2 = q_dot[..., 2]

            # S_dot = [0; S_ang_dot]
            # S_ang_dot columns:
            # Col 3: 0
            # Col 2: -qd2 * (e3 x S2)
            # Col 1: -qd2 * (e3 x S1) - qd1 * Rz(-q2) * (e2 x (Ry(-q1) * e1))

            Rz_neg_q2 = self.math.Rz(-q2)
            Ry_neg_q1 = self.math.Ry(-q1)

            batch_shape = q.shape[:-1]

            def make_axis(v):
                return self.math.asarray(v)

            x_axis = make_axis([1.0, 0.0, 0.0])
            y_axis = make_axis([0.0, 1.0, 0.0])
            z_axis = make_axis([0.0, 0.0, 1.0])

            if len(batch_shape) > 0:
                x_axis = self.math.tile(x_axis, batch_shape + (1,))
                y_axis = self.math.tile(y_axis, batch_shape + (1,))
                z_axis = self.math.tile(z_axis, batch_shape + (1,))

            # Calculate S columns again (angular part)
            col3_ang = z_axis
            col2_ang = self.math.mxv(Rz_neg_q2, y_axis)
            Ry_neg_q1_x = self.math.mxv(Ry_neg_q1, x_axis)
            col1_ang = self.math.mxv(Rz_neg_q2, Ry_neg_q1_x)

            # Calculate derivatives
            # Col 3 dot is zero
            col3_dot_ang = self.math.zeros_like(col3_ang)

            # Col 2 dot = -qd2 * (z x col2_ang)
            z_skew = self.math.skew(z_axis)
            z_cross_col2 = self.math.mxv(z_skew, col2_ang)

            # qd2 needs to be broadcastable to vector
            qd2_expanded = qd2[..., None]
            col2_dot_ang = -qd2_expanded * z_cross_col2

            # Col 1 dot
            # Term 1: -qd2 * (z x col1_ang)
            z_cross_col1 = self.math.mxv(z_skew, col1_ang)
            term1 = -qd2_expanded * z_cross_col1

            # Term 2: -qd1 * Rz(-q2) * (y x Ry(-q1) * x)
            # y x Ry(-q1) * x = y x Ry_neg_q1_x
            y_skew = self.math.skew(y_axis)
            y_cross_Ry_x = self.math.mxv(y_skew, Ry_neg_q1_x)
            Rz_y_cross_Ry_x = self.math.mxv(Rz_neg_q2, y_cross_Ry_x)
            qd1_expanded = qd1[..., None]
            term2 = -qd1_expanded * Rz_y_cross_Ry_x

            col1_dot_ang = term1 + term2

            S_dot_ang = self.math.stack(
                [col1_dot_ang, col2_dot_ang, col3_dot_ang], axis=-1
            )
            S_dot_lin = self.math.zeros_like(S_dot_ang)
            return self.math.concatenate([S_dot_lin, S_dot_ang], axis=-2)
