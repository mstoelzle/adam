# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

import numpy.typing as npt

from adam.core.constants import Representations
from adam.core.spatial_math import ArrayLike, SpatialMath
from adam.model import Model, Node, Joint


class RBDAlgorithms:
    """This is a small class that implements Rigid body algorithms retrieving robot quantities, for Floating Base systems - as humanoid robots."""

    def __init__(self, model: Model, math: SpatialMath) -> None:
        """
        Args:
            model (Model): the adam.model representing the robot
            math (SpatialMath): the spatial math.
        """

        self.model = model
        self.NDoF = model.NDoF
        self.NPosDof = model.NPosDof
        self.root_link = self.model.tree.root
        self.math = math
        self.frame_velocity_representation = (
            Representations.MIXED_REPRESENTATION
        )  # default
        # Cache root quantities that are reused at every call
        self._root_spatial_transform = self.math.spatial_transform(
            self.math.factory.eye(3),
            self.math.factory.zeros((3, 1)),
        )
        self._root_motion_subspace = self.math.factory.eye(6)
        self._prepare_tree_cache()

        # add boolean to check if casadi math is used
        self._is_casadi_math = type(math.factory).__name__ == "CasadiLikeFactory"

    def set_frame_velocity_representation(self, representation: Representations):
        """Sets the frame velocity representation

        Args:
            representation (str): The representation of the frame velocity
        """
        self.frame_velocity_representation = representation

    def crba(self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike):
        """
        Batched Composite Rigid Body Algorithm (CRBA) + Orin's Centroidal Momentum Matrix.

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            M, Jcm (npt.ArrayLike, npt.ArrayLike): The mass matrix and the centroidal momentum matrix
        """
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )

        math = self.math
        n = self.NDoF
        node_count = self._node_count
        node_indices = self._node_indices
        rev_indices = self._rev_node_indices
        parent_indices = self._parent_indices
        joint_indices = self._joint_indices_per_node
        vel_joint_indices = self._vel_joint_indices_per_node
        inertias = self._spatial_inertias
        joints = self._joints_per_node
        root_idx = self._root_index
        joint_to_node = self._joint_index_to_node
        joint_dofs = self._dofs_per_node

        if len(joint_positions.shape) >= 2:
            batch_shape = joint_positions.shape[:-1]
        elif base_transform.ndim > 2:
            batch_shape = base_transform.shape[:-2]
        else:
            batch_shape = ()

        if joint_positions.shape[-1] > 0:
            zero_scalar = math.zeros_like(joint_positions[..., 0])
        else:
            zero_scalar = math.zeros_like(base_transform[..., 0, 0])

        def zeros_for_dofs(count: int):
            if count == 0:
                return math.factory.zeros(batch_shape + ())
            if count == 1:
                return zero_scalar
            return math.factory.zeros(batch_shape + (count,))

        def tile_batch(arr):
            if not batch_shape:
                return arr
            reps = batch_shape + (1,) * len(arr.shape)
            return math.tile(arr, reps)

        Xup = [None] * node_count
        Phi = [None] * node_count
        Ic = [None] * node_count

        for idx in node_indices:
            Ic[idx] = tile_batch(inertias[idx])
            if idx == root_idx:
                Xup[idx] = tile_batch(self._root_spatial_transform)
                Phi[idx] = tile_batch(self._root_motion_subspace)
            else:
                joint = joints[idx]
                joint_idx = joint_indices[idx]
                dof_count = joint_dofs[idx]
                q_i = (
                    joint_positions[..., joint_idx]
                    if joint_idx is not None
                    else zeros_for_dofs(dof_count)
                )
                X_current = (
                    joint.spatial_transform(q=q_i)
                    if joint is not None
                    else self._root_spatial_transform
                )
                Xup[idx] = X_current
                S_i = (
                    joint.motion_subspace(q_i)
                    if joint is not None
                    else self._root_motion_subspace
                )
                Phi[idx] = tile_batch(S_i)

        Xup_T = [math.swapaxes(Xup[i], -2, -1) for i in range(node_count)]
        Ic_comp = Ic[:]

        for idx in rev_indices:
            parent = parent_indices[idx]
            if parent >= 0:
                Ic_comp[parent] = Ic_comp[parent] + math.mtimes(
                    math.mtimes(Xup_T[idx], Ic_comp[idx]),
                    Xup[idx],
                )

        sizes = [6] + [1] * n
        blocks: list[list[npt.ArrayLike | None]] = [
            [None for _ in range(n + 1)] for _ in range(n + 1)
        ]

        for idx in rev_indices:
            Phi_i = Phi[idx]
            Phi_T = math.swapaxes(Phi_i, -2, -1)
            F = math.mtimes(Ic_comp[idx], Phi_i)
            # Use vel_joint_indices for mass matrix block placement
            dof_indices_i = self._expand_joint_indices(vel_joint_indices[idx])

            if idx == root_idx:
                blocks[0][0] = math.mtimes(Phi_T, F)
                continue

            if dof_indices_i:
                PhiTF = math.mtimes(Phi_T, F)
                for local_r, dof_r in enumerate(dof_indices_i):
                    for local_c, dof_c in enumerate(dof_indices_i):
                        blocks[1 + dof_r][1 + dof_c] = PhiTF[
                            ..., local_r : local_r + 1, local_c : local_c + 1
                        ]

            current = idx
            F_path = F
            while current != root_idx:
                parent = parent_indices[current]
                F_path = math.mtimes(Xup_T[current], F_path)
                Bij = math.mtimes(math.swapaxes(F_path, -2, -1), Phi[parent])
                if not dof_indices_i:
                    current = parent
                    continue
                if parent == root_idx:
                    for local_r, dof_r in enumerate(dof_indices_i):
                        row_block = Bij[..., local_r : local_r + 1, :]
                        blocks[1 + dof_r][0] = row_block
                        blocks[0][1 + dof_r] = math.swapaxes(row_block, -2, -1)
                else:
                    dof_indices_parent = self._expand_joint_indices(
                        vel_joint_indices[parent]
                    )
                    if not dof_indices_parent:
                        current = parent
                        continue
                    for local_r, dof_r in enumerate(dof_indices_i):
                        for local_p, dof_p in enumerate(dof_indices_parent):
                            sub_block = Bij[
                                ..., local_r : local_r + 1, local_p : local_p + 1
                            ]
                            blocks[1 + dof_r][1 + dof_p] = sub_block
                            blocks[1 + dof_p][1 + dof_r] = math.swapaxes(
                                sub_block, -2, -1
                            )
                current = parent

        row_tensors = []
        for r in range(n + 1):
            row_blocks = []
            for c in range(n + 1):
                block = blocks[r][c]
                if block is None:
                    block = math.factory.zeros(batch_shape + (sizes[r], sizes[c]))
                row_blocks.append(block)
            row_tensors.append(math.concatenate(row_blocks, axis=-1))
        M = math.concatenate(row_tensors, axis=-2)

        A = math.swapaxes(M[..., :3, 3:6], -2, -1) / M[..., 0, 0][..., None, None]
        I3 = math.factory.eye(batch_shape + (3,))
        Z3 = math.factory.zeros(batch_shape + (3, 3))
        top = math.concatenate([I3, A], axis=-1)
        bot = math.concatenate([Z3, I3], axis=-1)
        O_X_G = math.concatenate([top, bot], axis=-2)

        X_G = [None] * node_count
        for idx in node_indices:
            if idx == root_idx:
                X_G[idx] = O_X_G
            else:
                parent = parent_indices[idx]
                X_G[idx] = math.mtimes(Xup[idx], X_G[parent])

        J_base = math.mtimes(
            math.swapaxes(X_G[root_idx], -2, -1),
            math.mtimes(Ic_comp[root_idx], Phi[root_idx]),
        )

        zero_col = math.factory.zeros(batch_shape + (6, 1))
        joint_cols = []
        for jidx in range(n):
            node_idx = joint_to_node.get(jidx)
            if node_idx is not None:
                local_indices = self._expand_joint_indices(
                    vel_joint_indices[node_idx]
                )
                local_pos = local_indices.index(jidx)
                F_node = math.mtimes(Ic_comp[node_idx], Phi[node_idx])
                col_full = math.mtimes(
                    math.swapaxes(X_G[node_idx], -2, -1),
                    F_node,
                )
                col = col_full[..., :, local_pos : local_pos + 1]
            else:
                col = zero_col
            joint_cols.append(col)

        Jcm = math.concatenate([J_base] + joint_cols, axis=-1)

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return M, Jcm

        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            Xm = math.adjoint_mixed_inverse(base_transform)
            In = math.factory.eye(batch_shape + (n,))
            Z6n = math.factory.zeros(batch_shape + (6, n))
            Zn6 = math.factory.zeros(batch_shape + (n, 6))

            top = math.concatenate([Xm, Z6n], axis=-1)
            bot = math.concatenate([Zn6, In], axis=-1)
            X_to_mixed = math.concatenate([top, bot], axis=-2)

            M_mixed = math.mtimes(
                math.swapaxes(X_to_mixed, -2, -1), math.mtimes(M, X_to_mixed)
            )
            Jcm_mixed = math.mtimes(
                math.swapaxes(Xm, -2, -1), math.mtimes(Jcm, X_to_mixed)
            )
            return M_mixed, Jcm_mixed

        raise ValueError(
            f"Unknown frame velocity representation: {self.frame_velocity_representation}"
        )

    def forward_kinematics(
        self, frame, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Computes the forward kinematics relative to the specified `frame`.

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            I_H_L (npt.ArrayLike): The fk represented as Homogenous transformation matrix
        """
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )
        chain = self.model.get_joints_chain(self.root_link, frame)
        I_H_L = base_transform
        for joint in chain:
            dof_count = getattr(joint, "dofs", 1)
            if joint.idx is not None:
                q_ = joint_positions[..., joint.idx]
            else:
                batch = joint_positions.shape[:-1] if joint_positions.ndim > 1 else ()
                if dof_count == 0:
                    q_ = self.math.factory.zeros(batch + ())
                elif dof_count == 1:
                    q_ = self.math.zeros_like(joint_positions[..., 0])
                else:
                    q_ = self.math.factory.zeros(batch + (dof_count,))
            H_joint = joint.homogeneous(q=q_)
            I_H_L = I_H_L @ H_joint
        return I_H_L

    def joints_jacobian(
        self, frame: str, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the Jacobian relative to the specified `frame`.

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            J (npt.ArrayLike): The Joints Jacobian relative to the frame
        """
        joint_positions = self._convert_to_arraylike(joint_positions)

        batch_size = (
            joint_positions.shape[:-1] if len(joint_positions.shape) >= 2 else ()
        )

        chain = self.model.get_joints_chain(self.root_link, frame)
        eye = self.math.factory.eye(batch_size + (4,))
        B_H_j = eye
        B_H_L = self.forward_kinematics(frame, eye, joint_positions)
        L_H_B = self.math.homogeneous_inverse(B_H_L)
        cols = [None] * self.NDoF
        for joint in chain:
            dof_count = getattr(joint, "dofs", 1)
            pos_dof_count = getattr(joint, "pos_dofs", dof_count)
            if joint.idx is not None:
                q = joint_positions[..., joint.idx]
            else:
                batch = joint_positions.shape[:-1] if joint_positions.ndim > 1 else ()
                if pos_dof_count == 0:
                    q = self.math.factory.zeros(batch + ())
                elif pos_dof_count == 1:
                    q = self.math.zeros_like(joint_positions[..., 0])
                else:
                    q = self.math.factory.zeros(batch + (pos_dof_count,))
            H_j = joint.homogeneous(q=q)
            B_H_j = B_H_j @ H_j
            L_H_j = L_H_B @ B_H_j
            S = joint.motion_subspace(q)
            # Use vel_idx for Jacobian column placement
            vel_idx = getattr(joint, "vel_idx", joint.idx)
            if vel_idx is not None:
                J_j = self.math.adjoint(L_H_j) @ S
                for local, global_idx in enumerate(
                    self._expand_joint_indices(vel_idx)
                ):
                    cols[global_idx] = J_j[..., :, local : local + 1]

        zero_col = self.math.zeros(batch_size + (6, 1))
        cols = [zero_col if col is None else col for col in cols]
        # Use concatenate instead of stack to get (batch, 6, joints) directly
        return self.math.concatenate(cols, axis=-1)

    def jacobian(
        self, frame: str, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the Jacobian for `frame`.

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            npt.ArrayLike: The Jacobian for the specified frame
        """

        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )

        eye = self.math.factory.eye(4)
        J = self.joints_jacobian(frame, joint_positions)
        B_H_L = self.forward_kinematics(frame, eye, joint_positions)
        L_X_B = self.math.adjoint_inverse(B_H_L)
        J_tot = self.math.concatenate([L_X_B, J], axis=-1)
        w_H_B = base_transform
        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return J_tot
        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            w_H_L = w_H_B @ B_H_L
            LI_X_L = self.math.adjoint_mixed(w_H_L)

            top_left = self.math.adjoint_mixed_inverse(base_transform)
            top_right = self.math.factory.zeros(
                joint_positions.shape[:-1] + (6, self.NDoF)
            )
            bottom_left = self.math.factory.zeros(
                joint_positions.shape[:-1] + (self.NDoF, 6)
            )
            bottom_right = self.math.factory.eye(
                joint_positions.shape[:-1] + (self.NDoF,)
            )
            top = self.math.concatenate([top_left, top_right], axis=-1)
            bottom = self.math.concatenate([bottom_left, bottom_right], axis=-1)
            X = self.math.concatenate([top, bottom], axis=-2)

            J_tot = LI_X_L @ J_tot @ X
            return J_tot
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION and MIXED_REPRESENTATION are implemented"
            )

    def relative_jacobian(
        self, frame: str, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the Jacobian between the root link and a specified `frame`.

        Args:
            frame (str): The tip of the chain
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            J (npt.ArrayLike): The 6 x NDoF Jacobian between the root and the `frame`
        """
        joint_positions = self._convert_to_arraylike(joint_positions)
        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return self.joints_jacobian(frame, joint_positions)
        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            eye = self.math.factory.eye(4)
            B_H_L = self.forward_kinematics(frame, eye, joint_positions)
            LI_X_L = self.math.adjoint_mixed(B_H_L)
            return LI_X_L @ self.joints_jacobian(frame, joint_positions)
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION and MIXED_REPRESENTATION are implemented"
            )

    def jacobian_dot(
        self,
        frame: str,
        base_transform: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
        base_velocity: npt.ArrayLike,
        joint_velocities: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """Returns the Jacobian time derivative for `frame`.

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position
            base_velocity (npt.ArrayLike): The spatial velocity of the base
            joint_velocities (npt.ArrayLike): The joints velocities

        Returns:
            J_dot (npt.ArrayLike): The Jacobian derivative relative to the frame

        """

        base_transform, joint_positions, base_velocity, joint_velocities = (
            self._convert_to_arraylike(
                base_transform, joint_positions, base_velocity, joint_velocities
            )
        )

        batch_size = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()

        I4 = self.math.factory.eye(batch_size + (4,))

        B_H_L = self.forward_kinematics(frame, I4, joint_positions)
        L_H_B = self.math.homogeneous_inverse(B_H_L)

        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            B_v_IB = self.math.mxv(
                self.math.adjoint_mixed_inverse(base_transform), base_velocity
            )
        elif (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            B_v_IB = base_velocity
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION and MIXED_REPRESENTATION are implemented"
            )

        v = self.math.mxv(self.math.adjoint(L_H_B), B_v_IB)
        a = self.math.mxv(self.math.adjoint_derivative(L_H_B, v), B_v_IB)

        J_base_full = self.math.adjoint_inverse(B_H_L)
        J_base_cols = [J_base_full[..., :, i : i + 1] for i in range(6)]
        J_dot_base_full = self.math.adjoint_derivative(L_H_B, v)
        J_dot_base_cols = [J_dot_base_full[..., :, i : i + 1] for i in range(6)]

        cols: list = [None] * self.NDoF
        cols_dot: list = [None] * self.NDoF

        B_H_j = I4
        chain = self.model.get_joints_chain(self.root_link, frame)

        for joint in chain:
            dof_count = getattr(joint, "dofs", 1)
            vel_idx = getattr(joint, "vel_idx", joint.idx)
            if joint.idx is not None:
                q = joint_positions[..., joint.idx]
            else:
                q = self._zeros_for_dofs(
                    dof_count, batch_size, reference=joint_positions
                )
            if vel_idx is not None:
                q_dot = joint_velocities[..., vel_idx]
            else:
                q_dot = self._zeros_for_dofs(
                    dof_count, batch_size, reference=joint_velocities
                )

            H_j = joint.homogeneous(q=q)
            B_H_j = B_H_j @ H_j
            L_H_j = L_H_B @ B_H_j
            S = joint.motion_subspace(q)
            if S.shape[-1] == 0:
                J_j = self.math.factory.zeros(batch_size + (6, 0))
                J_dot_j = self.math.factory.zeros(batch_size + (6, 0))
            else:
                J_j = self.math.adjoint(L_H_j) @ S
                q_dot_vec = self._ensure_vector(q_dot, dof_count, batch_size)
                v = v + self.math.mxv(J_j, q_dot_vec)
                S_dot = joint.motion_subspace_dot(q, q_dot)
                J_dot_j = (
                    self.math.adjoint_derivative(L_H_j, v) @ S
                    + self.math.adjoint(L_H_j) @ S_dot
                )
                a = a + self.math.mxv(J_dot_j, q_dot_vec)

            if vel_idx is not None:
                for local, global_idx in enumerate(
                    self._expand_joint_indices(vel_idx)
                ):
                    cols[global_idx] = J_j[..., :, local : local + 1]
                    cols_dot[global_idx] = J_dot_j[..., :, local : local + 1]

        zero_col = self.math.factory.zeros(batch_size + (6, 1))
        cols = [zero_col if c is None else c for c in cols]
        cols_dot = [zero_col if c is None else c for c in cols_dot]

        J = self.math.concatenate([*J_base_cols, *cols], axis=-1)
        J_dot = self.math.concatenate([*J_dot_base_cols, *cols_dot], axis=-1)

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return J_dot

        I_H_L = self.forward_kinematics(frame, base_transform, joint_positions)
        LI_X_L = self.math.adjoint_mixed(I_H_L)
        LI_v_L = self.math.mxv(LI_X_L, v)
        LI_X_L_dot = self.math.adjoint_mixed_derivative(I_H_L, LI_v_L)

        adj_mixed_inv = self.math.adjoint_mixed_inverse(base_transform)

        Z_6xN = self.math.factory.zeros(batch_size + (6, self.NDoF))
        Z_Nx6 = self.math.factory.zeros(batch_size + (self.NDoF, 6))
        I_N = self.math.factory.eye(batch_size + (self.NDoF,))

        top = self.math.concatenate([adj_mixed_inv, Z_6xN], axis=-1)
        bottom = self.math.concatenate([Z_Nx6, I_N], axis=-1)
        X = self.math.concatenate([top, bottom], axis=-2)

        B_H_I = self.math.homogeneous_inverse(base_transform)
        B_H_I_deriv = self.math.adjoint_mixed_derivative(B_H_I, -B_v_IB)

        Z_NxN = self.math.factory.zeros(batch_size + (self.NDoF, self.NDoF))
        topd = self.math.concatenate([B_H_I_deriv, Z_6xN], axis=-1)
        bottomd = self.math.concatenate([Z_Nx6, Z_NxN], axis=-1)
        X_dot = self.math.concatenate([topd, bottomd], axis=-2)

        return (LI_X_L_dot @ J @ X) + (LI_X_L @ J_dot @ X) + (LI_X_L @ J @ X_dot)

    def CoM_position(
        self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the CoM position

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            com (npt.ArrayLike): The CoM position
        """
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )
        batch_size = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()

        com_pos = self.math.factory.zeros(batch_size + (3,))
        for item in self.model.tree:
            link = item.link
            I_H_l = self.forward_kinematics(link.name, base_transform, joint_positions)
            H_link = link.homogeneous()
            I_H_l = I_H_l @ H_link
            link_pos = I_H_l[..., :3, 3:4]
            com_pos += self.math.vxs(link_pos, link.inertial.mass)
        com_pos /= self._convert_to_arraylike(self.get_total_mass())
        return com_pos

    def CoM_jacobian(
        self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the center of mass (CoM) Jacobian using the centroidal momentum matrix.

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            J_com (npt.ArrayLike): The CoM Jacobian
        """
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )
        batch_size = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()
        ori_frame_velocity_representation = self.frame_velocity_representation
        self.frame_velocity_representation = Representations.MIXED_REPRESENTATION
        _, Jcm = self.crba(base_transform, joint_positions)
        if (
            ori_frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            Xm = self.math.adjoint_mixed(base_transform)
            In = self.math.factory.eye(batch_size + (self.NDoF,))
            Z6n = self.math.factory.zeros(batch_size + (6, self.NDoF))
            Zn6 = self.math.factory.zeros(batch_size + (self.NDoF, 6))

            top = self.math.concatenate([Xm, Z6n], axis=-1)
            bot = self.math.concatenate([Zn6, In], axis=-1)
            X = self.math.concatenate([top, bot], axis=-2)
            Jcm = Jcm @ X
        self.frame_velocity_representation = ori_frame_velocity_representation
        return Jcm[..., :3, :] / self._convert_to_arraylike(self.get_total_mass())

    def get_total_mass(self):
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        return self.model.get_total_mass()

    def rnea(
        self,
        base_transform: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
        base_velocity: npt.ArrayLike,
        joint_velocities: npt.ArrayLike,
        g: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Batched Recursive Newton-Euler (reduced: no joint/base accelerations, no external forces).

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position
            base_velocity (npt.ArrayLike): The base spatial velocity
            joint_velocities (npt.ArrayLike): The joints velocities
            g (npt.ArrayLike): The gravity vector

        Returns:
            tau (npt.ArrayLike): The vector of generalized forces
        """
        base_transform, joint_positions, base_velocity, joint_velocities, g = (
            self._convert_to_arraylike(
                base_transform, joint_positions, base_velocity, joint_velocities, g
            )
        )

        math = self.math
        n = self.NDoF
        node_count = self._node_count
        parent_indices = self._parent_indices
        joint_indices = self._joint_indices_per_node
        vel_joint_indices = self._vel_joint_indices_per_node
        inertias = self._spatial_inertias
        joints = self._joints_per_node
        root_idx = self._root_index
        node_indices = self._node_indices
        rev_indices = self._rev_node_indices
        joint_dofs = self._dofs_per_node

        batch_shape = (
            tuple(base_transform.shape[:-2]) if base_transform.ndim > 2 else ()
        )

        gravity_X = math.adjoint_mixed_inverse(base_transform)  # (...,6,6)

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            B_X_BI = math.factory.eye(batch_shape + (6,))
            transformed_acc = math.factory.zeros(batch_shape + (6,))
        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            B_X_BI = math.adjoint_mixed_inverse(base_transform)
            omega = base_velocity[..., 3:]
            vlin = base_velocity[..., :3]
            skew_omega_times_vlin = math.mxv(math.skew(omega), vlin)
            top3 = -math.mxv(B_X_BI[..., :3, :3], skew_omega_times_vlin)
            bot3 = math.factory.zeros(batch_shape + (3,))
            transformed_acc = math.concatenate([top3, bot3], axis=-1)
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION and MIXED_REPRESENTATION are implemented"
            )

        a0 = -(math.mxv(gravity_X, g)) + transformed_acc

        zero_scalar_q = (
            math.zeros_like(joint_positions[..., 0])
            if joint_positions.shape[-1] > 0
            else math.zeros_like(base_velocity[..., 0])
        )
        zero_scalar_qd = (
            math.zeros_like(joint_velocities[..., 0])
            if n > 0
            else zero_scalar_q
        )

        def zeros_for_dofs(count: int, zero_scalar):
            if count == 0:
                return math.factory.zeros(batch_shape + ())
            if count == 1:
                return zero_scalar
            return math.factory.zeros(batch_shape + (count,))

        def zeros6():
            return math.factory.zeros(batch_shape + (6,))

        Ic = [None] * node_count
        Xup = [None] * node_count
        v = [None] * node_count
        a = [None] * node_count
        f = [None] * node_count

        for idx in node_indices:
            Ic[idx] = inertias[idx]

            if idx == root_idx:
                Xup[idx] = self._root_spatial_transform
                v[idx] = math.mxv(B_X_BI, base_velocity)
                a[idx] = math.mxv(Xup[idx], a0)
                continue

            joint = joints[idx]
            parent = parent_indices[idx]
            joint_idx = joint_indices[idx]
            vel_joint_idx = vel_joint_indices[idx]

            dof_count = joint_dofs[idx]
            q = (
                joint_positions[..., joint_idx]
                if joint_idx is not None
                else zeros_for_dofs(dof_count, zero_scalar_q)
            )
            qd = (
                joint_velocities[..., vel_joint_idx]
                if vel_joint_idx is not None
                else zeros_for_dofs(dof_count, zero_scalar_qd)
            )

            X = joint.spatial_transform(q=q)
            Xup[idx] = X

            # Only compute velocity contribution for joints in the selected list
            if vel_joint_idx is not None:
                Phi_i = joint.motion_subspace(q)
                if Phi_i.shape[-1] == 0:
                    phi_qd = zeros6()
                else:
                    qd_vec = self._ensure_vector(qd, dof_count, batch_shape)
                    phi_qd = math.mxv(Phi_i, qd_vec)
            else:
                phi_qd = zeros6()
            v[idx] = math.mxv(X, v[parent]) + phi_qd
            a[idx] = math.mxv(X, a[parent]) + math.mxv(
                math.spatial_skew(v[idx]), phi_qd
            )

            f[idx] = math.mxv(Ic[idx], a[idx]) + math.mxv(
                math.spatial_skew_star(v[idx]), math.mxv(Ic[idx], v[idx])
            )

        # Root wrench contribution (skipped in loop above)
        f[root_idx] = math.mxv(Ic[root_idx], a[root_idx]) + math.mxv(
            math.spatial_skew_star(v[root_idx]), math.mxv(Ic[root_idx], v[root_idx])
        )

        tau_base = None
        tau_joint_cols = [None] * n if n > 0 else []

        for idx in rev_indices:
            joint = joints[idx]
            joint_idx = joint_indices[idx]
            vel_joint_idx = vel_joint_indices[idx]
            if joint is not None and joint_idx is not None:
                dof_count = joint_dofs[idx]
                q_i = joint_positions[..., joint_idx]
                Phi_i = joint.motion_subspace(q_i)
            elif joint is not None:
                Phi_i = joint.motion_subspace(None)
            else:
                Phi_i = self._root_motion_subspace
            Fi = f[idx]
            Phi_T = math.swapaxes(Phi_i, -2, -1)

            if idx == root_idx:
                tau_base = math.mxv(Phi_T, Fi)
            else:
                if vel_joint_idx is not None:
                    joint_tau = math.mxv(Phi_T, Fi)
                    for local, global_idx in enumerate(
                        self._expand_joint_indices(vel_joint_idx)
                    ):
                        tau_joint_cols[global_idx] = joint_tau[
                            ..., local : local + 1
                        ]
                parent = parent_indices[idx]
                if parent >= 0:
                    f[parent] = f[parent] + math.mxv(
                        math.swapaxes(Xup[idx], -2, -1), Fi
                    )

        tau_base = math.mxv(math.swapaxes(B_X_BI, -2, -1), tau_base)

        if n > 0:
            zero_tau = math.factory.zeros(batch_shape + (1,))
            tau_joints = [
                col if col is not None else zero_tau for col in tau_joint_cols
            ]
            tau_joints_vec = math.concatenate(tau_joints, axis=-1)
        else:
            tau_joints_vec = math.factory.zeros(batch_shape + (0,))

        return math.concatenate([tau_base, tau_joints_vec], axis=-1)

    def aba(
        self,
        base_transform: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
        base_velocity: npt.ArrayLike,
        joint_velocities: npt.ArrayLike,
        joint_torques: npt.ArrayLike,
        g: npt.ArrayLike,
        external_wrenches: dict[str, npt.ArrayLike] | None = None,
    ) -> npt.ArrayLike:
        """Featherstone Articulated Body Algorithm for floating-base forward dynamics.

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position
            base_velocity (npt.ArrayLike): The spatial velocity of the base
            joint_velocities (npt.ArrayLike): The joints velocities
            joint_torques (npt.ArrayLike): The joints torques/forces
            g (npt.ArrayLike | None, optional): The gravity vector
            external_wrenches (dict[str, npt.ArrayLike] | None, optional): A dictionary of external wrenches applied to specific links. Keys are link names, and values are 6D wrench vectors. Defaults to None.

        Returns:
            accelerations (npt.ArrayLike): The spatial acceleration of the base and joints accelerations
        """
        math = self.math
        n = self.NDoF
        node_count = self._node_count
        parent_indices = self._parent_indices
        joint_indices = self._joint_indices_per_node
        vel_joint_indices = self._vel_joint_indices_per_node
        inertias = self._spatial_inertias
        joints = self._joints_per_node
        root_idx = self._root_index
        node_indices = self._node_indices
        rev_indices = self._rev_node_indices
        joint_dofs = self._dofs_per_node

        (
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            joint_torques,
            g,
        ) = self._convert_to_arraylike(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            joint_torques,
            g,
        )

        batch_shape = (
            tuple(base_transform.shape[:-2]) if base_transform.ndim > 2 else ()
        )

        if external_wrenches is not None:
            generalized_ext = math.factory.zeros(batch_shape + (6 + n,))
            for frame, wrench in external_wrenches.items():
                wrench_arr = self._convert_to_arraylike(wrench)
                J = self.jacobian(frame, base_transform, joint_positions)
                generalized_ext = generalized_ext + math.mxv(
                    math.swapaxes(J, -2, -1), wrench_arr
                )

            base_ext = generalized_ext[..., :6]
            joint_ext = (
                generalized_ext[..., 6:] if n > 0 else math.zeros_like(joint_torques)
            )
        else:
            base_ext = math.factory.zeros(batch_shape + (6,))
            joint_ext = math.zeros_like(joint_torques)

        joint_torques_eff = joint_torques + joint_ext

        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            B_X_BI = math.adjoint_mixed_inverse(base_transform)
        elif (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            B_X_BI = math.factory.eye(batch_shape + (6,))
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION and MIXED_REPRESENTATION are implemented"
            )

        base_velocity_body = math.mxv(B_X_BI, base_velocity)
        base_ext_body = math.mxv(B_X_BI, base_ext)

        a0_input = math.mxv(math.adjoint_mixed_inverse(base_transform), g)

        def zeros6():
            return math.factory.zeros(batch_shape + (6,))

        def eye6():
            return math.factory.eye(batch_shape + (6,))

        def expand_to_match(vec, reference):
            expanded = math.expand_dims(vec, axis=-1)
            expanded_ndim = expanded.ndim
            reference_ndim = reference.ndim
            if expanded_ndim != reference_ndim:
                expanded = math.expand_dims(expanded, axis=-1)
            return expanded

        def tile_batch(arr):
            if not batch_shape:
                return arr
            reps = batch_shape + (1,) * len(arr.shape)
            return math.tile(arr, reps)

        Xup = [None] * node_count
        Scols: list[ArrayLike | None] = [None] * node_count
        v = [None] * node_count
        c = [None] * node_count
        IA = [None] * node_count
        pA = [None] * node_count
        g_acc = [None] * node_count

        for idx in node_indices:
            IA[idx] = tile_batch(inertias[idx])

            if idx == root_idx:
                Xup[idx] = eye6()
                v[idx] = base_velocity_body
                c[idx] = zeros6()
                g_acc[idx] = a0_input
            else:
                parent = parent_indices[idx]
                joint = joints[idx]
                joint_idx = joint_indices[idx]
                vel_joint_idx = vel_joint_indices[idx]
                dof_count = joint_dofs[idx]
                if joint_idx is not None:
                    q_i = joint_positions[..., joint_idx]
                else:
                    q_i = self._zeros_for_dofs(
                        dof_count, batch_shape, reference=joint_positions
                    )
                X_current = joint.spatial_transform(q=q_i)
                Xup[idx] = X_current

                g_acc[idx] = math.mxv(X_current, g_acc[parent])

                # Only compute motion for joints that are in the selected list
                # (vel_joint_idx is not None) and have non-zero DOFs
                if joint is not None and vel_joint_idx is not None:
                    S_i = joint.motion_subspace(q_i)
                    if S_i.shape[-1] > 0:
                        Si = tile_batch(S_i)
                        Scols[idx] = Si
                        qd_i = joint_velocities[..., vel_joint_idx]
                        qd_vec = self._ensure_vector(
                            qd_i, dof_count, batch_shape
                        )
                        vJ = math.mxv(Si, qd_vec)
                    else:
                        Scols[idx] = None
                        vJ = zeros6()
                else:
                    Scols[idx] = None
                    vJ = zeros6()

                v[idx] = math.mxv(X_current, v[parent]) + vJ
                c[idx] = math.mxv(math.spatial_skew(v[idx]), vJ)

            pA[idx] = math.mxv(
                math.spatial_skew_star(v[idx]), math.mxv(IA[idx], v[idx])
            )

        d_list: list[ArrayLike | None] = [None] * node_count
        inv_d_list: list[ArrayLike | None] = [None] * node_count
        u_list: list[ArrayLike | None] = [None] * node_count
        U_list: list[ArrayLike | None] = [None] * node_count

        for idx in rev_indices:
            if idx == root_idx:
                continue

            parent = parent_indices[idx]
            Xpt = math.swapaxes(Xup[idx], -2, -1)
            Si = Scols[idx]
            if Scols[idx] is not None:
                U_i = math.mtimes(IA[idx], Si)
                d_i = math.mtimes(math.swapaxes(Si, -2, -1), U_i)
                vel_joint_idx = vel_joint_indices[idx]
                dof_count = joint_dofs[idx]
                if vel_joint_idx is not None:
                    tau_vec = joint_torques_eff[..., vel_joint_idx]
                    tau_vec = self._ensure_vector(tau_vec, dof_count, batch_shape)
                else:
                    tau_vec = self._zeros_for_dofs(
                        dof_count, batch_shape, reference=joint_torques_eff
                    )
                Si_T_pA = math.mxv(math.swapaxes(Si, -2, -1), pA[idx])
                u_i = tau_vec - Si_T_pA
                u_i = self._ensure_vector(u_i, dof_count, batch_shape)

                d_list[idx] = d_i
                u_list[idx] = u_i
                U_list[idx] = U_i

                inv_d = math.inv(d_i)
                inv_d_list[idx] = inv_d
                Ia = IA[idx] - math.mtimes(
                    U_i, math.mtimes(inv_d, math.swapaxes(U_i, -2, -1))
                )
                gain = math.mtimes(inv_d, expand_to_match(u_i, inv_d))
                if self._is_casadi_math:
                    # we need to preserve the extra dimension for casadi
                    gain_vec = gain
                else:
                    # for other backends, we can squeeze the last dimension
                    gain_vec = gain[..., 0]
                pa = pA[idx] + math.mxv(Ia, c[idx]) + math.mxv(U_i, gain_vec)
            else:
                Ia = IA[idx]
                pa = pA[idx] + math.mxv(Ia, c[idx])

            IA[parent] = IA[parent] + math.mtimes(math.mtimes(Xpt, Ia), Xup[idx])
            pA[parent] = pA[parent] + math.mxv(Xpt, pa)

        rhs_root = base_ext_body - pA[root_idx] + math.mxv(IA[root_idx], a0_input)
        a_base = math.solve(IA[root_idx], rhs_root)

        a = [None] * node_count
        a[root_idx] = a_base

        qdd_entries: list[ArrayLike | None] = [None] * n if n > 0 else []

        for idx in node_indices:
            if idx == root_idx:
                continue

            parent = parent_indices[idx]
            a_pre = math.mxv(Xup[idx], a[parent]) + c[idx]
            free_acc = g_acc[idx]
            rel_acc = a_pre - free_acc if free_acc is not None else a_pre

            Si = Scols[idx]
            vel_joint_idx = vel_joint_indices[idx]
            dof_count = joint_dofs[idx]

            if Si is not None and vel_joint_idx is not None:
                U_i = U_list[idx]
                U_T_rel_acc = math.mxv(math.swapaxes(U_i, -2, -1), rel_acc)
                num = u_list[idx] - U_T_rel_acc
                inv_d = inv_d_list[idx]
                num_expanded = expand_to_match(num, inv_d)
                qdd_vec = math.mtimes(inv_d, num_expanded)
                qdd_vector = qdd_vec[..., :, 0]
                for local, global_idx in enumerate(
                    self._expand_joint_indices(vel_joint_idx)
                ):
                    if global_idx < n:
                        qdd_entries[global_idx] = qdd_vec[
                            ..., local : local + 1, 0
                        ]
                a_correction_vec = math.mxv(Si, qdd_vector)
                a[idx] = a_pre + a_correction_vec
            else:
                a[idx] = a_pre

        if n > 0:
            zero_col = math.factory.zeros(batch_shape + (1,))
            qdd_cols = [
                entry if entry is not None else zero_col for entry in qdd_entries
            ]
            joint_qdd = math.concatenate(qdd_cols, axis=-1)
        else:
            joint_qdd = math.factory.zeros(batch_shape + (0,))

        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            Xm = math.adjoint_mixed(base_transform)
            base_vel_mixed = math.mxv(Xm, base_velocity_body)
            Xm_dot = math.adjoint_mixed_derivative(base_transform, base_vel_mixed)
            base_acc = math.mxv(Xm, a_base) + math.mxv(Xm_dot, base_velocity_body)
        else:
            base_acc = a_base

        return math.concatenate([base_acc, joint_qdd], axis=-1)

    def _convert_to_arraylike(self, *args):
        """Convert inputs to ArrayLike if they are not already.
        Args:
            *args: Input arguments to be converted.
        Returns:
            Converted arguments as ArrayLike.
        """
        if not args:
            raise ValueError("At least one argument is required")

        converted = []
        for arg in args:
            if isinstance(arg, ArrayLike):
                converted.append(arg)
            else:
                converted.append(self.math.asarray(arg))
        return converted[0] if len(converted) == 1 else converted

    @staticmethod
    def _expand_joint_indices(
        joint_idx: int | tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if joint_idx is None:
            return ()
        if isinstance(joint_idx, int):
            return (joint_idx,)
        return tuple(joint_idx)

    def _zeros_for_dofs(
        self,
        dof_count: int,
        batch_shape: tuple[int, ...],
        reference: npt.ArrayLike | None = None,
    ):
        """
        Returns a zero vector with shape matching the provided dof_count and batch_shape.
        If reference is provided and dof_count==1, it is used to preserve dtype/device.
        """
        if dof_count == 0:
            return self.math.factory.zeros(batch_shape + ())
        if dof_count == 1:
            if reference is not None:
                try:
                    return self.math.zeros_like(reference[..., 0])
                except Exception:
                    return self.math.zeros_like(reference)
            return self.math.factory.zeros(batch_shape + ())
        return self.math.factory.zeros(batch_shape + (dof_count,))

    def _ensure_vector(
        self, vec: npt.ArrayLike, dof_count: int, batch_shape: tuple[int, ...]
    ) -> npt.ArrayLike:
        """Ensure joint coordinate vectors have an explicit trailing dimension."""
        arr = getattr(vec, "array", None)
        # If CasADi/ArrayAPI row vector (1, dof) was produced, transpose to column
        if arr is not None and len(getattr(arr, "shape", ())) == 2:
            r, c = arr.shape
            if r == 1 and c == dof_count:
                return self.math.factory.asarray(arr.T)
            if c == 1 and r == dof_count:
                return vec
        if dof_count == 0:
            return vec
        # If vec has no trailing dimension, add one
        if getattr(vec, "shape", ()) == batch_shape:
            return self.math.expand_dims(vec, axis=-1)
        if vec.shape and vec.shape[-1] == dof_count:
            return vec
        return vec

    def _prepare_tree_cache(self) -> None:
        """Pre-compute static tree data so the dynamic algorithms avoid repeated Python work."""
        nodes = list(self.model.tree)
        node_count = len(nodes)
        self._node_indices = tuple(range(node_count))
        self._rev_node_indices = tuple(reversed(self._node_indices))
        self._parent_indices = [-1] * node_count
        self._joint_indices_per_node: list[int | tuple[int, ...] | None] = [
            None
        ] * node_count
        self._vel_joint_indices_per_node: list[int | tuple[int, ...] | None] = [
            None
        ] * node_count
        self._spatial_inertias: list[npt.ArrayLike] = [None] * node_count
        self._joints_per_node: list[Joint | None] = [None] * node_count
        self._joint_index_to_node: dict[int, int] = {}
        self._dofs_per_node: list[int] = [0] * node_count

        for idx, node in enumerate(nodes):
            link, joint, parent_link = node.get_elements()
            self._joints_per_node[idx] = joint
            self._spatial_inertias[idx] = link.spatial_inertia()
            parent_idx = (
                self.model.tree.get_idx_from_name(parent_link.name)
                if parent_link is not None
                else -1
            )
            self._parent_indices[idx] = parent_idx
            if joint is None:
                self._joint_indices_per_node[idx] = None
                self._vel_joint_indices_per_node[idx] = None
                self._dofs_per_node[idx] = self._root_motion_subspace.shape[-1]
            else:
                self._joint_indices_per_node[idx] = joint.idx
                # vel_idx for indexing into velocity arrays
                vel_idx = getattr(joint, "vel_idx", joint.idx)
                self._vel_joint_indices_per_node[idx] = vel_idx
                # If joint.idx is None, the joint is fixed (not in selected joint list)
                if joint.idx is None:
                    self._dofs_per_node[idx] = 0
                else:
                    self._dofs_per_node[idx] = getattr(joint, "dofs", 1)
                    for dof in self._expand_joint_indices(vel_idx):
                        self._joint_index_to_node[int(dof)] = idx

        self._root_index = self.model.tree.get_idx_from_name(self.root_link)
        self._node_count = node_count
