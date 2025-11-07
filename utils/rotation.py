"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Dict, Union

import numpy as np
import torch as th


rowcache: Dict[th.device, th.Tensor] = {}
rowcache_2x3: Dict[th.device, th.Tensor] = {}
eps = 1e-8


def rrot(
    pts: th.Tensor, rvec: th.Tensor, dim: int = -1, safe: bool = True, _eps: float = eps
) -> th.Tensor:
    """Applies the rotations defined by a tensor of Rodrigues vectors `rvec` to
    a set of points `pts` along dimension `dim`."""

    theta = th.norm(rvec, dim=dim, keepdim=True).to(rvec.dtype, non_blocking=True)
    costheta = th.cos(theta)
    sintheta = th.sin(theta)
    theta = theta.expand_as(rvec)

    # Convert the global eps to a local variable so we can TorchScript this
    # function.
    good = theta > _eps
    if safe:
        vec = th.empty_like(rvec).requires_grad_(False)
        vec[good] = rvec[good] / theta[good]
    else:
        vec = rvec / theta.clamp(min=_eps)

    dot = (pts * vec).sum(dim=dim, keepdim=True).to(pts.dtype, non_blocking=True)

    vecb, rvecb, ptsb = th.broadcast_tensors(vec, rvec, pts)

    good_rot = (
        costheta * pts + (1 - costheta) * dot * vec + sintheta * th.cross(vecb, ptsb, dim=dim)
    )
    bad_rot = pts + th.cross(rvecb, ptsb, dim=dim)
    return th.where(good, good_rot, bad_rot)


# Rodrigues Vectors
def rvec_to_R(rvec: th.Tensor) -> th.Tensor:
    """Computes the rotation matrix R from a tensor of Rodrigues vectors.

    n = ||rvec||
    rn = rvec/||rvec||
    N = [rn]_x = [[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]]
    R = I + sin(n)*N + (1-cos(n))*N*N
    """
    n = rvec.norm(dim=-1, p=2).clamp(min=1e-6)[..., None, None]
    rn = rvec / n[..., :, 0]
    zero = th.zeros_like(n[..., 0, 0])
    N = th.stack(
        (
            zero,
            -rn[..., 2],
            rn[..., 1],
            rn[..., 2],
            zero,
            -rn[..., 0],
            -rn[..., 1],
            rn[..., 0],
            zero,
        ),
        -1,
    ).view(rvec.shape[:-1] + (3, 3))
    R = (
        th.eye(3, dtype=n.dtype, device=n.device).view([1] * (rvec.dim() - 1) + [3, 3])
        + th.sin(n) * N
        + ((1 - th.cos(n)) * N) @ N
    )
    return R


def pad_3x4_to_4x4(mat: th.Tensor) -> th.Tensor:
    """Pads a 3x4 pose matrix into a 4x4 with 1 in the bottom right entry."""
    # mat: Bx3x4
    assert mat.shape[-2:] == (3, 4)
    if mat.device not in rowcache:
        rowcache[mat.device] = th.tensor([[0, 0, 0, 1]], dtype=mat.dtype, device=mat.device)

    row = (
        rowcache[mat.device]
        .view([1] * len(mat.shape[:-2]) + [1, 4])
        .expand(list(mat.shape[:-2]) + [-1, -1])
    )
    return th.cat([mat, row], dim=-2)


def R_to_rvec(R: th.Tensor) -> th.Tensor:
    """Computes the Rodrigues vectors rvec from the rotation matrices `R`."""
    return qvec_to_rvec(R_to_qvec(R))


def qvec_to_rvec(qvec: th.Tensor, dim: int = -1) -> th.Tensor:
    """Converts a tensor of quaternions [wxyz] to a tensor of Rodrigues
    vectors."""
    a, bcd = qvec.narrow(dim, 0, 1), qvec.narrow(dim, 1, 3)
    norm = th.norm(bcd, dim=dim, keepdim=True, p=2)
    theta = 2 * th.atan2(norm, a)
    return theta * (bcd / (norm + 1e-6))


def R_to_qvec(R: th.Tensor, _eps: float = eps) -> th.Tensor:
    """Converts rotation matrices to quaternions [wxyz].

    See: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
    """
    assert R.shape[-2:] == (3, 3), "R must be a [..., 3, 3] th.Tensor"
    orig_shp = R.shape
    R = R.view(-1, 3, 3)

    rmat_t = th.transpose(R, -2, -1)

    mask_d2 = rmat_t[:, 2, 2] < _eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = th.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = th.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = th.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = th.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).T

    mask_c0 = mask_d2 & mask_d0_d1
    mask_c1 = mask_d2 & (~mask_d0_d1)
    mask_c2 = (~mask_d2) & mask_d0_nd1
    mask_c3 = (~mask_d2) & (~mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= th.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3)
    q *= 0.5
    return q.view(orig_shp[:-2] + (4,))


#
# Quaternions
#
def qmult(qa: th.Tensor, qb: th.Tensor, dim: int = -1) -> th.Tensor:
    """Performs quaternion multiplication qa x qb. Quaternion order is [wxyz]."""
    assert qa.shape[dim] == 4, f"Quaternions must have shape 4 at dim {dim}."
    assert qb.shape[dim] == 4, f"Quaternions must have shape 4 at dim {dim}."

    wa, xa, ya, za = qa.split(1, dim=dim)
    wb, xb, yb, zb = qb.split(1, dim=dim)

    return th.cat(
        [
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ],
        dim=dim,
    )


def qconj(q: th.Tensor, dim: int = -1) -> th.Tensor:
    """Computes the conjugate of a quaternion[wxyz]."""
    assert q.shape[dim] == 4, f"Quaternions must have shape 4 at dim {dim}."
    coeff_shape = [1] * q.dim()
    coeff_shape[dim] = 4
    coeffs = th.tensor([1, -1, -1, -1], dtype=q.dtype, device=q.device).view(coeff_shape)
    return q * coeffs


def qvec_to_R(qvec: th.Tensor, dim: int = -1) -> th.Tensor:
    """Converts quaternions [wxyz] to rotation matrices."""
    assert qvec.shape[dim] == 4, f"Quaternions must have shape 4 at dim {dim}."

    w, x, y, z = qvec.split(1, dim=dim)
    n = qvec.norm(dim=dim, keepdim=True)
    s = 2 / (n**2).clamp(min=1e-6)
    x2 = x**2
    y2 = y**2
    z2 = z**2

    R = th.cat(
        [
            1 - s * (y2 + z2),
            s * (x * y - z * w),
            s * (x * z + y * w),
            s * (x * y + z * w),
            1 - s * (x2 + z2),
            s * (y * z - x * w),
            s * (x * z - y * w),
            s * (y * z + x * w),
            1 - s * (x2 + y2),
        ],
        dim=dim,
    )

    if dim == -1:
        dim = qvec.dim() - 1
    new_shp = list(qvec.shape)
    new_shp[dim : dim + 1] = [3, 3]
    return R.view(new_shp)


#
# Dual Quaternions
#
def dq_rotation(dq: th.Tensor) -> th.Tensor:
    """Dual quaternion (qr[wxyz], qd[wxyz]) rotation = qr"""
    return dq[:, 0]


def dq_translation(dq: th.Tensor) -> th.Tensor:
    """Dual quaternion (qr[wxyz], qd[wxyz]) translation = 2 qd qr*"""
    return 2 * qmult(dq[:, 1], qconj(dq[:, 0]))[:, 1:4]


def dq_apply_transform(dq: th.Tensor, p: th.Tensor) -> th.Tensor:
    """A dual quaternion transformation can be expressed
    as the quaternion transform for the rotation, to
    which is added the dual quaternion translation:

    dq x p x dq_dual_quat_conjugate = q x p x q_conjugate + dq_translation
    """
    assert p.shape[-1] == 3, "Points must have shape [..., 3]."

    q_r = dq[:, 0:1, 0:1]
    q_v = dq[:, 0:1, 1:4]
    qxp_r = -(q_v * p).sum(dim=2, keepdim=True)
    qxp_v = q_r * p + th.cross(q_v.expand_as(p), p)
    qc_r = q_r
    qc_v = -q_v
    qxpxqc_v = qxp_r * qc_v + qc_r * qxp_v + th.cross(qxp_v, qc_v.expand_as(p))
    return qxpxqc_v + dq_translation(dq)[:, None]


def dq_normalize(dq: th.Tensor) -> th.Tensor:
    """Normalize a dual quaternion by the rotation norm."""
    return dq / dq[:, 0:1].norm(dim=2, keepdim=True).clamp(min=1e-12)


def qnlerp(
    q0: th.Tensor, q1: th.Tensor, t: Union[float, int], dim: int = -1, _eps: float = eps
) -> th.Tensor:
    """Normalized linear interpolation of quaternions."""
    q = q0 + t * (q1 - q0)
    q = q / q.norm(dim=dim, keepdim=True).clamp(min=_eps)
    return q


def qslerp(
    q0: th.Tensor, q1: th.Tensor, t: Union[float, int], dim: int = -1, _eps: float = eps
) -> th.Tensor:
    """Performs spherical linear interpolation of two quaternions. Quaternions
    should be normalized. Does not check for symmetries"""
    q_dot = th.sum(q0 * q1, dim=dim, keepdim=True).clamp(min=-1, max=1)
    close = q_dot > 0.9995

    theta_0 = th.acos(q_dot)
    sin_theta_0 = th.sin(theta_0)

    # Shoemake formulation. When the two quaternions are very similar,
    # sin(theta) approaches zero and the division becomes unstable so we fall
    # back to non-spherical interpolation.
    q = q0 * th.sin((1.0 - t) * theta_0) / sin_theta_0 + q1 * th.sin(t * theta_0) / sin_theta_0
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=_eps)
    return th.where(close, qnlerp(q0, q1, t, dim), q)


def get_rotate_Rt(theta: float, center: np.ndarray, radius: np.ndarray, flip_up: bool = False):
    """Generates a camera matrix corresponding to a camera lying on a circle
    with a given radius pointing at the center, rotated by angle ``theta`` around
    the center.

    Args:
        theta: Angle by which to rotate the camera around ``center``.

        center: Center point about which to rotate.

        radius: Distance from the ``center``.

        flip_up: Whether or not the model uses a flipped Y-axis.

    Returns:
        np.array: 3x4 transformation matrix representing the camera pose.
    """

    x = radius * np.cos(theta)
    if flip_up:
        z = radius * np.sin(theta)
    else:
        z = -radius * np.sin(theta)
    pos = np.array(center) + np.array([x, 0, z])

    if flip_up:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    forward = center - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    R = np.array([right, up, forward], dtype=np.float32)
    t = -R.dot(pos)
    Rt = np.c_[R, t[:, None]]
    return Rt.astype(np.float32)
