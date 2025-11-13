import numpy as np


def se3_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ri = R.T
    ti = -Ri @ t
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = Ri
    Ti[:3, 3] = ti
    return Ti


def rot_err_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    dR = R_pred @ R_gt.T
    cos = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def umeyama_alignment(X: np.ndarray, Y: np.ndarray):
    """Similarity alignment (Sim(3)) of points X to Y.
    Returns scale s, rotation R, translation t, such that s*R*X + t ≈ Y.
    X, Y: (N,3)
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    mu_X, mu_Y = X.mean(0), Y.mean(0)
    Xc, Yc = X - mu_X, Y - mu_Y
    cov = (Yc.T @ Xc) / X.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_X = (Xc ** 2).sum() / X.shape[0]
    s = np.trace(np.diag(S) @ D) / max(var_X, 1e-12)
    t = mu_Y - s * (R @ mu_X)
    return s, R, t


def align_poses_sim3(T_c2w_pred: np.ndarray, T_c2w_gt: np.ndarray):
    """Align a sequence of predicted camera poses to GT using Sim(3).

    Inputs:
      - T_c2w_pred: (N,4,4) predicted camera-to-world poses
      - T_c2w_gt:   (N,4,4) ground truth camera-to-world poses
    Returns:
      - T_c2w_pred_aligned: (N,4,4) aligned poses
      - (s, R, t): similarity params with X' = s*R*X + t
    """
    assert T_c2w_pred.shape == T_c2w_gt.shape
    N = T_c2w_pred.shape[0]
    X = T_c2w_pred[:, :3, 3]
    Y = T_c2w_gt[:, :3, 3]
    s, R, t = umeyama_alignment(X, Y)

    T_c2w_aligned = T_c2w_pred.copy()
    # Apply similarity to both rotation and translation: R' = R_sim * R_pred, t' = s*R_sim*t_pred + t
    for i in range(N):
        Rp = T_c2w_aligned[i, :3, :3]
        tp = T_c2w_aligned[i, :3, 3]
        T_c2w_aligned[i, :3, :3] = R @ Rp
        T_c2w_aligned[i, :3, 3] = s * (R @ tp) + t
    return T_c2w_aligned, (s, R, t)


def compute_ate(trans_errs: np.ndarray, rmse: bool = True) -> float:
    if rmse:
        return float(np.sqrt(np.mean(trans_errs ** 2)))
    else:
        return float(np.mean(trans_errs))


def rpe_between(T1: np.ndarray, T2: np.ndarray) -> tuple[float, float]:
    """RPE between two relative motions (SE3):
    error = inv(T_rel_gt) * T_rel_pred
    Returns (trans_error, rot_error_deg)
    """
    T_err = se3_inv(T1) @ T2
    t_err = np.linalg.norm(T_err[:3, 3])
    r_err = rot_err_deg(T_err[:3, :3], np.eye(3))
    return float(t_err), float(r_err)


def compute_rpe(T_c2w_gt: np.ndarray, T_c2w_pred: np.ndarray, delta: int = 1):
    """Compute RPE (trans/rot) at step=delta along the trajectory.
    Returns mean translational and rotational RPE.
    """
    assert T_c2w_gt.shape == T_c2w_pred.shape
    N = T_c2w_gt.shape[0]
    trans_errs, rot_errs = [], []
    for i in range(N - delta):
        Ti_gt0, Ti_gt1 = T_c2w_gt[i], T_c2w_gt[i + delta]
        Ti_pr0, Ti_pr1 = T_c2w_pred[i], T_c2w_pred[i + delta]

        T_rel_gt = se3_inv(Ti_gt0) @ Ti_gt1
        T_rel_pr = se3_inv(Ti_pr0) @ Ti_pr1

        t_e, r_e = rpe_between(T_rel_gt, T_rel_pr)
        trans_errs.append(t_e)
        rot_errs.append(r_e)

    return float(np.mean(trans_errs)), float(np.mean(rot_errs))

