#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGGT 位姿评估（Sim(3)对齐 + RRA@5↑ RTA@5↑ ATE↓ + 推理时间）
"""

import os, glob, time, argparse
import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ---------- 工具 ----------
def quat_to_R(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    qw, qx, qy, qz = q
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy+zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),   1 - 2*(xx+zz),   2*(yz - wx)],
        [2*(xz - wy),   2*(yz + wx),     1 - 2*(xx+yy)]
    ], dtype=np.float64)

def se3_inv(T):
    R, t = T[:3,:3], T[:3,3]
    Ri = R.T
    ti = -Ri @ t
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = Ri
    Ti[:3,3] = ti
    return Ti

def rot_err_deg(R_pred, R_gt):
    dR = R_pred @ R_gt.T
    cos = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    return float(np.degrees(np.abs(np.arccos(cos))))

def umeyama_alignment(X, Y):
    mu_X, mu_Y = X.mean(0), Y.mean(0)
    Xc, Yc = X - mu_X, Y - mu_Y
    cov = (Yc.T @ Xc) / X.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_X = (Xc**2).sum() / X.shape[0]
    s = np.trace(np.diag(S) @ D) / var_X
    t = mu_Y - s * R @ mu_X
    return s, R, t


# ---------- 读取GT ----------
def load_gt_c2w(txt_path, convention="w2c"):
    gt = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ss = s.split()
            if len(ss) != 8:
                continue
            name = os.path.basename(ss[0])
            qw, qx, qy, qz, tx, ty, tz = map(float, ss[1:])
            R = quat_to_R(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            T = np.eye(4)
            T[:3,:3] = R
            T[:3,3] = t
            if convention == "w2c":
                T = se3_inv(T)
            gt[name] = T
    print(f"载入 {len(gt)} 条 GT 位姿（统一为 cam→world）")
    return gt


# ---------- 主函数 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--gt_convention", default="w2c", choices=["w2c", "c2w"])
    ap.add_argument("--print_per_image", action="store_true")
    args = ap.parse_args()

    img_dir = os.path.join(args.data_dir, "images")
    pose_path = os.path.join(args.data_dir, "pose.txt")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"未找到 {img_dir}")
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"未找到 {pose_path}")

    image_paths = sorted(sum([glob.glob(os.path.join(img_dir, ext))
                              for ext in ("*.png","*.jpg","*.jpeg")], []))
    names = [os.path.basename(p) for p in image_paths]
    gt_c2w = load_gt_c2w(pose_path, args.gt_convention)

    # 模型加载
    ckpt_path = "/home/wentaocheng/Documents/model_zoo/model_tracker_fixed_e20.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    model = VGGT().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # 图像预处理
    images = load_and_preprocess_images(image_paths).to(device)
    if images.dim() == 5:
        images = images.squeeze(0)
    S, C, H, W = images.shape
    patch = model.aggregator.patch_size
    model.update_patch_dimensions(W // patch, H // patch)

    # 推理计时（GPU上）
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        preds = model(images)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    infer_time = t1 - t0
    print(f"\n✅ 推理耗时（GPU计时）: {infer_time:.3f} 秒 ({infer_time / len(images):.3f} 秒/帧)")

    # 提取姿态
    pose_enc = preds["pose_enc"].to(torch.float32)
    extrinsic_w2c, _ = pose_encoding_to_extri_intri(pose_enc, (H, W))
    extrinsic_w2c = extrinsic_w2c.to(torch.float32).cpu().numpy().squeeze(0)

    # ---- 计算预测pose并对齐 ----
    matched_names, pred_centers, gt_centers = [], [], []
    R_pred_list, R_gt_list = [], []

    for name, ext34 in zip(names, extrinsic_w2c):
        if name not in gt_c2w:
            continue
        T_pred_w2c = np.eye(4)
        T_pred_w2c[:3, :3] = ext34[:, :3]
        T_pred_w2c[:3, 3]  = ext34[:, 3]
        T_pred_c2w = se3_inv(T_pred_w2c)

        matched_names.append(name)
        pred_centers.append(T_pred_c2w[:3, 3])
        gt_centers.append(gt_c2w[name][:3, 3])
        R_pred_list.append(T_pred_c2w[:3, :3])
        R_gt_list.append(gt_c2w[name][:3, :3])

    if not matched_names:
        print("⚠️ 没有匹配到GT文件名。")
        return

    X = np.stack(pred_centers)
    Y = np.stack(gt_centers)

    # ---- Sim(3) 对齐 ----
    s, R_sim, t_sim = umeyama_alignment(X, Y)
    X_aligned = (s * (R_sim @ X.T).T) + t_sim
    trans_errs = np.linalg.norm(X_aligned - Y, axis=1)

    rot_errs = []
    for R_pred, R_gt in zip(R_pred_list, R_gt_list):
        R_pred_aligned = R_sim @ R_pred
        rot_errs.append(rot_err_deg(R_pred_aligned, R_gt))
    rot_errs = np.array(rot_errs)

    # ---------- 计算指标 ----------
    ATE = np.mean(trans_errs)
    RRA5 = np.mean(rot_errs < 5.0) * 100.0
    base_dist = np.linalg.norm(Y - Y.mean(0), axis=1).mean() + 1e-8
    RTA5 = np.mean(trans_errs < 0.05 * base_dist) * 100.0

    # ---- 打印结果 ----
    print(f"\n=== Sim(3) 对齐后指标 ===")
    print(f"尺度因子 s = {s:.4f}")
    print(f"RRA@5°↑  = {RRA5:6.2f}%")
    print(f"RTA@5%↑  = {RTA5:6.2f}%")
    print(f"ATE↓     = {ATE:8.4f}")
    print(f"旋转误差(°): mean={rot_errs.mean():.2f}, median={np.median(rot_errs):.2f}")
    print(f"平移误差:   mean={trans_errs.mean():.4f}, median={np.median(trans_errs):.4f}, "
          f"p90={np.percentile(trans_errs,90):.4f}, max={np.max(trans_errs):.4f}")

    if args.print_per_image:
        for name, r, t_ in zip(matched_names, rot_errs, trans_errs):
            print(f"{name:30s}  RotErr={r:6.2f}°  TransErr={t_:8.4f}")


if __name__ == "__main__":
    main()
