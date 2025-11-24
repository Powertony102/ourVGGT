import argparse
import os
import os.path as osp
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(str(REPO_ROOT / "eval"))

from eval.mv_recon.data import NRGBD
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from eval.pose.metrics import (
    align_poses_sim3,
    compute_ate,
    compute_rpe,
)


def get_args_parser():
    parser = argparse.ArgumentParser("Camera Pose evaluation", add_help=False)
    parser.add_argument("--weights", type=str, default="", help="ckpt path")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="vggt")
    parser.add_argument("--output_dir", type=str, default="eval_pose_out")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--delta", type=int, default=1, help="RPE step")
    return parser


def main(args):
    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset: NRGBD sequences
    dataset = NRGBD(
        split="test",
        ROOT=args.dataset_root,
        resolution=resolution,
        num_seq=1,
        full_video=True,
        kf_every=20,
    )

    # Model
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model = VGGT().to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Use float16 on CUDA; else float32
    if device != "cpu" and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    log_file = osp.join(args.output_dir, "logs.txt")
    fps_all = []
    ate_all = []
    rpe_t_all = []
    rpe_r_all = []

    for data_idx in tqdm(range(len(dataset)), desc="NRGBD sequences"):
        # Prepare batch (list of views)
        batch = default_collate([dataset[data_idx]])
        try:
            scene_id_dbg = str(batch[0]["label"]).rsplit("/", 1)[0]
        except Exception:
            scene_id_dbg = f"seq_{data_idx}"
        print(f"[Seq {data_idx}] scene_id={scene_id_dbg}; batch_len={len(batch)}")

        # Move tensors to device
        ignore_keys = {
            "depthmap",
            "dataset",
            "label",
            "instance",
            "idx",
            "true_shape",
            "rng",
            "img_mask",
            "ray_mask",
            "ray_map",
            "update",
            "reset",
            "valid_mask",
            "pts3d",
            # keep GT pose/intrinsics on CPU to avoid cuda->numpy conversion issues
            "camera_pose",
            "camera_intrinsics",
        }
        for view in batch:
            for name in list(view.keys()):
                if name in ignore_keys:
                    continue
                val = view[name]
                if isinstance(val, (list, tuple)):
                    view[name] = [x.to(device, non_blocking=True) for x in val]
                elif hasattr(val, "to"):
                    view[name] = val.to(device, non_blocking=True)

        # Build image tensor [1, S, C, H, W]
        images_seq = torch.stack(
            [view["img"][0] if view["img"].dim() == 4 else view["img"] for view in batch],
            dim=0,
        )
        images_seq = (images_seq + 1.0) / 2.0  # undo ImgNorm
        images_seq = images_seq.unsqueeze(0)
        print(f"[Seq {data_idx}] images_seq.shape={tuple(images_seq.shape)} (B,S,C,H,W)")

        # Update aggregator patch dims
        _, _, _, H, W = images_seq.shape
        patch_size = model.aggregator.patch_size
        model.update_patch_dimensions(W // patch_size, H // patch_size)

        # Forward
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=(device != "cpu")):
            predictions = model(images_seq)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        # Extract predicted extrinsics and convert to c2w
        pose_enc = predictions["pose_enc"].to(torch.float32)[0]
        seq_len_pred = pose_enc.shape[0]
        seq_len_in = len(batch)
        if seq_len_pred <= 0:
            print("[Warn] No predicted poses returned; skipping sequence.")
            continue
        print(f"[Seq {data_idx}] pose_enc.shape={tuple(pose_enc.shape)}; seq_len_pred={seq_len_pred}; seq_len_in={seq_len_in}")
        assert seq_len_pred == seq_len_in, (
            f"Prediction length ({seq_len_pred}) != input length ({seq_len_in}). This indicates a mismatch; aborting this sequence."
        )
        # Use all frames as-is; no temporal re-alignment
        extrinsic_w2c, _ = pose_encoding_to_extri_intri(pose_enc.unsqueeze(0), (H, W))
        extrinsic_w2c = extrinsic_w2c[0].to(torch.float32).cpu().numpy()  # (S_pred,3,4)
        print(f"[Seq {data_idx}] extrinsic_w2c.shape={tuple(extrinsic_w2c.shape)} (S,3,4)")

        T_c2w_pred = []
        for ext in extrinsic_w2c:
            T_w2c = np.eye(4, dtype=np.float32)
            T_w2c[:3, :3] = ext[:, :3]
            T_w2c[:3, 3] = ext[:, 3]
            # invert to get c2w
            R = T_w2c[:3, :3]
            t = T_w2c[:3, 3]
            Ri = R.T
            ti = -Ri @ t
            T_c2w = np.eye(4, dtype=np.float32)
            T_c2w[:3, :3] = Ri
            T_c2w[:3, 3] = ti
            T_c2w_pred.append(T_c2w)
        T_c2w_pred = np.stack(T_c2w_pred, axis=0)  # (S_pred,4,4)
        print(f"[Seq {data_idx}] T_c2w_pred.shape={tuple(T_c2w_pred.shape)}")

        # Collect GT c2w from batch (ensure numpy on CPU)
        T_c2w_gt = []
        for vi, view in enumerate(batch):
            T = view["camera_pose"]
            try:
                import torch as _torch
                if isinstance(T, _torch.Tensor):
                    T = T.detach().cpu().numpy()
            except Exception:
                pass
            T = np.asarray(T)
            if T.ndim == 3 and T.shape[0] == 1:
                T = T[0]
            if T.shape != (4, 4):
                print(f"[Seq {data_idx}] warn: unexpected GT pose shape at view {vi}: {T.shape}, trying reshape")
                T = T.reshape(4, 4)
            T_c2w_gt.append(T.astype(np.float32))
        T_c2w_gt = np.stack(T_c2w_gt, axis=0).astype(np.float32)
        print(f"[Seq {data_idx}] T_c2w_gt.shape={tuple(T_c2w_gt.shape)}")

        # FPS measured on full sequence
        fps = seq_len_in / max(1e-6, (t1 - t0))
        fps_all.append(fps)
        # Strict check: pred and gt must already be aligned
        assert T_c2w_pred.shape[0] == T_c2w_gt.shape[0], (
            f"Pred/GT frame count mismatch after load. pred={T_c2w_pred.shape[0]}, gt={T_c2w_gt.shape[0]}"
        )

        # Align with Sim(3)
        T_c2w_pred_aligned, (s, _, _) = align_poses_sim3(T_c2w_pred, T_c2w_gt)

        # Metrics
        X_aligned = T_c2w_pred_aligned[:, :3, 3]
        Y = T_c2w_gt[:, :3, 3]
        trans_errs = np.linalg.norm(X_aligned - Y, axis=1)
        ate = compute_ate(trans_errs, rmse=False)
        rpe_t, rpe_r = compute_rpe(T_c2w_gt, T_c2w_pred_aligned, delta=args.delta)
        # Optional: report average GT step to interpret RPE_t scale
        gt_steps = np.linalg.norm(T_c2w_gt[1:, :3, 3] - T_c2w_gt[:-1, :3, 3], axis=1)
        mean_step = float(gt_steps.mean()) if gt_steps.size else 0.0

        scene_id = str(batch[0]["label"]).rsplit("/", 1)[0]
        log_line = (
            f"Idx: {scene_id}, ATE: {ate:.6f}, RPE_t: {rpe_t:.6f}, RPE_r: {rpe_r:.6f}, "
            f"Scale: {float(s):.6f}, FPS: {fps:.2f}, mean_step: {mean_step:.4f}"
        )
        print(log_line)
        with open(log_file, "a", encoding="utf-8") as f:
            print(log_line, file=f)

        ate_all.append(ate)
        rpe_t_all.append(rpe_t)
        rpe_r_all.append(rpe_r)

        # release cuda memory between sequences
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(ate_all):
        summary = (
            f"Mean: ATE={np.mean(ate_all):.6f}, RPE_t={np.mean(rpe_t_all):.6f}, RPE_r={np.mean(rpe_r_all):.6f}, FPS={np.mean(fps_all):.2f}"
        )
        print(summary)
        with open(log_file, "a", encoding="utf-8") as f:
            print(summary, file=f)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
