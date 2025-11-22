import argparse
import os
import os.path as osp
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(str(REPO_ROOT / "eval"))
import numpy as np
import open3d as o3d
import torch
from accelerate import Accelerator
from dust3r.utils.camera import camera_to_pose_encoding
from dust3r.utils.geometry import geotrf
from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
from eval.mv_recon.data import SevenScenes, NRGBD
from eval.mv_recon.utils import accuracy, completion
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

SUBSAMPLE_RNG = np.random.default_rng(seed=42)
try:
    import cupoch as cph  # optional GPU-accelerated Open3D-like library
    HAS_CUPOCH = True
except Exception:
    cph = None
    HAS_CUPOCH = False


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    return parser


def main(args):
    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError
    datasets_all = {
        # "7scenes": SevenScenes(
        #     split="test",
        #     ROOT="/root/autodl-tmp/data/7-scenes",
        #     resolution=resolution,
        #     num_seq=1,
        #     full_video=True,
        #     kf_every=1,
        # ),  # 20),
        "NRGBD": NRGBD(
            split="test",
            ROOT="/root/autodl-tmp/data/neural_rgbd_data/",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=1,
        ),
    }

    accelerator = Accelerator()
    device = accelerator.device
    model_name = args.model_name
    # model = VGGT().to(device)
    model = VGGT(enable_point=True).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    if device.type != "cuda":
        raise RuntimeError(
            "VGGT evaluation requires CUDA so the aggregator can run in bfloat16."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device_type = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    device_type = torch.float16
    model.eval()
    model = model.to(device_type)
    os.makedirs(args.output_dir, exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            fps_all = []
            time_all = []

            with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                for data_idx in tqdm(idxs):
                    batch = default_collate([dataset[data_idx]])
                    ignore_keys = set(
                        [
                            "depthmap",
                            "dataset",
                            "label",
                            "instance",
                            "idx",
                            "true_shape",
                            "rng",
                        ]
                    )
                    for view in batch:
                        for name in view.keys():  # pseudo_focal
                            if name in ignore_keys:
                                continue
                            if isinstance(view[name], tuple) or isinstance(
                                view[name], list
                            ):
                                view[name] = [
                                    x.to(device, non_blocking=True) for x in view[name]
                                ]
                            else:
                                view[name] = view[name].to(device, non_blocking=True)

                    if model_name == "vggt":
                        revisit = max(1, args.revisit)
                        # VGGT expects images in [0, 1], so undo ImgNorm scaling.
                        images_seq = torch.stack(
                            [
                                view["img"][0] if view["img"].dim() == 4 else view["img"]
                                for view in batch
                            ],
                            dim=0,
                        )
                        images_seq = (images_seq + 1.0) / 2.0
                        images_seq = images_seq.unsqueeze(0)
                        if revisit > 1:
                            images_seq = images_seq.repeat(1, revisit, 1, 1, 1)

                        _, _, _, H, W = images_seq.shape
                        patch_size = model.aggregator.patch_size
                        model.update_patch_dimensions(W // patch_size, H // patch_size)

                        with torch.no_grad(), torch.cuda.amp.autocast(dtype=device_type):
                        # with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                # reset peak stats for a clean measurement of this forward
                                torch.cuda.reset_peak_memory_stats()
                            start = time.time()
                            predictions = model(images_seq)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            end = time.time()
                            # report peak memory for this forward
                            if torch.cuda.is_available():
                                peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
                                peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                                print(f"Peak GPU mem: alloc={peak_alloc:.2f} GB, reserved={peak_reserved:.2f} GB")

                        frames_to_keep = len(batch)
                        fps = frames_to_keep / (end - start)
                        print(f"frames_to_keep: {frames_to_keep}")
                        world_points = predictions["world_points"].to(torch.float32)[0]
                        world_points = world_points[-frames_to_keep:]
                        world_conf = predictions.get("world_points_conf")
                        if world_conf is not None:
                            world_conf = world_conf.to(torch.float32)[0]
                            world_conf = world_conf[-frames_to_keep:]
                        else:
                            world_conf = torch.ones(
                                world_points.shape[:-1],
                                dtype=world_points.dtype,
                                device=world_points.device,
                            )

                        pose_enc = predictions["pose_enc"].to(torch.float32)[0]
                        pose_enc = pose_enc[-frames_to_keep:]
                        extrinsic_w2c, _ = pose_encoding_to_extri_intri(
                            pose_enc.unsqueeze(0), (H, W)
                        )
                        extrinsic_w2c = extrinsic_w2c[0]
                        R_w2c = extrinsic_w2c[:, :, :3]
                        t_w2c = extrinsic_w2c[:, :, 3]

                        pts_cam = (
                            torch.einsum("sij,shwj->shwi", R_w2c, world_points)
                            + t_w2c[:, None, None, :]
                        )

                        R_c2w = R_w2c.transpose(1, 2)
                        t_c2w = -torch.einsum("sij,sj->si", R_c2w, t_w2c)
                        cam2world = (
                            torch.eye(4, device=R_c2w.device, dtype=R_c2w.dtype)
                            .unsqueeze(0)
                            .repeat(frames_to_keep, 1, 1)
                        )
                        cam2world[:, :3, :3] = R_c2w
                        cam2world[:, :3, 3] = t_c2w
                        cam_pose_enc = camera_to_pose_encoding(cam2world)

                        preds = []
                        for idx in range(frames_to_keep):
                            preds.append(
                                {
                                    "pts3d_in_self_view": pts_cam[idx].unsqueeze(0),
                                    "camera_pose": cam_pose_enc[idx].unsqueeze(0),
                                    "conf": world_conf[idx].unsqueeze(0),
                                }
                            )

                        print(
                            f"Finished reconstruction for {name_data} {data_idx+1}/{len(dataset)}, FPS: {fps:.2f}"
                        )
                        # continue
                        fps_all.append(fps)
                        time_all.append(end - start)

                        # Evaluation
                        print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                            criterion.get_all_pts3d_t(batch, preds)
                        )
                        pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
                            monitoring["pred_scale"],
                            monitoring["gt_scale"],
                            monitoring["pred_shift_z"],
                            monitoring["gt_shift_z"],
                        )

                        in_camera1 = None
                        pts_all = []
                        pts_gt_all = []
                        images_all = []
                        masks_all = []
                        conf_all = []

                        for j, view in enumerate(batch):
                            if in_camera1 is None:
                                in_camera1 = view["camera_pose"][0].cpu()

                            image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                            mask = view["valid_mask"].cpu().numpy()[0]

                            # pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
                            pts = pred_pts[j].cpu().numpy()[0]
                            conf = preds[j]["conf"].cpu().data.numpy()[0]
                            # mask = mask & (conf > 1.8)

                            pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                            H, W = image.shape[:2]
                            cx = W // 2
                            cy = H // 2
                            l, t = cx - 112, cy - 112
                            r, b = cx + 112, cy + 112
                            image = image[t:b, l:r]
                            mask = mask[t:b, l:r]
                            pts = pts[t:b, l:r]
                            pts_gt = pts_gt[t:b, l:r]

                            #### Align predicted 3D points to the ground truth
                            pts[..., -1] += gt_shift_z.cpu().numpy().item()
                            pts = geotrf(in_camera1, pts)

                            pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                            pts_gt = geotrf(in_camera1, pts_gt)

                            images_all.append((image[None, ...] + 1.0) / 2.0)
                            pts_all.append(pts[None, ...])
                            pts_gt_all.append(pts_gt[None, ...])
                            masks_all.append(mask[None, ...])
                            conf_all.append(conf[None, ...])

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)

                    scene_id = view["label"][0].rsplit("/", 1)[0]

                    if "DTU" in name_data:
                        threshold = 100
                    else:
                        threshold = 0.1

                    pts_all_masked = pts_all[masks_all > 0]
                    pts_gt_all_masked = pts_gt_all[masks_all > 0]
                    images_all_masked = images_all[masks_all > 0]
                    max_points = int(1e6)
                    cur_points = pts_all_masked.shape[0]
                    if cur_points > max_points:
                        # Randomly subsample to keep ICP computationally feasible
                        choice = SUBSAMPLE_RNG.choice(cur_points, max_points, replace=False)
                        pts_all_masked = pts_all_masked[choice]
                        pts_gt_all_masked = pts_gt_all_masked[choice]
                        images_all_masked = images_all_masked[choice]
                    # Log effective points used for ICP and mask coverage
                    eff_pred = int(pts_all_masked.shape[0])
                    eff_gt = int(pts_gt_all_masked.shape[0])
                    valid_px = int((masks_all > 0).sum())
                    total_px = int(np.prod(masks_all.shape))
                    ratio = valid_px / max(1, total_px)
                    print(f"[Eval] Effective points: pred={eff_pred}, gt={eff_gt}; valid_mask={valid_px}/{total_px} ({ratio:.2%})")

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(
                        pts_all_masked.reshape(-1, 3)
                    )
                    pcd.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(
                        pts_gt_all_masked.reshape(-1, 3)
                    )
                    pcd_gt.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )

                    trans_init = np.eye(4, dtype=np.float32)

                    # Prefer Cupoch (CUDA) ICP if available; fallback to legacy Open3D CPU ICP
                    if HAS_CUPOCH:
                        # Build Cupoch point clouds from numpy (float32)
                        src_pts = pts_all_masked.reshape(-1, 3).astype(np.float32)
                        tgt_pts = pts_gt_all_masked.reshape(-1, 3).astype(np.float32)
                        cpcd = cph.geometry.PointCloud()
                        cpcd.points = cph.utility.Vector3fVector(src_pts)
                        cpcd_gt = cph.geometry.PointCloud()
                        cpcd_gt.points = cph.utility.Vector3fVector(tgt_pts)
                        cuda_flag = getattr(cph.utility, "is_cuda_available", lambda: False)()
                        print(f"[ICP] Using Cupoch ICP (point-to-point){' [CUDA]' if cuda_flag else ''}")
                        # Cupoch API places registration under cph.registration (not pipelines)
                        reg = cph.registration.registration_icp(
                            cpcd,
                            cpcd_gt,
                            float(threshold),
                            trans_init,
                            cph.registration.TransformationEstimationPointToPoint(),
                        )
                        transformation = reg.transformation
                    else:
                        print("[ICP] Using legacy Open3D CPU ICP (point-to-point)")
                        reg_p2p = o3d.pipelines.registration.registration_icp(
                            pcd,
                            pcd_gt,
                            float(threshold),
                            trans_init,
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        )
                        transformation = reg_p2p.transformation

                    pcd = pcd.transform(transformation)
                    pcd.estimate_normals()
                    pcd_gt.estimate_normals()

                    gt_normal = np.asarray(pcd_gt.normals)
                    pred_normal = np.asarray(pcd.normals)

                    acc, acc_med, nc1, nc1_med = accuracy(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    comp, comp_med, nc2, nc2_med = completion(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    log_line = (
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} "
                        f"- Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}, FPS: {fps}"
                    )
                    print(log_line)
                    print(log_line, file=open(log_file, "a"))

                    acc_all += acc
                    comp_all += comp
                    nc1_all += nc1
                    nc2_all += nc2

                    acc_all_med += acc_med
                    comp_all_med += comp_med
                    nc1_all_med += nc1_med
                    nc2_all_med += nc2_med
                    
                    print(f"Dataset: {name_data}, Accuracy: {acc_all/len(fps_all)}, Completion: {comp_all/len(fps_all)}, NC1: {nc1_all/len(fps_all)}, NC2: {nc2_all/len(fps_all)} - Acc_med: {acc_all_med/len(fps_all)}, Comp_med: {comp_all_med/len(fps_all)}, NC1_med: {nc1_all_med/len(fps_all)}, NC2_med: {nc2_all_med/len(fps_all)}", file=open(log_file, "a"))
                    print(f"Average fps: {sum(fps_all) / len(fps_all)}, Average time: {sum(time_all) / len(time_all)}", file=open(log_file, "a"))

                    # release cuda memory
                    torch.cuda.empty_cache()

            accelerator.wait_for_everyone()
            # Get depth from pcd and run TSDFusion
            if accelerator.is_main_process:
                to_write = ""
                # Copy the error log from each process to the main error log
                for i in range(8):
                    if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                        break
                    with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                        to_write += f_sub.read()

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            # Exclude 'scene_id' from metrics as it's an identifier
                            for key, value in data.items():
                                if key != "scene_id":
                                    metrics[key].append(float(value))
                            metrics["nc"].append(
                                (float(data["nc1"]) + float(data["nc2"])) / 2
                            )
                            metrics["nc_med"].append(
                                (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                            )
                            metrics["fps"].append(float(data["fps"]))
                    mean_metrics = {
                        metric: sum(values) / len(values)
                        for metric, values in metrics.items()
                    }

                    c_name = "mean"
                    print_str = f"{c_name.ljust(20)}: "
                    for m_name in mean_metrics:
                        print_num = np.mean(mean_metrics[m_name])
                        print_str = print_str + f"{m_name}: {print_num:.3f} | "
                    print_str = print_str + "\n"
                    f.write(to_write + print_str)


from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+),\s*
    FPS:\s*(?P<fps>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
