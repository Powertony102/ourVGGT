import argparse
from pathlib import Path
import numpy as np
import torch
import time
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

CUT3R_DIR = os.path.abspath(os.path.join(ROOT_DIR, "CUT3R"))
CUT3R_SRC_DIR = os.path.join(CUT3R_DIR, "src")
for p in (CUT3R_DIR, CUT3R_SRC_DIR):
    if p not in sys.path:
        sys.path.append(p)

from vggt.utils.eval_utils import (
    load_poses,
    get_sorted_image_paths,
    get_all_scenes,
    build_frame_selection,
    evaluate_scene_and_save,
    compute_average_metrics_and_save,
)
from dust3r.model import ARCroco3DStereo
from dust3r.utils.image import load_images as cut3r_load_images
from dust3r.inference import loss_of_one_batch
from accelerate import Accelerator
from CUT3R.add_ckpt_path import add_path_to_dust3r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=Path, default="/home/jovyan/shared/xinzeli/scannetv2/process_scannet"
    )
    parser.add_argument(
        "--gt_ply_dir",
        type=Path,
        default="/home/jovyan/shared/xinzeli/scannetv2/scannet",
    )
    parser.add_argument("--output_path", type=Path, default="./eval_results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=3.0,
        help="Depth confidence threshold for filtering low confidence depth values",
    )
    parser.add_argument(
        "--chamfer_max_dist",
        type=float,
        default=0.5,
        help="Maximum distance threshold in Chamfer Distance computation, distances exceeding this value will be clipped",
    )
    parser.add_argument(
        "--input_frame",
        type=int,
        default=200,
        help="Maximum number of frames selected for processing per scene",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=50,
        help="Maximum number of scenes to evaluate",
    )
    parser.add_argument("--cut3r_ckpt", type=str, default="/home/jovyan/work/furina/ourVGGT/CUT3R/src/cut3r_512_dpt_4_64.pth")
    parser.add_argument(
        "--vis_attn_map",
        action="store_true",
        help="Whether to visualize attention maps during inference",
    )
    args = parser.parse_args()
    torch.manual_seed(33)
    np.random.seed(0)

    scannet_scenes = get_all_scenes(args.data_dir, args.num_scenes)
    print(f"Evaluate {len(scannet_scenes)} scenes")

    all_scenes_metrics = {"scenes": {}, "average": {}}
    from collections import defaultdict
    scene_infer_times = defaultdict(list)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device(args.device)
    if os.path.exists(args.cut3r_ckpt):
        add_path_to_dust3r(args.cut3r_ckpt)
    try:
        model = ARCroco3DStereo.from_pretrained(args.cut3r_ckpt)
    except Exception:
        raise
    model = model.to(device).eval()
    accelerator = Accelerator()

    for scene in scannet_scenes:
        scene_dir = args.data_dir / f"{scene}"
        output_scene_dir = args.output_path / f"input_frame_{args.input_frame}" / scene
        if (output_scene_dir / "metrics.json").exists():
            continue

        images_dir = scene_dir / "color"
        pose_path = scene_dir / "pose"
        image_paths = get_sorted_image_paths(images_dir)
        poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)
        if (
            poses_gt is None
            or first_gt_pose is None
            or available_pose_frame_ids is None
        ):
            print(f"Skipping scene {scene}: no pose data")
            continue

        selected_frame_ids, selected_image_paths, selected_pose_indices = (
            build_frame_selection(
                image_paths, available_pose_frame_ids, args.input_frame
            )
        )

        c2ws = poses_gt[selected_pose_indices]
        image_paths = selected_image_paths
        frame_ids = selected_frame_ids

        if len(image_paths) == 0:
            print(f"No images found in {images_dir}")
            continue

        print("🚩Processing", scene, f"Found {len(image_paths)} images")
        all_cam_to_world_mat = []
        all_world_points = []

        try:
            cut3r_views = cut3r_load_images([str(p) for p in image_paths], size=512, verbose=False)
            for v_idx, v in enumerate(cut3r_views):
                try:
                    B, C, H, W = v["img"].shape
                    if isinstance(v.get("true_shape", None), np.ndarray):
                        v["true_shape"] = torch.from_numpy(v["true_shape"])  
                    if "ray_map" not in v:
                        v["ray_map"] = torch.full((B, 6, H, W), torch.nan, dtype=v["img"].dtype)
                    v["img_mask"] = torch.tensor(True).unsqueeze(0)
                    v["ray_mask"] = torch.tensor(False).unsqueeze(0)
                    v["reset"] = torch.tensor(False).unsqueeze(0)
                    v["update"] = torch.tensor(True).unsqueeze(0)
                    v["camera_pose"] = torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(0)
                    v["idx"] = v_idx
                    v["instance"] = str(v_idx)
                    # 关键张量迁移到 GPU，避免与模型权重设备不一致
                    v["img"] = v["img"].to(device, non_blocking=True)
                    v["img_mask"] = v["img_mask"].to(device, non_blocking=True)
                    v["ray_mask"] = v["ray_mask"].to(device, non_blocking=True)
                    v["reset"] = v["reset"].to(device, non_blocking=True)
                    v["update"] = v["update"].to(device, non_blocking=True)
                    v["camera_pose"] = v["camera_pose"].to(device, non_blocking=True)
                except Exception as e:
                    raise RuntimeError(f"Invalid view data at index {v_idx}: {e}.")
            start = time.time()
            output, _ = loss_of_one_batch(
                cut3r_views,
                model,
                None,
                accelerator,
                symmetrize_batch=False,
                use_amp=False,
                ret=None,
                img_mask=None,
                inference=True,
            )
            preds = output["pred"]
            end = time.time()
            all_cam_to_world_mat = [c2ws[i] for i in range(len(c2ws))]
            all_world_points = []
            for view_idx, pred in enumerate(preds):
                pts = pred["pts3d_in_other_view"]
                conf = pred.get("conf", None)
                pts_np = pts.to(torch.float32).cpu().numpy()[0]
                H, W, _ = pts_np.shape
                if conf is not None:
                    conf_np = conf.to(torch.float32).cpu().numpy()[0]
                    conf_mask = conf_np > args.depth_conf_thresh
                else:
                    conf_mask = np.ones((H, W), dtype=bool)
                pts_np = pts_np.reshape(-1, 3)
                conf_mask = conf_mask.reshape(-1)
                valid_mask = np.isfinite(pts_np).all(axis=1) & conf_mask
                pts_np = pts_np[valid_mask]
                c2w = c2ws[view_idx]
                R = c2w[:3, :3]
                t = c2w[:3, 3]
                pts_world = (R @ pts_np.T).T + t
                all_world_points.append(pts_world)
            total_time = end - start
            inference_time_ms = float(total_time * 1000.0)
            frame_count = len(cut3r_views)
            fps = frame_count / total_time if total_time > 0 else float("inf")
            print(f"Inference FPS (frames/s): {fps:.2f} [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
            scene_infer_times[scene].append(float(fps))

            merged_points = np.vstack(all_world_points) if len(all_world_points) > 0 else np.empty((0, 3), dtype=np.float32)
            if merged_points.shape[0] > 999999:
                sample_indices = np.random.choice(
                    merged_points.shape[0], 999999, replace=False
                )
                merged_points = merged_points[sample_indices]
            all_world_points = [merged_points]

            if not all_cam_to_world_mat or merged_points.shape[0] == 0:
                print(f"Skipping {scene}: failed to obtain valid camera poses or point clouds")
                continue

            metrics = evaluate_scene_and_save(
                scene,
                c2ws,
                first_gt_pose,
                frame_ids,
                all_cam_to_world_mat,
                all_world_points,
                output_scene_dir,
                args.gt_ply_dir,
                args.chamfer_max_dist,
                inference_time_ms,
                args.plot,
            )
            if metrics is not None:
                all_scenes_metrics["scenes"][scene] = {
                    key: float(value)
                    for key, value in metrics.items()
                    if key
                    in [
                        "chamfer_distance",
                        "ate",
                        "are",
                        "rpe_rot",
                        "rpe_trans",
                        "inference_time_ms",
                    ]
                }
                all_scenes_metrics["scenes"][scene]["fps"] = float(fps)
                print("Complete metrics", all_scenes_metrics["scenes"][scene])

        except Exception as e:
            print(f"Error processing scene {scene}: {e}")
            import traceback
            traceback.print_exc()

    for sid, times in scene_infer_times.items():
        if len(times) > 0:
            avg_fps = np.mean(times)
            print(f"Idx: {sid}, FPS_avg: {avg_fps:.3f}")
    compute_average_metrics_and_save(
        all_scenes_metrics,
        args.output_path,
        args.input_frame,
    )
