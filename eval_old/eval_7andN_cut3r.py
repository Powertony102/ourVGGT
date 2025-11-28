import os
import sys

# Ensure project root is on sys.path for absolute imports like fast3r.* / CUT3R.*
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

EVAL_OLD_DIR = os.path.abspath(os.path.dirname(__file__))
if EVAL_OLD_DIR not in sys.path:
    sys.path.insert(0, EVAL_OLD_DIR)

FAST3R_PROJECT_DIR = os.path.abspath(os.path.join(ROOT_DIR, "fast3r"))
FAST3R_PKG_DIR = os.path.abspath(os.path.join(ROOT_DIR, "fast3r", "fast3r"))
for p in (FAST3R_PROJECT_DIR, FAST3R_PKG_DIR):
    if p not in sys.path:
        sys.path.append(p)

CUT3R_DIR = os.path.abspath(os.path.join(ROOT_DIR, "CUT3R"))
if CUT3R_DIR not in sys.path:
    sys.path.insert(0, CUT3R_DIR)
CUT3R_SRC_DIR = os.path.join(CUT3R_DIR, "src")
if CUT3R_SRC_DIR not in sys.path:
    sys.path.insert(0, CUT3R_SRC_DIR)
CUT3R_EVAL_DIR = os.path.join(CUT3R_DIR, "eval")
if CUT3R_EVAL_DIR not in sys.path:
    sys.path.insert(0, CUT3R_EVAL_DIR)

import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join("CUT3R", "src", "cut3r_512_dpt_4_64.pth"),
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
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--input_frame",
        type=int,
        default=200,
        help="Maximum number of frames selected for processing per scene",
    )
    return parser


def main(args):
    from CUT3R.add_ckpt_path import add_path_to_dust3r
    from pathlib import Path
    from vggt.utils.eval_utils import build_frame_selection
    if not args.weights:
        args.weights = os.path.join("CUT3R", "src", "cut3r_512_dpt_4_64.pth")
    looks_like_local = os.path.isabs(args.weights) or args.weights.startswith('.') or args.weights.endswith('.pth')
    if looks_like_local and not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    if looks_like_local and os.path.exists(args.weights):
        add_path_to_dust3r(args.weights)
    from mv_recon.data import SevenScenes, NRGBD
    from mv_recon.utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError
    datasets_all = {
        "7scenes": SevenScenes(
            split="test",
            ROOT="/home/jovyan/shared/xinzeli/fastplus/7-scenes",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=3,
        ),  # 20),
        "NRGBD": NRGBD(
            split="test",
            ROOT="/home/jovyan/shared/xinzeli/fastplus/nrgbd/",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=3,
        ),
    }

    accelerator = Accelerator()
    device = accelerator.device
    model_name = args.model_name
    if model_name == "ours" or model_name == "cut3r":
        from dust3r.model import ARCroco3DStereo
        try:
            from mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        except ImportError:
            sys.path.insert(0, CUT3R_EVAL_DIR)
            from mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy

        dtype = torch.bfloat16
        model = ARCroco3DStereo.from_pretrained(args.weights).to(device).to(dtype)
        model.eval()
    else:
        raise NotImplementedError
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
                    full_views = dataset[data_idx]
                    views = full_views if isinstance(full_views, list) else [full_views]

                    image_paths = []
                    available_pose_frame_ids = []
                    for i, v in enumerate(views):
                        label = v.get("label", None)
                        frame_id = None
                        if isinstance(label, str):
                            try:
                                frame_id_str = label.rsplit("/", 1)[-1]
                                frame_id = int(frame_id_str.lstrip("0") or "0")
                            except Exception:
                                frame_id = None
                        if frame_id is None:
                            frame_id = i
                        available_pose_frame_ids.append(frame_id)
                        image_paths.append(Path(f"{frame_id:06d}"))

                    selected_frame_ids, selected_image_paths, selected_pose_indices = build_frame_selection(
                        image_paths, np.array(available_pose_frame_ids), args.input_frame
                    )
                    batch = [views[i] for i in selected_pose_indices if i < len(views)]
                    if len(batch) == 0:
                        print(f"No frames selected for {name_data} idx={data_idx}, skipping")
                        continue
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
                        img_t = view.get("img", None)
                        if torch.is_tensor(img_t):
                            if img_t.ndim == 3:
                                img_t = img_t.unsqueeze(0)
                            view["img"] = img_t.to(device, non_blocking=True)
                        if "true_shape" in view:
                            ts = view["true_shape"]
                            if isinstance(ts, np.ndarray):
                                ts = torch.from_numpy(ts.astype(np.int32))
                            elif isinstance(ts, (tuple, list)):
                                ts = torch.tensor(ts, dtype=torch.int32)
                            elif torch.is_tensor(ts):
                                ts = ts.to(torch.int32)
                            else:
                                ts = torch.tensor(view["img"].shape[-2:], dtype=torch.int32)
                            if ts.ndim == 1 and ts.numel() == 2:
                                ts = ts.unsqueeze(0).repeat(view["img"].shape[0], 1)
                            view["true_shape"] = ts.to(device, non_blocking=True)
                        if "valid_mask" in view:
                            vm = view["valid_mask"]
                            if isinstance(vm, np.ndarray):
                                vm = torch.from_numpy(vm.astype(np.bool_))
                            if torch.is_tensor(vm) and vm.ndim == 2:
                                vm = vm.unsqueeze(0)
                            view["valid_mask"] = vm.to(device, non_blocking=True)
                        if "camera_pose" in view:
                            cp = view["camera_pose"]
                            if isinstance(cp, np.ndarray):
                                cp = torch.from_numpy(cp.astype(np.float32))
                            if torch.is_tensor(cp) and cp.ndim == 2:
                                cp = cp.unsqueeze(0)
                            view["camera_pose"] = cp.to(device, non_blocking=True)
                        if "pts3d" in view:
                            p3d = view["pts3d"]
                            if isinstance(p3d, np.ndarray):
                                p3d = torch.from_numpy(p3d.astype(np.float32))
                            elif not torch.is_tensor(p3d):
                                p3d = torch.as_tensor(p3d, dtype=torch.float32)
                            if p3d.ndim == 3:
                                p3d = p3d.unsqueeze(0)
                            view["pts3d"] = p3d.to(device, non_blocking=True)
                        if "ray_map" in view:
                            rm = view["ray_map"]
                            if torch.is_tensor(rm):
                                if rm.ndim == 3:
                                    if rm.shape[0] == 6:
                                        rm = rm.permute(1, 2, 0).unsqueeze(0)
                                    elif rm.shape[-1] == 6:
                                        rm = rm.unsqueeze(0)
                                    else:
                                        rm = rm.unsqueeze(0)
                                elif rm.ndim == 4:
                                    if rm.shape[1] == 6:
                                        rm = rm.permute(0, 2, 3, 1)
                                view["ray_map"] = rm.to(device, non_blocking=True)
                            else:
                                rma = np.asarray(rm)
                                if rma.ndim == 3 and rma.shape[0] == 6:
                                    rma = np.transpose(rma, (1, 2, 0))[None]
                                elif rma.ndim == 3 and rma.shape[-1] == 6:
                                    rma = rma[None]
                                else:
                                    rma = rma[None]
                                view["ray_map"] = torch.from_numpy(rma.astype(np.float32)).to(device, non_blocking=True)
                        if "img_mask" in view:
                            im = view["img_mask"]
                            if isinstance(im, (bool, np.bool_)):
                                im = torch.tensor(im).unsqueeze(0)
                            view["img_mask"] = im.to(device, non_blocking=True)
                        if "ray_mask" in view:
                            rmask = view["ray_mask"]
                            if isinstance(rmask, (bool, np.bool_)):
                                rmask = torch.tensor(rmask).unsqueeze(0)
                            view["ray_mask"] = rmask.to(device, non_blocking=True)
                        if "reset" in view:
                            r = view["reset"]
                            if isinstance(r, (bool, np.bool_)):
                                r = torch.tensor(r).unsqueeze(0)
                            view["reset"] = r.to(device, non_blocking=True)
                        if "update" in view:
                            u = view["update"]
                            if isinstance(u, (bool, np.bool_)):
                                u = torch.tensor(u).unsqueeze(0)
                            view["update"] = u.to(device, non_blocking=True)
                    for view in batch:
                        for k, v in list(view.items()):
                            if torch.is_tensor(v) and torch.is_floating_point(v):
                                view[k] = v.to(torch.bfloat16)

                    if model_name == "ours" or model_name == "cut3r":
                        revisit = args.revisit
                        update = not args.freeze
                        if revisit > 1:
                            # repeat input for 'revisit' times
                            new_views = []
                            for r in range(revisit):
                                for i in range(len(batch)):
                                    new_view = deepcopy(batch[i])
                                    new_view["idx"] = [
                                        (r * len(batch) + i)
                                        for _ in range(len(batch[i]["idx"]))
                                    ]
                                    new_view["instance"] = [
                                        str(r * len(batch) + i)
                                        for _ in range(len(batch[i]["instance"]))
                                    ]
                                    if r > 0:
                                        if not update:
                                            new_view["update"] = torch.zeros_like(
                                                batch[i]["update"]
                                            ).bool()
                                    new_views.append(new_view)
                            batch = new_views
                        try:
                            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                                start = time.time()
                                output = model(batch)
                                end = time.time()
                                preds, batch = output.ress, output.views
                        except Exception as e:
                            import traceback
                            print(f"cut3r inference failed: {e}")
                            try:
                                print(f"Model class: {model.__class__.__name__}")
                                p = next(model.parameters())
                                print(f"Model param dtype: {p.dtype}")
                            except Exception:
                                pass
                            try:
                                dv = batch[0]
                                dbg = {k: (dv[k].dtype if torch.is_tensor(dv[k]) else type(dv[k])) for k in dv.keys()}
                                print(f"First CUT3R view dtypes: {dbg}")
                            except Exception:
                                pass
                            traceback.print_exc()
                            continue
                        valid_length = len(preds) // revisit
                        preds = preds[-valid_length:]
                        batch = batch[-valid_length:]
                        fps = len(batch) / (end - start)
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
                                in_camera1 = (
                                    view["camera_pose"][0].to(torch.float32).cpu().numpy()
                                )

                            image = (
                                view["img"].permute(0, 2, 3, 1).to(torch.float32).cpu().numpy()[0]
                            )
                            image = (image + 1.0) / 2.0
                            mask = view["valid_mask"].to(torch.float32).cpu().numpy()[0]

                            pts = pred_pts[j].to(torch.float32).cpu().numpy()[0]
                            conf = preds[j]["conf"].to(torch.float32).cpu().data.numpy()[0]
                            if args.conf_thresh and args.conf_thresh > 0:
                                mask = mask & (conf > args.conf_thresh)

                            pts_gt = gt_pts[j].detach().to(torch.float32).cpu().numpy()[0]

                            # restore absolute depth and unify coords
                            shift_val = gt_shift_z.to(torch.float32).cpu().item()
                            pts[..., -1] += shift_val
                            pts_gt[..., -1] += shift_val
                            pts = geotrf(in_camera1, pts)
                            pts_gt = geotrf(in_camera1, pts_gt)

                            images_all.append(image[None, ...])
                            pts_all.append(pts[None, ...])
                            pts_gt_all.append(pts_gt[None, ...])
                            masks_all.append(mask[None, ...])
                            conf_all.append(conf[None, ...])

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)

                    scene_label = batch[0]["label"]
                    try:
                        scene_id = (scene_label[0] if isinstance(scene_label, (list, tuple)) else scene_label).rsplit("/", 1)[0]
                    except Exception:
                        scene_id = str(scene_label)

                    if "DTU" in name_data:
                        threshold = 100
                    else:
                        threshold = 0.1

                    pts_all_masked = pts_all[masks_all > 0]
                    pts_gt_all_masked = pts_gt_all[masks_all > 0]
                    images_all_masked = images_all[masks_all > 0]

                    mask = np.isfinite(pts_all_masked)
                    pts_all_masked = pts_all_masked[mask]

                    mask_gt = np.isfinite(pts_gt_all_masked)
                    pts_gt_all_masked = pts_gt_all_masked[mask_gt]
                    images_all_masked = images_all_masked[mask]

                    pts_all_masked = pts_all_masked.reshape(-1, 3)
                    pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                    images_all_masked = images_all_masked.reshape(-1, 3)

                    if pts_all_masked.shape[0] > 999999:
                        sample_indices = np.random.choice(
                            pts_all_masked.shape[0], 999999, replace=False
                        )
                        pts_all_masked = pts_all_masked[sample_indices]
                        images_all_masked = images_all_masked[sample_indices]

                    if pts_gt_all_masked.shape[0] > 999999:
                        sample_indices_gt = np.random.choice(
                            pts_gt_all_masked.shape[0], 999999, replace=False
                        )
                        pts_gt_all_masked = pts_gt_all_masked[sample_indices_gt]

                    trans_init = np.eye(4)

                    # Guard against empty point clouds
                    if pts_all_masked.shape[0] == 0 or pts_gt_all_masked.shape[0] == 0:
                        print(f"Empty point cloud for {scene_id}; skipping metrics for this scene")
                        print(
                            f"Empty point cloud for {scene_id}; skipping metrics for this scene",
                            file=open(log_file, "a"),
                        )
                        continue

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts_all_masked)
                    pcd.colors = o3d.utility.Vector3dVector(images_all_masked)

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked)
                    pcd_gt.colors = o3d.utility.Vector3dVector(images_all_masked)

                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        pcd,
                        pcd_gt,
                        threshold,
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
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                    )
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                        file=open(log_file, "a"),
                    )

                    acc_all += acc
                    comp_all += comp
                    nc1_all += nc1
                    nc2_all += nc2

                    acc_all_med += acc_med
                    comp_all_med += comp_med
                    nc1_all_med += nc1_med
                    nc2_all_med += nc2_med

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
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
