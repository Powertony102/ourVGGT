import os
import sys

# Ensure project root is on sys.path for absolute imports like `vggt.*`
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

import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from collections import defaultdict
import torchvision.transforms as transforms


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/jovyan/shared/xinzeli/ckpt/model_tracker_fixed_e20.pt",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="VGGT")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/eval_7andN_ours+fast/",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument(
        "--merging", type=int, default=None, help="VGGT aggregator merging steps"
    )
    parser.add_argument("--kf", type=int, default=2, help="key frame")
    parser.add_argument(
        "--input_frame",
        type=int,
        default=200,
        help="Maximum number of frames selected for processing per scene",
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=None,
        help="Enable subscene grouping with the specified number of groups",
    )
    parser.add_argument(
        "--cut3r_model_path",
        type=str,
        default=os.path.join("CUT3R", "src", "cut3r_512_dpt_4_64.pth"),
        help="Checkpoint path for CUT3R model",
    )
    return parser


def main(args):
    from eval_old.data import SevenScenes, NRGBD
    from eval_old.utils import accuracy, completion
    from vggt.utils.eval_utils import build_frame_selection
    from pathlib import Path
    # Ensure deterministic numpy sampling (e.g., point subsampling)
    np.random.seed(0)

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
    else:
        raise NotImplementedError
    datasets_all = {
        "7scenes": SevenScenes(
            split="test",
            ROOT="/home/jovyan/shared/xinzeli/fastplus/7-scenes",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=1,  # Set to 1 to get all frames, will use build_frame_selection later
        ),  # 20),
        "NRGBD": NRGBD(
            split="test",
            ROOT="/home/jovyan/shared/xinzeli/fastplus/nrgbd",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=1,  # Set to 1 to get all frames, will use build_frame_selection later
        ),
    }

    device = args.device
    model_name = args.model_name

    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from criterion import Regr3D_t_ScaleShiftInv, L21

    dtype = torch.bfloat16
    model_name_upper = args.model_name.upper()
    cut3r_ctx = None
    if model_name_upper == "VGGT":
        from vggt.models.vggt import VGGT
        model = VGGT(merging=args.merging, enable_point=True)
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        model = model.cuda().eval()
        model = model.to(torch.bfloat16)
        del ckpt
    elif model_name_upper == "FAST3R":
        from fast3r.dust3r.inference_multiview import inference as fast3r_inference
        from fast3r.models.fast3r import Fast3R
        from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
        device_obj = torch.device(args.device)
        model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")
        model = model.to(device_obj).to(torch.bfloat16).eval()
        lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
        lit_module.eval()
        try:
            p = next(model.parameters())
            print(f"Model selected: {args.model_name}, class={model.__class__.__name__}, param_dtype={p.dtype}")
        except Exception:
            print(f"Model selected: {args.model_name}, class={model.__class__.__name__}")
    elif model_name_upper == "CUT3R":
        try:
            from CUT3R.add_ckpt_path import add_path_to_dust3r
            add_path_to_dust3r(args.cut3r_model_path)
            from dust3r.model import ARCroco3DStereo
            from dust3r.inference import inference as cut3r_inference
            from dust3r.utils.image import load_images as cut3r_load_images
        except Exception as e:
            raise ImportError(f"Failed to import CUT3R modules: {e}")
        device_obj = torch.device(args.device)
        model = ARCroco3DStereo.from_pretrained(args.cut3r_model_path)
        model = model.to(device_obj).to(torch.float32).eval()
        cut3r_ctx = {"inference": cut3r_inference, "load_images": cut3r_load_images, "device": device_obj}
        try:
            p = next(model.parameters())
            print(f"Model selected: {args.model_name}, class={model.__class__.__name__}, param_dtype={p.dtype}")
        except Exception:
            print(f"Model selected: {args.model_name}, class={model.__class__.__name__}")
    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")
    os.makedirs(osp.join(args.output_dir, f"input_frame_{args.input_frame}"), exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(osp.join(args.output_dir, f"input_frame_{args.input_frame}"), name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, "logs.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0
            scene_infer_times = defaultdict(list)

            for data_idx in tqdm(range(len(dataset))):
                # Get full batch from dataset
                full_batch = dataset[data_idx]
                
                # Extract frame information for build_frame_selection
                if isinstance(full_batch, list):
                    views = full_batch
                else:
                    views = [full_batch]
                
                # Build numeric image_paths and available_pose_frame_ids from views
                image_paths = []
                available_pose_frame_ids = []
                for i, view in enumerate(views):
                    label = view.get("label", None)
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
                    # Ensure numeric stem to satisfy build_frame_selection
                    image_paths.append(Path(f"{frame_id:06d}"))
                
                # Use build_frame_selection to select frames
                if len(available_pose_frame_ids) > 0:
                    available_pose_frame_ids = np.array(available_pose_frame_ids)
                    selected_frame_ids, selected_image_paths, selected_pose_indices = (
                        build_frame_selection(
                            image_paths, available_pose_frame_ids, args.input_frame
                        )
                    )
                    print(
                        f"[FrameSelection] total_frames={len(available_pose_frame_ids)} selected_ids={selected_frame_ids}"
                    )
                    
                    # Filter views based on selected indices
                    selected_views = [views[i] for i in selected_pose_indices if i < len(views)]
                    
                    if len(selected_views) == 0:
                        print(f"No frames selected for data_idx {data_idx}, skipping")
                        continue
                    
                    batch = selected_views
                else:
                    # Fallback: use original batch
                    print(
                        f"[FrameSelection] total_frames=0 selected_ids=[] (fallback to original batch)"
                    )
                    batch = [full_batch]
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
                    for name in list(view.keys()):
                        if name in ignore_keys:
                            continue
                        val = view[name]
                        if isinstance(val, (tuple, list)):
                            view[name] = [x.to(device, non_blocking=True) for x in val]
                        else:
                            try:
                                view[name] = val.to(device, non_blocking=True)
                            except AttributeError:
                                view[name] = val

                # Ensure masks and poses are torch tensors with batch dimension
                for view in batch:
                    if "valid_mask" in view:
                        vm = view["valid_mask"]
                        if isinstance(vm, np.ndarray):
                            vm_t = torch.from_numpy(vm.astype(np.bool_)).unsqueeze(0)
                            view["valid_mask"] = vm_t.to(device, non_blocking=True)
                        elif torch.is_tensor(vm) and vm.ndim == 2:
                            view["valid_mask"] = vm.unsqueeze(0).to(device, non_blocking=True)
                    if "camera_pose" in view:
                        cp = view["camera_pose"]
                        if isinstance(cp, np.ndarray):
                            cp_t = torch.from_numpy(cp.astype(np.float32)).unsqueeze(0)
                            view["camera_pose"] = cp_t.to(device, non_blocking=True)
                        elif torch.is_tensor(cp) and cp.ndim == 2:
                            view["camera_pose"] = cp.unsqueeze(0).to(device, non_blocking=True)
                    if "pts3d" in view:
                        p3d = view["pts3d"]
                        if isinstance(p3d, np.ndarray):
                            p3d_t = torch.from_numpy(p3d.astype(np.float32))
                        elif torch.is_tensor(p3d):
                            p3d_t = p3d.to(torch.float32)
                        else:
                            p3d_t = torch.as_tensor(p3d, dtype=torch.float32)
                        if p3d_t.ndim == 3:
                            p3d_t = p3d_t.unsqueeze(0)
                        view["pts3d"] = p3d_t.to(device, non_blocking=True)

                pts_all = []
                pts_gt_all = []
                images_all = []
                masks_all = []
                conf_all = []
                in_camera1 = None

                dtype = (
                    torch.bfloat16
                    if torch.cuda.get_device_capability()[0] >= 8
                    else torch.float16
                )
                if model_name_upper == "VGGT":
                    dtype_autocast = (
                        torch.bfloat16
                        if torch.cuda.get_device_capability()[0] >= 8
                        else torch.float16
                    )
                    with torch.cuda.amp.autocast(dtype=dtype_autocast):
                        for view in batch:
                            view["img"] = (view["img"] + 1.0) / 2.0
                        imgs_tensor = torch.stack([v["img"] for v in batch], dim=0)
                    with torch.cuda.amp.autocast(dtype=dtype_autocast):
                        with torch.no_grad():
                            torch.cuda.synchronize()
                            start = time.time()
                            predictions = model(imgs_tensor, num_groups=args.num_groups)
                            torch.cuda.synchronize()
                            end = time.time()
                            elapsed_s = end - start
                            frame_count = imgs_tensor.shape[0]
                            fps = frame_count / elapsed_s if elapsed_s > 0 else float("inf")
                            print(f"Inference FPS (frames/s): {fps:.2f}")
                    views = batch
                    if "pose_enc" in predictions:
                        B, S = predictions["pose_enc"].shape[:2]
                    elif "world_points" in predictions:
                        B, S = predictions["world_points"].shape[:2]
                    else:
                        raise KeyError("predictions is missing a key to infer sequence length")
                    ress = []
                    for s in range(S):
                        res = {
                            "pts3d_in_other_view": predictions["world_points"][:, s],
                            "conf": predictions["world_points_conf"][:, s],
                            "depth": predictions["depth"][:, s],
                            "depth_conf": predictions["depth_conf"][:, s],
                            "camera_pose": predictions["pose_enc"][:, s, :],
                        }
                        if isinstance(views, list) and s < len(views) and "valid_mask" in views[s]:
                            res["valid_mask"] = views[s]["valid_mask"]
                        if "track" in predictions:
                            res.update(
                                {
                                    "track": predictions["track"][:, s],
                                    "vis": (predictions.get("vis", None)[:, s] if "vis" in predictions else None),
                                    "track_conf": (predictions.get("conf", None)[:, s] if "conf" in predictions else None),
                                }
                            )
                        ress.append(res)
                    preds = ress
                else:
                    views = batch
                    try:
                        if model_name_upper == "CUT3R":
                            img_paths = []
                            all_resolved = True
                            for v in views:
                                impath = v.get("instance", None)
                                if isinstance(impath, str) and osp.isfile(impath):
                                    img_paths.append(impath)
                                    continue
                                ds_name = v.get("dataset", None)
                                label = v.get("label", None)
                                resolved = None
                                if isinstance(ds_name, str) and isinstance(label, str):
                                    try:
                                        scene_id, im_idx = label.rsplit("/", 1)
                                        if ds_name.lower() == "7scenes":
                                            candidate = osp.join(dataset.ROOT, scene_id, f"frame-{im_idx}.color.png")
                                            if osp.isfile(candidate):
                                                resolved = candidate
                                        elif ds_name.lower() == "nrgbd":
                                            candidate = osp.join(dataset.ROOT, scene_id, "images", f"img{im_idx}.png")
                                            if osp.isfile(candidate):
                                                resolved = candidate
                                    except Exception:
                                        resolved = None
                                if resolved is None:
                                    all_resolved = False
                                    break
                                img_paths.append(resolved)
                            if all_resolved:
                                images = cut3r_ctx["load_images"](img_paths, size=args.size, verbose=False)
                                cut3r_views = []
                                for i in range(len(images)):
                                    img = images[i]["img"].to(cut3r_ctx["device"], dtype=torch.float32)
                                    true_shape = torch.from_numpy(images[i]["true_shape"]).to(img.device)
                                    cut3r_views.append(
                                        {
                                            "img": img,
                                            "ray_map": torch.full((img.shape[0], img.shape[-2], img.shape[-1], 6), torch.nan, device=img.device, dtype=torch.float32),
                                            "true_shape": true_shape,
                                            "idx": i,
                                            "instance": str(i),
                                            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0).to(img.device).to(torch.float32),
                                            "img_mask": torch.tensor(True, device=img.device).unsqueeze(0),
                                            "ray_mask": torch.tensor(True, device=img.device).unsqueeze(0),
                                            "update": torch.tensor(True, device=img.device).unsqueeze(0),
                                            "reset": torch.tensor(False, device=img.device).unsqueeze(0),
                                        }
                                    )
                            else:
                                cut3r_views = []
                                for i, v in enumerate(views):
                                    img = v["img"].to(cut3r_ctx["device"], dtype=torch.float32)
                                    true_shape = torch.tensor([img.shape[-2], img.shape[-1]], dtype=torch.int32, device=img.device)
                                    cut3r_views.append(
                                        {
                                            "img": img,
                                            "ray_map": torch.full((img.shape[0], img.shape[-2], img.shape[-1], 6), torch.nan, device=img.device, dtype=torch.float32),
                                            "true_shape": true_shape,
                                            "idx": i,
                                            "instance": str(i),
                                            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0).to(img.device).to(torch.float32),
                                            "img_mask": torch.tensor(True, device=img.device).unsqueeze(0),
                                            "ray_mask": torch.tensor(True, device=img.device).unsqueeze(0),
                                            "update": torch.tensor(True, device=img.device).unsqueeze(0),
                                            "reset": torch.tensor(False, device=img.device).unsqueeze(0),
                                        }
                                    )
                            torch.cuda.synchronize()
                            start = time.time()
                            output = model(cut3r_views)
                            torch.cuda.synchronize()
                            end = time.time()
                            outputs = {"pred": output.ress, "views": output.views}
                            elapsed_s = end - start
                            frame_count = len(cut3r_views)
                            fps = frame_count / elapsed_s if elapsed_s > 0 else float("inf")
                            print(f"Inference FPS (frames/s): {fps:.2f}")
                            preds = []
                            valid_length = len(outputs["pred"]) // args.revisit
                            pred_slice = outputs["pred"][ -valid_length : ]
                            for s, pred in enumerate(pred_slice):
                                pts = pred.get("pts3d_in_other_view", None)
                                conf = pred.get("conf", None)
                                if pts is None:
                                    raise KeyError("CUT3R outputs missing 'pts3d_in_other_view'")
                                res = {
                                    "pts3d_in_other_view": pts,
                                    "conf": conf if conf is not None else torch.ones_like(pts[..., 0]),
                                }
                                if isinstance(views, list) and s < len(views) and "valid_mask" in views[s]:
                                    res["valid_mask"] = views[s]["valid_mask"]
                                preds.append(res)
                        else:
                            fast3r_views = []
                            for v in views:
                                img = v["img"]
                                if img.ndim == 3:
                                    img = img.unsqueeze(0)
                                img_batched = img.to(device_obj, dtype=torch.bfloat16)
                                ts = v.get("true_shape")
                                if ts is None:
                                    true_shape_batched = torch.tensor([img_batched.shape[-2], img_batched.shape[-1]], dtype=torch.int32, device=device_obj).unsqueeze(0)
                                else:
                                    if isinstance(ts, np.ndarray):
                                        ts = torch.from_numpy(ts)
                                    true_shape_batched = ts.to(device_obj)
                                    if true_shape_batched.ndim == 1:
                                        true_shape_batched = true_shape_batched.unsqueeze(0)
                                idx_val = v.get("idx", 0)
                                if torch.is_tensor(idx_val):
                                    idx_val = int(idx_val.item())
                                else:
                                    idx_val = int(idx_val)
                                fast3r_views.append(dict(img=img_batched, true_shape=true_shape_batched, idx=idx_val, instance=str(v.get("instance", ""))))
                            torch.cuda.synchronize()
                            start = time.time()
                            output_dict, profiling_info = fast3r_inference(fast3r_views, model, device_obj, dtype=torch.bfloat16, verbose=False, profiling=True)
                            torch.cuda.synchronize()
                            end = time.time()
                            total_time = profiling_info.get("total_time", None) if isinstance(profiling_info, dict) else None
                            elapsed_s = total_time if (total_time and total_time > 0) else (end - start)
                            frame_count = len(fast3r_views)
                            fps = frame_count / elapsed_s if elapsed_s > 0 else float("inf")
                            print(f"Inference FPS (frames/s): {fps:.2f}")
                            preds_raw = output_dict["preds"]
                            ress = []
                            for s, pred in enumerate(preds_raw):
                                pts = pred["pts3d_in_other_view"]
                                conf = pred.get("conf", None)
                                res = {"pts3d_in_other_view": pts, "conf": conf if conf is not None else torch.ones_like(pts[..., 0])}
                                if isinstance(views, list) and s < len(views) and "valid_mask" in views[s]:
                                    res["valid_mask"] = views[s]["valid_mask"]
                                ress.append(res)
                            preds = ress
                    except Exception as e:
                        import traceback
                        print(f"{args.model_name} inference failed: {e}")
                        try:
                            print(f"Model class: {model.__class__.__name__}")
                            p = next(model.parameters())
                            print(f"Model param dtype: {p.dtype}")
                        except Exception:
                            pass
                        try:
                            dv = views[0]
                            dbg = {k: (dv[k].dtype if torch.is_tensor(dv[k]) else type(dv[k])) for k in dv.keys()}
                            print(f"First view dtypes: {dbg}")
                        except Exception:
                            pass
                        traceback.print_exc()
                        continue
                    try:
                        for j in range(len(preds)):
                            pts3d = preds[j].get("pts3d_in_other_view")
                            confm = preds[j].get("conf")
                            if pts3d is not None and torch.is_tensor(pts3d):
                                if pts3d.ndim == 3:
                                    preds[j]["pts3d_in_other_view"] = pts3d.unsqueeze(0)
                                elif pts3d.ndim == 5 and pts3d.shape[-1] == 3:
                                    preds[j]["pts3d_in_other_view"] = pts3d.squeeze(1)
                            if confm is not None and torch.is_tensor(confm):
                                if confm.ndim == 2:
                                    preds[j]["conf"] = confm.unsqueeze(0)
                                elif confm.ndim == 4 and confm.shape[-1] == 1:
                                    preds[j]["conf"] = confm.squeeze(-1)
                    except Exception:
                        pass

                    valid_length = len(preds) // args.revisit
                    if args.revisit > 1:
                        preds = preds[-valid_length:]
                        batch = batch[-valid_length:]

                    # Evaluation
                    print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                        criterion.get_all_pts3d_t(batch, preds)
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

                        img_t = view["img"]
                        if torch.is_tensor(img_t) and img_t.ndim == 3:
                            image = img_t.permute(1, 2, 0).cpu().numpy()
                        else:
                            image = img_t.permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]

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

                        images_all.append(image[None, ...])
                        pts_all.append(pts[None, ...])
                        pts_gt_all.append(pts_gt[None, ...])
                        masks_all.append(mask[None, ...])
                        conf_all.append(conf[None, ...])

                images_all = np.concatenate(images_all, axis=0)
                pts_all = np.concatenate(pts_all, axis=0)
                pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                masks_all = np.concatenate(masks_all, axis=0)

                scene_id = view["label"].rsplit("/", 1)[0]
                # Record FPS per scene for averaging later
                try:
                    scene_infer_times[scene_id].append(float(fps))
                except Exception:
                    pass

                save_params = {}

                save_params["images_all"] = images_all
                save_params["pts_all"] = pts_all
                save_params["pts_gt_all"] = pts_gt_all
                save_params["masks_all"] = masks_all

                pts_all_masked = pts_all[masks_all > 0]
                pts_gt_all_masked = pts_gt_all[masks_all > 0]
                images_all_masked = images_all[masks_all > 0]

                mask = np.isfinite(pts_all_masked)
                pts_all_masked = pts_all_masked[mask]

                mask_gt = np.isfinite(pts_gt_all_masked)
                pts_gt_all_masked = pts_gt_all_masked[mask_gt]
                images_all_masked = images_all_masked[mask]

                # Reshape to point cloud (N, 3) before sampling
                pts_all_masked = pts_all_masked.reshape(-1, 3)
                pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                images_all_masked = images_all_masked.reshape(-1, 3)

                # If number of points exceeds threshold, sample by points
                if pts_all_masked.shape[0] > 999999:
                    sample_indices = np.random.choice(
                        pts_all_masked.shape[0], 999999, replace=False
                    )
                    pts_all_masked = pts_all_masked[sample_indices]
                    images_all_masked = images_all_masked[sample_indices]

                # Apply the same sampling to GT point cloud
                if pts_gt_all_masked.shape[0] > 999999:
                    sample_indices_gt = np.random.choice(
                        pts_gt_all_masked.shape[0], 999999, replace=False
                    )
                    pts_gt_all_masked = pts_gt_all_masked[sample_indices_gt]

                if args.use_proj:

                    def umeyama_alignment(
                        src: np.ndarray, dst: np.ndarray, with_scale: bool = True
                    ):
                        assert src.shape == dst.shape
                        N, dim = src.shape

                        mu_src = src.mean(axis=0)
                        mu_dst = dst.mean(axis=0)
                        src_c = src - mu_src
                        dst_c = dst - mu_dst

                        Sigma = dst_c.T @ src_c / N  # (3,3)

                        U, D, Vt = np.linalg.svd(Sigma)

                        S = np.eye(dim)
                        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                            S[-1, -1] = -1

                        R = U @ S @ Vt

                        if with_scale:
                            var_src = (src_c**2).sum() / N
                            s = (D * S.diagonal()).sum() / var_src
                        else:
                            s = 1.0

                        t = mu_dst - s * R @ mu_src

                        return s, R, t

                    pts_all_masked = pts_all_masked.reshape(-1, 3)
                    pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                    s, R, t = umeyama_alignment(
                        pts_all_masked, pts_gt_all_masked, with_scale=True
                    )
                    pts_all_aligned = (s * (R @ pts_all_masked.T)).T + t  # (N,3)
                    pts_all_masked = pts_all_aligned

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_all_masked)
                pcd.colors = o3d.utility.Vector3dVector(images_all_masked)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked)
                pcd_gt.colors = o3d.utility.Vector3dVector(images_all_masked)

                trans_init = np.eye(4)

                threshold = 0.1
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
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}, FPS: {fps:.2f}"
                )
                print(
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}, FPS: {fps:.2f}",
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

            # Get depth from pcd and run TSDFusion
            to_write = ""
            # Read the log file
            if os.path.exists(osp.join(save_path, "logs.txt")):
                with open(osp.join(save_path, "logs.txt"), "r") as f_sub:
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
                            if key == "scene_id" or value is None:
                                continue
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
                # Summarize per-scene average FPS
                time_lines = []
                for sid, times in scene_infer_times.items():
                    if len(times) > 0:
                        avg_fps = np.mean(times)
                        time_lines.append(f"Idx: {sid}, FPS_avg: {avg_fps:.2f}")
                time_block = "\n".join(time_lines) + (
                    "\n" if len(time_lines) > 0 else ""
                )

                f.write(to_write + time_block + print_str)


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
    (?:,\s*FPS:\s*(?P<fps>[^,]+))?
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

    main(args)
