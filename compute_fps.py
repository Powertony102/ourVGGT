#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import numpy as np
import warnings
import os
import sys

# 保持与 eval_scannet.py 相同的导入风格，确保路径解析一致
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 仅用于保持结构一致，不在本脚本中使用模型推理
# from vggt.utils.eval_utils import get_all_scenes  # 如需从数据集目录列举场景可启用


def _strictly_increasing(series):
    """返回严格递增的子序列，并报告不连续位置"""
    filtered = []
    discontinuities = []
    last = None
    for i, v in enumerate(series):
        if last is None or (v > last):
            filtered.append(v)
            last = v
        else:
            discontinuities.append(i)
    return np.asarray(filtered), discontinuities


def _span_to_seconds(first, last):
    """根据跨度大小进行单位推断，返回秒"""
    span = float(last - first)
    if span <= 0:
        return 0.0
    # 经验阈值：>1e7 视为微秒，>1e5 视为毫秒，其余视为秒
    if span > 1e7:
        return span / 1e6
    if span > 1e5:
        return span / 1e3
    return span


def _fps_from_timestamps(ts_list):
    """按要求公式计算 FPS：总帧数 / (最后时间戳 - 第一个时间戳)"""
    ts = np.asarray(ts_list, dtype=float)
    ts = ts[np.isfinite(ts)]
    if ts.size < 2:
        raise ValueError("时间戳数量不足，至少需要两个时间戳")
    filtered, disc_idx = _strictly_increasing(ts)
    if disc_idx:
        warnings.warn(
            f"时间戳不连续，已忽略 {len(disc_idx)} 个非递增点：索引 {disc_idx}",
            RuntimeWarning,
        )
    if filtered.size < 2:
        raise ValueError("严格递增的时间戳不足以计算跨度")
    span_s = _span_to_seconds(filtered[0], filtered[-1])
    if span_s <= 0:
        raise ValueError("时间跨度非正，无法计算 FPS")
    total_frames = int(filtered.size)
    fps = float(total_frames) / span_s
    return total_frames, span_s, fps


def _fps_from_intervals(intervals, unit="auto"):
    """根据帧间隔计算 FPS；intervals 为 N-1 长度，帧数为 N"""
    arr = np.asarray(intervals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 1:
        raise ValueError("帧间隔数量不足，至少需要一个间隔")
    # 单位转换：auto 基于总和推断，ms/s/us 三种显式
    total = float(np.sum(arr))
    if unit == "ms" or (unit == "auto" and total > 1e5):
        span_s = total / 1e3
    elif unit == "us" or (unit == "auto" and total > 1e7):
        span_s = total / 1e6
    else:
        span_s = total
    total_frames = int(arr.size + 1)
    fps = float(total_frames) / span_s
    return total_frames, span_s, fps


def find_scene_dirs(root: Path):
    """在 root 下递归查找包含 metrics.json 的场景目录（名以 scene 开头）"""
    scenes = []
    # 一级目录扫描
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("scene") and (d / "metrics.json").exists():
            scenes.append(d)
    # 兼容 eval_scannet 输出结构 input_frame_*/scene*/metrics.json
    for sub in sorted(root.glob("input_frame_*/scene*")):
        if sub.is_dir() and (sub / "metrics.json").exists():
            scenes.append(sub)
    return scenes


def compute_and_print_fps(root: Path):
    scenes = find_scene_dirs(root)
    if not scenes:
        warnings.warn(
            f"在目录 {root} 下未发现包含 metrics.json 的场景目录",
            RuntimeWarning,
        )
    for scene_dir in scenes:
        mpath = scene_dir / "metrics.json"
        try:
            with open(mpath, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            warnings.warn(f"读取 {mpath} 失败：{e}")
            continue

        scene_name = scene_dir.name
        total_frames = None
        span_s = None
        fps = None

        # 优先使用时间戳
        ts_keys = [
            "timestamps",
            "frame_timestamps",
            "frame_times_s",
        ]
        ts = None
        for k in ts_keys:
            if k in metrics and isinstance(metrics[k], (list, tuple)):
                ts = metrics[k]
                break

        # 次选使用帧间隔
        intervals_keys = [
            "frame_intervals",
            "frame_intervals_ms",
            "frame_intervals_s",
        ]
        intervals = None
        intervals_unit = "auto"
        for k in intervals_keys:
            if k in metrics and isinstance(metrics[k], (list, tuple)):
                intervals = metrics[k]
                if k.endswith("_ms"):
                    intervals_unit = "ms"
                elif k.endswith("_s"):
                    intervals_unit = "s"
                break

        try:
            if ts is not None:
                total_frames, span_s, fps = _fps_from_timestamps(ts)
            elif intervals is not None:
                total_frames, span_s, fps = _fps_from_intervals(intervals, unit=intervals_unit)
            else:
                warnings.warn(
                    f"{scene_name} 的 metrics.json 不包含时间戳或帧间隔，无法按要求公式计算 FPS",
                    RuntimeWarning,
                )
                # 提供近似：使用 inference_time_ms 与已保存的 fps（若存在）
                inf_ms = metrics.get("inference_time_ms")
                approx_frames = (
                    metrics.get("n_frames")
                    or metrics.get("frame_count")
                    or metrics.get("num_frames")
                )
                span_s = (float(inf_ms) / 1000.0) if isinstance(inf_ms, (int, float)) else None
                if approx_frames is not None and span_s and span_s > 0:
                    total_frames = int(approx_frames)
                    fps = float(total_frames) / span_s
                else:
                    total_frames = approx_frames if approx_frames is not None else None
                    fps = metrics.get("fps")

        except Exception as e:
            warnings.warn(f"{scene_name} 计算 FPS 时发生错误：{e}")
            continue

        # 输出格式：场景名称、总帧数、时间跨度(秒)、FPS
        frames_str = str(total_frames) if total_frames is not None else "NA"
        span_str = f"{span_s:.3f}" if isinstance(span_s, (int, float)) else "NA"
        fps_str = f"{float(fps):.3f}" if isinstance(fps, (int, float)) else "NA"
        print(
            f"Scene: {scene_name} | Frames: {frames_str} | Span(s): {span_str} | FPS: {fps_str}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path(__file__).parent,
        help="包含场景 metrics.json 的目录（可为 input_frame_* 或 scene* 结构）",
    )
    args = parser.parse_args()
    # 与 eval_scannet.py 保持一致的随机性控制
    try:
        import torch

        torch.manual_seed(33)
    except Exception:
        pass
    np.random.seed(0)
    compute_and_print_fps(args.results_dir)


if __name__ == "__main__":
    main()
