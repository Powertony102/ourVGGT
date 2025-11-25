#!/usr/bin/env python3
"""
Frame Selection and Video Synthesis Script
Replicates the exact frame selection logic from eval_scannet.py and creates videos from selected frames.
"""

import argparse
import json
import logging
import subprocess
import time
import zipfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_sorted_image_paths(images_dir: Path) -> List[Path]:
    """Get sorted image paths from directory - same as eval_scannet.py"""
    image_paths = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        image_paths.extend(sorted(images_dir.glob(ext)))
    return image_paths


def load_poses(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load poses from directory - same as eval_scannet.py"""
    # Read all txt files from pose directory
    pose_files = sorted(
        path.glob("*.txt"), key=lambda x: int(x.stem)
    )  # Sort by numerical order

    # Check if pose files exist
    if len(pose_files) == 0:
        logger.warning(f"No pose files (.txt) found in directory {path}")
        return None, None, None

    c2ws = []
    available_frame_ids = []

    for pose_file in pose_files:
        try:
            with open(pose_file, "r") as f:
                # Each file contains 16 numbers representing a 4x4 transformation matrix
                nums = [float(x) for x in f.read().strip().split()]
                pose = np.array(nums).reshape(4, 4)
                # Check if pose is valid (no infinite or NaN values)
                if not (np.isinf(pose).any() or np.isnan(pose).any()):
                    c2ws.append(pose)
                    available_frame_ids.append(int(pose_file.stem))
                else:
                    continue
        except Exception as e:
            logger.error(f"Error reading pose file {pose_file}: {e}")
            continue

    if len(c2ws) == 0:
        logger.warning(f"No valid pose files found in directory {path}")
        return None, None, None

    c2ws = np.stack(c2ws)
    available_frame_ids = np.array(available_frame_ids)

    # Transform all poses to first frame coordinate system
    first_gt_pose = c2ws[0].copy()  # Save original pose of first frame
    c2ws = np.linalg.inv(c2ws[0]) @ c2ws
    return c2ws, first_gt_pose, available_frame_ids


def build_frame_selection(
    image_paths: List[Path],
    available_pose_frame_ids: np.ndarray,
    input_frame: int,
) -> Tuple[List[int], List[Path], List[int]]:
    """
    Frame selection logic - EXACTLY same as eval_scannet.py
    
    Args:
        image_paths: List of all image paths
        available_pose_frame_ids: Array of frame IDs that have valid poses
        input_frame: Maximum number of frames to select
        
    Returns:
        selected_frame_ids: List of selected frame IDs
        selected_image_paths: List of selected image paths
        selected_pose_indices: List of pose indices for selected frames
    """
    all_image_frame_ids = [int(path.stem) for path in image_paths]
    valid_frame_ids = sorted(
        list(set(all_image_frame_ids) & set(available_pose_frame_ids))
    )
    
    if len(valid_frame_ids) > input_frame:
        first_frame = valid_frame_ids[0]
        remaining_frames = valid_frame_ids[1:]
        step = max(1, len(remaining_frames) // (input_frame - 1))
        selected_remaining = remaining_frames[::step][: input_frame - 1]
        selected_frame_ids = [first_frame] + selected_remaining
    else:
        selected_frame_ids = valid_frame_ids

    frame_id_to_path = {int(path.stem): path for path in image_paths}
    selected_image_paths = [
        frame_id_to_path[fid] for fid in selected_frame_ids if fid in frame_id_to_path
    ]

    pose_frame_to_idx = {fid: idx for idx, fid in enumerate(available_pose_frame_ids)}
    selected_pose_indices = [
        pose_frame_to_idx[fid] for fid in selected_frame_ids if fid in pose_frame_to_idx
    ]

    return selected_frame_ids, selected_image_paths, selected_pose_indices


def create_video_from_frames(
    image_paths: List[Path],
    output_path: Path,
    fps: int = 30,
    codec: str = "libx264",
    quality: str = "high",
) -> bool:
    """
    Create video from frame images using ffmpeg
    
    Args:
        image_paths: List of image paths in order
        output_path: Output video file path
        fps: Frames per second
        codec: Video codec (default: libx264 for H.264)
        quality: Quality preset (high, medium, low)
        
    Returns:
        Success status
    """
    if not image_paths:
        logger.error("No images provided for video creation")
        return False
        
    # Create temporary directory for symlinked frames
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Create sequential symlinks to ensure correct order
        for i, img_path in enumerate(image_paths):
            symlink_path = temp_dir / f"frame_{i:06d}{img_path.suffix}"
            if symlink_path.exists():
                symlink_path.unlink()
            symlink_path.symlink_to(img_path.resolve())
        
        # Build ffmpeg command
        first_frame = temp_dir / f"frame_000000{image_paths[0].suffix}"
        
        # Get frame dimensions
        test_img = cv2.imread(str(first_frame))
        if test_img is None:
            logger.error(f"Cannot read first frame: {first_frame}")
            return False
            
        height, width = test_img.shape[:2]
        logger.info(f"Video dimensions: {width}x{height}")
        
        # Quality presets
        quality_settings = {
            "high": {"crf": "18", "preset": "slow"},
            "medium": {"crf": "23", "preset": "medium"},
            "low": {"crf": "28", "preset": "fast"}
        }
        
        settings = quality_settings.get(quality, quality_settings["high"])
        
        # Construct ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", str(temp_dir / f"frame_*{image_paths[0].suffix}"),
            "-c:v", codec,
            "-crf", settings["crf"],
            "-preset", settings["preset"],
            "-pix_fmt", "yuv420p",  # For compatibility
            "-movflags", "+faststart",  # For web streaming
            str(output_path)
        ]
        
        logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
        
        # Execute ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Video created successfully: {output_path}")
            return True
        else:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating video: {e}")
        return False
    finally:
        # Cleanup temporary directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def create_frames_zip(
    selected_image_paths: List[Path],
    selected_frame_ids: List[int],
    output_path: Path,
    scene_name: str,
) -> bool:
    """
    Create a ZIP file containing all selected frames
    
    Args:
        selected_image_paths: List of selected image paths
        selected_frame_ids: List of corresponding frame IDs
        output_path: Output ZIP file path
        scene_name: Name of the scene
        
    Returns:
        Success status
    """
    if not selected_image_paths:
        logger.error("No images to add to ZIP file")
        return False
    
    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add each selected frame to the ZIP
            for i, (img_path, frame_id) in enumerate(zip(selected_image_paths, selected_frame_ids)):
                if img_path.exists():
                    # Create a sequential filename for easy ordering
                    arcname = f"{scene_name}_frame_{i:06d}_id_{frame_id:06d}{img_path.suffix}"
                    zipf.write(img_path, arcname)
                    logger.debug(f"Added {img_path.name} as {arcname}")
                else:
                    logger.warning(f"Image file not found: {img_path}")
            
            # Add a manifest file with frame information
            manifest = {
                "scene_name": scene_name,
                "total_frames": len(selected_image_paths),
                "frames": [
                    {
                        "sequence_index": i,
                        "frame_id": int(frame_id),
                        "original_filename": img_path.name,
                        "zip_filename": f"{scene_name}_frame_{i:06d}_id_{frame_id:06d}{img_path.suffix}"
                    }
                    for i, (img_path, frame_id) in enumerate(zip(selected_image_paths, selected_frame_ids))
                ],
                "creation_timestamp": datetime.now().isoformat(),
                "zip_creation_notes": "Frames selected using eval_scannet.py logic"
            }
            
            zipf.writestr(f"{scene_name}_manifest.json", json.dumps(manifest, indent=2))
            logger.info(f"Added manifest file with frame metadata")
        
        logger.info(f"ZIP file created successfully: {output_path}")
        logger.info(f"ZIP file size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        logger.error(f"Error creating ZIP file: {e}")
        return False


def save_frame_metadata(
    selected_frame_ids: List[int],
    selected_image_paths: List[Path],
    selected_pose_indices: List[int],
    output_path: Path,
    scene_name: str,
    input_frame_param: int,
) -> None:
    """Save metadata about selected frames"""
    metadata = {
        "scene_name": scene_name,
        "timestamp": datetime.now().isoformat(),
        "total_frames_requested": input_frame_param,
        "total_frames_selected": len(selected_frame_ids),
        "frame_selection_method": "first_frame + uniform_sampling",
        "frames": []
    }
    
    for i, (frame_id, img_path, pose_idx) in enumerate(zip(
        selected_frame_ids, selected_image_paths, selected_pose_indices
    )):
        frame_info = {
            "sequence_index": i,
            "frame_id": int(frame_id),
            "image_path": str(img_path),
            "pose_index": int(pose_idx),
            "timestamp": datetime.now().isoformat()
        }
        metadata["frames"].append(frame_info)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Frame metadata saved to: {output_path}")


def process_scene(
    scene_dir: Path,
    output_dir: Path,
    input_frames: int,
    fps: int = 30,
    quality: str = "high",
    create_zip: bool = True,
    skip_video: bool = False,
) -> bool:
    """
    Process a single scene: select frames and create video
    
    Args:
        scene_dir: Directory containing scene data
        output_dir: Output directory for videos and metadata
        input_frames: Number of frames to select
        fps: Video frame rate
        quality: Video quality preset
        create_zip: Whether to create ZIP file with frames
        skip_video: Whether to skip video creation
        
    Returns:
        Success status
    """
    scene_name = scene_dir.name
    logger.info(f"Processing scene: {scene_name}")
    
    try:
        # Load scene data
        images_dir = scene_dir / "color"
        pose_path = scene_dir / "pose"
        
        if not images_dir.exists():
            logger.error(f"Images directory not found: {images_dir}")
            return False
            
        if not pose_path.exists():
            logger.error(f"Pose directory not found: {pose_path}")
            return False
        
        # Get image paths
        image_paths = get_sorted_image_paths(images_dir)
        if not image_paths:
            logger.error(f"No images found in {images_dir}")
            return False
        
        logger.info(f"Found {len(image_paths)} total images")
        
        # Load poses
        poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)
        if poses_gt is None or first_gt_pose is None or available_pose_frame_ids is None:
            logger.error(f"Failed to load poses for scene {scene_name}")
            return False
        
        logger.info(f"Found {len(available_pose_frame_ids)} valid poses")
        
        # Select frames using exact same logic as eval_scannet.py
        selected_frame_ids, selected_image_paths, selected_pose_indices = build_frame_selection(
            image_paths, available_pose_frame_ids, input_frames
        )
        
        if not selected_image_paths:
            logger.error(f"No frames selected for scene {scene_name}")
            return False
        
        logger.info(f"Selected {len(selected_frame_ids)} frames")
        logger.info(f"Selected frame IDs: {selected_frame_ids}")
        
        # Create output directory for this scene
        scene_output_dir = output_dir / f"input_frame_{input_frames}" / scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = None
        
        # Create video (unless skipped)
        if not skip_video:
            video_path = scene_output_dir / f"{scene_name}_frames_{len(selected_frame_ids)}.mp4"
            success = create_video_from_frames(
                selected_image_paths, video_path, fps=fps, quality=quality
            )
            
            if not success:
                logger.error(f"Failed to create video for scene {scene_name}")
                return False
        
        # Save metadata
        metadata_path = scene_output_dir / f"{scene_name}_metadata.json"
        save_frame_metadata(
            selected_frame_ids,
            selected_image_paths,
            selected_pose_indices,
            metadata_path,
            scene_name,
            input_frames,
        )
        
        # Create ZIP file with selected frames (if requested)
        zip_path = None
        if create_zip:
            zip_path = scene_output_dir / f"{scene_name}_selected_frames.zip"
            zip_success = create_frames_zip(
                selected_image_paths,
                selected_frame_ids,
                zip_path,
                scene_name,
            )
            
            if zip_success:
                logger.info(f"ZIP file created for download: {zip_path}")
            else:
                logger.warning(f"Failed to create ZIP file for scene {scene_name}")
        
        # Log summary
        logger.info(f"Successfully processed scene: {scene_name}")
        if video_path:
            logger.info(f"  Video: {video_path}")
        if zip_path:
            logger.info(f"  ZIP: {zip_path}")
        logger.info(f"  Metadata: {metadata_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing scene {scene_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Frame Selection and Video Synthesis")
    parser.add_argument(
        "--data_dir", 
        type=Path, 
        required=True,
        help="Directory containing scene data"
    )
    parser.add_argument(
        "--output_path", 
        type=Path, 
        default="./video_output",
        help="Output directory for videos"
    )
    parser.add_argument(
        "--input_frame",
        type=int,
        default=200,
        help="Maximum number of frames to select per scene"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate (default: 30)"
    )
    parser.add_argument(
        "--quality",
        type=str,
        choices=["high", "medium", "low"],
        default="high",
        help="Video quality preset (default: high)"
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process (default: all)"
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="Process specific scene only"
    )
    parser.add_argument(
        "--create_zip",
        action="store_true",
        default=True,
        help="Create ZIP file with selected frames (default: True)"
    )
    parser.add_argument(
        "--skip_video",
        action="store_true",
        default=False,
        help="Skip video creation (only create ZIP and metadata)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_path.mkdir(parents=True, exist_ok=True)
    
    # Get scenes to process
    if args.scene_name:
        # Process specific scene
        scene_dirs = [args.data_dir / args.scene_name]
        if not scene_dirs[0].exists():
            logger.error(f"Scene directory not found: {scene_dirs[0]}")
            return
    else:
        # Process all scenes
        scene_dirs = sorted([d for d in args.data_dir.iterdir() if d.is_dir()])
        if args.num_scenes:
            scene_dirs = scene_dirs[:args.num_scenes]
    
    logger.info(f"Processing {len(scene_dirs)} scenes")
    
    # Process each scene
    successful_scenes = 0
    failed_scenes = 0
    
    for scene_dir in scene_dirs:
        if process_scene(
            scene_dir,
            args.output_path,
            args.input_frame,
            fps=args.fps,
            quality=args.quality,
            create_zip=args.create_zip,
            skip_video=args.skip_video,
        ):
            successful_scenes += 1
        else:
            failed_scenes += 1
    
    logger.info(f"Processing complete!")
    logger.info(f"Successful scenes: {successful_scenes}")
    logger.info(f"Failed scenes: {failed_scenes}")


if __name__ == "__main__":
    main()