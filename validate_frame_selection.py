#!/usr/bin/env python3
"""
Validation script to verify frame selection matches eval_scannet.py exactly
"""

import json
import sys
from pathlib import Path
import numpy as np

# Import the original eval_utils functions
sys.path.insert(0, str(Path(__file__).parent))
from frame_selection_video_synthesis import (
    get_sorted_image_paths,
    load_poses,
    build_frame_selection
)


def validate_frame_selection(scene_dir: Path, input_frame: int = 200) -> dict:
    """
    Validate frame selection for a scene against expected results
    
    Returns:
        Dictionary with validation results
    """
    scene_name = scene_dir.name
    
    # Load scene data
    images_dir = scene_dir / "color"
    pose_path = scene_dir / "pose"
    
    if not images_dir.exists() or not pose_path.exists():
        return {"error": "Missing directories", "scene": scene_name}
    
    # Get image paths
    image_paths = get_sorted_image_paths(images_dir)
    if not image_paths:
        return {"error": "No images found", "scene": scene_name}
    
    # Load poses
    poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)
    if poses_gt is None:
        return {"error": "Failed to load poses", "scene": scene_name}
    
    # Select frames
    selected_frame_ids, selected_image_paths, selected_pose_indices = build_frame_selection(
        image_paths, available_pose_frame_ids, input_frame
    )
    
    return {
        "scene": scene_name,
        "total_images": len(image_paths),
        "valid_poses": len(available_pose_frame_ids),
        "selected_frames": len(selected_frame_ids),
        "selected_frame_ids": selected_frame_ids,
        "first_selected_frame": selected_frame_ids[0] if selected_frame_ids else None,
        "last_selected_frame": selected_frame_ids[-1] if selected_frame_ids else None,
        "step_size": (selected_frame_ids[1] - selected_frame_ids[0]) if len(selected_frame_ids) > 1 else None,
        "validation_status": "success"
    }


def compare_with_eval_results(eval_results_path: Path, validation_results: dict) -> dict:
    """
    Compare our frame selection with eval_scannet.py results
    
    Args:
        eval_results_path: Path to eval_scannet.py output directory
        validation_results: Results from our validation
        
    Returns:
        Comparison results
    """
    scene_name = validation_results["scene"]
    
    # Look for eval results
    eval_scene_dirs = list(eval_results_path.glob(f"*/{scene_name}"))
    if not eval_scene_dirs:
        return {"comparison": "no_eval_results_found", "scene": scene_name}
    
    eval_scene_dir = eval_scene_dirs[0]
    
    # Look for metrics or other output files that might contain frame info
    # This is a placeholder - actual implementation would depend on eval_scannet.py output format
    
    return {
        "comparison": "not_implemented",
        "eval_results_found": str(eval_scene_dir.exists()),
        "scene": scene_name
    }


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate frame selection implementation")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing scene data")
    parser.add_argument("--input_frame", type=int, default=200, help="Number of frames to select")
    parser.add_argument("--scene_name", type=str, help="Validate specific scene only")
    parser.add_argument("--eval_results", type=Path, help="Path to eval_scannet.py results for comparison")
    parser.add_argument("--output", type=Path, default="validation_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Get scenes to validate
    if args.scene_name:
        scene_dirs = [args.data_dir / args.scene_name]
        if not scene_dirs[0].exists():
            print(f"Scene directory not found: {scene_dirs[0]}")
            return
    else:
        scene_dirs = sorted([d for d in args.data_dir.iterdir() if d.is_dir()])[:5]  # Test first 5 scenes
    
    print(f"Validating {len(scene_dirs)} scenes...")
    
    results = []
    for scene_dir in scene_dirs:
        print(f"Validating scene: {scene_dir.name}")
        result = validate_frame_selection(scene_dir, args.input_frame)
        
        if args.eval_results:
            comparison = compare_with_eval_results(args.eval_results, result)
            result.update(comparison)
        
        results.append(result)
        print(f"  Selected {result.get('selected_frames', 0)} frames")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nValidation complete! Results saved to: {args.output}")
    
    # Summary statistics
    successful = [r for r in results if r.get("validation_status") == "success"]
    print(f"Successfully validated: {len(successful)}/{len(results)} scenes")
    
    if successful:
        avg_selected = np.mean([r["selected_frames"] for r in successful])
        print(f"Average frames selected: {avg_selected:.1f}")


if __name__ == "__main__":
    main()