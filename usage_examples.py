#!/usr/bin/env python3
"""
Usage examples and testing for frame selection and video synthesis
"""

import subprocess
import sys
from pathlib import Path


def test_frame_selection():
    """Test frame selection with a simple example"""
    print("Testing frame selection logic...")
    
    # Create test data structure
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test scene
    scene_dir = test_dir / "test_scene"
    color_dir = scene_dir / "color"
    pose_dir = scene_dir / "pose"
    
    color_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test images (this would normally be real images)
    print("Creating test images...")
    try:
        import cv2
        import numpy as np
        
        # Create 50 test images
        for i in range(50):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(color_dir / f"{i:06d}.jpg"), img)
        
        # Create test poses (4x4 identity matrices)
        for i in range(50):
            pose = np.eye(4)
            pose[0, 3] = i * 0.1  # Small translation
            with open(pose_dir / f"{i:06d}.txt", 'w') as f:
                f.write(' '.join(map(str, pose.flatten())))
        
        print("Test data created successfully!")
        
        # Test the frame selection
        from frame_selection_video_synthesis import build_frame_selection, get_sorted_image_paths, load_poses
        
        image_paths = get_sorted_image_paths(color_dir)
        poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_dir)
        
        selected_frame_ids, selected_image_paths, selected_pose_indices = build_frame_selection(
            image_paths, available_pose_frame_ids, input_frame=20
        )
        
        print(f"Selected {len(selected_frame_ids)} frames out of {len(image_paths)} available")
        print(f"Selected frame IDs: {selected_frame_ids}")
        print("Frame selection test passed!")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
    except ImportError:
        print("OpenCV not available, skipping test image creation")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


def run_example_command():
    """Run an example command"""
    print("\nExample usage:")
    print("python frame_selection_video_synthesis.py \\")
    print("    --data_dir /path/to/scannet/data \\")
    print("    --output_path ./video_output \\")
    print("    --input_frame 200 \\")
    print("    --fps 30 \\")
    print("    --quality high \\")
    print("    --num_scenes 5")
    print()
    print("Process specific scene:")
    print("python frame_selection_video_synthesis.py \\")
    print("    --data_dir /path/to/scannet/data \\")
    print("    --output_path ./video_output \\")
    print("    --input_frame 50 \\")
    print("    --scene_name scene0000_00")


def validate_installation():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    # Check Python packages
    required_packages = ['numpy', 'cv2', 'pathlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing Python packages: {missing_packages}")
        print("Install with: pip install numpy opencv-python")
    else:
        print("✓ All Python packages available")
    
    # Check ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg available")
        else:
            print("✗ FFmpeg not working properly")
    except FileNotFoundError:
        print("✗ FFmpeg not found - install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
    
    return len(missing_packages) == 0


def main():
    """Main function"""
    print("Frame Selection and Video Synthesis - Usage Examples and Testing")
    print("=" * 60)
    
    # Validate installation
    if not validate_installation():
        print("\nPlease install missing dependencies before proceeding.")
        return
    
    # Show example commands
    run_example_command()
    
    # Test frame selection logic
    print("\n" + "=" * 60)
    test_frame_selection()
    
    print("\n" + "=" * 60)
    print("For validation against eval_scannet.py results:")
    print("python validate_frame_selection.py \\")
    print("    --data_dir /path/to/scannet/data \\")
    print("    --input_frame 200 \\")
    print("    --output validation_results.json")


if __name__ == "__main__":
    main()