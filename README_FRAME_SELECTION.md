# Frame Selection and Video Synthesis Tool

This tool replicates the exact frame selection logic from `eval_scannet.py` and provides additional functionality to create videos and ZIP files containing the selected frames.

## Features

- **Exact Frame Selection**: Uses the same frame selection algorithm as `eval_scannet.py`
- **Video Creation**: Synthesizes selected frames into H.264 encoded videos
- **ZIP File Generation**: Creates downloadable ZIP files containing selected frames
- **Metadata Tracking**: Saves detailed information about selected frames
- **Flexible Processing**: Process single scenes or multiple scenes
- **Quality Options**: Multiple video quality presets (high, medium, low)

## Installation

```bash
# Install Python dependencies
pip install numpy opencv-python

# Install ffmpeg (required for video creation)
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html
```

## Usage

### Basic Usage

```bash
# Process multiple scenes with default settings
python frame_selection_video_synthesis.py \
    --data_dir /path/to/scannet/data \
    --output_path ./video_output \
    --input_frame 200 \
    --num_scenes 5
```

### Process Single Scene

```bash
# Process specific scene with custom frame count
python frame_selection_video_synthesis.py \
    --data_dir /path/to/scannet/data \
    --output_path ./video_output \
    --input_frame 50 \
    --scene_name scene0000_00
```

### Create ZIP File Only (No Video)

```bash
# Create only ZIP files with selected frames
python frame_selection_video_synthesis.py \
    --data_dir /path/to/scannet/data \
    --output_path ./frames_output \
    --input_frame 100 \
    --scene_name scene0000_00 \
    --skip_video \
    --create_zip
```

### Advanced Options

```bash
# Custom video settings
python frame_selection_video_synthesis.py \
    --data_dir /path/to/scannet/data \
    --output_path ./video_output \
    --input_frame 150 \
    --fps 24 \
    --quality medium \
    --num_scenes 10
```

## Frame Selection Algorithm

The tool uses the exact same frame selection logic as `eval_scannet.py`:

1. **Pose Validation**: Only considers frames that have valid pose data
2. **First Frame Priority**: Always includes the first valid frame
3. **Uniform Sampling**: Distributes remaining frames uniformly across the sequence
4. **Frame ID Matching**: Ensures selected frames have corresponding pose data

### Algorithm Steps:

```python
def build_frame_selection(image_paths, available_pose_frame_ids, input_frame):
    # 1. Find intersection of image frames and pose frames
    valid_frame_ids = sorted(list(set(image_frame_ids) & set(pose_frame_ids)))
    
    # 2. If more frames than requested, apply uniform sampling
    if len(valid_frame_ids) > input_frame:
        first_frame = valid_frame_ids[0]  # Always include first frame
        remaining_frames = valid_frame_ids[1:]
        step = max(1, len(remaining_frames) // (input_frame - 1))
        selected_remaining = remaining_frames[::step][:input_frame - 1]
        selected_frame_ids = [first_frame] + selected_remaining
    else:
        selected_frame_ids = valid_frame_ids
    
    return selected_frame_ids, selected_image_paths, selected_pose_indices
```

## Output Structure

```
output_path/
└── input_frame_XXX/
    └── scene_name/
        ├── scene_name_frames_N.mp4          # Video file (if not skipped)
        ├── scene_name_selected_frames.zip   # ZIP file with frames
        └── scene_name_metadata.json         # Frame metadata
```

### ZIP File Contents

Each ZIP file contains:
- All selected frame images with sequential naming
- Manifest file with frame metadata
- Original frame IDs preserved in filenames

Example ZIP structure:
```
scene0000_00_selected_frames.zip
├── scene0000_00_frame_000000_id_000000.jpg
├── scene0000_00_frame_000001_id_000010.jpg
├── scene0000_00_frame_000002_id_000020.jpg
└── scene0000_00_manifest.json
```

### Metadata Format

```json
{
  "scene_name": "scene0000_00",
  "timestamp": "2024-01-01T12:00:00",
  "total_frames_requested": 200,
  "total_frames_selected": 150,
  "frame_selection_method": "first_frame + uniform_sampling",
  "frames": [
    {
      "sequence_index": 0,
      "frame_id": 0,
      "image_path": "/path/to/original/image.jpg",
      "pose_index": 0,
      "timestamp": "2024-01-01T12:00:00"
    }
  ]
}
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data_dir` | Required | Directory containing scene data |
| `--output_path` | `./video_output` | Output directory for results |
| `--input_frame` | `200` | Maximum frames to select per scene |
| `--fps` | `30` | Video frame rate |
| `--quality` | `high` | Video quality (high/medium/low) |
| `--num_scenes` | All | Maximum scenes to process |
| `--scene_name` | None | Process specific scene only |
| `--create_zip` | `True` | Create ZIP file with frames |
| `--skip_video` | `False` | Skip video creation |

## Video Quality Settings

| Quality | CRF | Preset | File Size | Quality |
|---------|-----|--------|-----------|---------|
| high | 18 | slow | Larger | Best |
| medium | 23 | medium | Medium | Good |
| low | 28 | fast | Smaller | Acceptable |

## Validation

To validate that frame selection matches `eval_scannet.py`:

```bash
python validate_frame_selection.py \
    --data_dir /path/to/scannet/data \
    --input_frame 200 \
    --output validation_results.json
```

## Examples

### Research Use Case
```bash
# Create high-quality videos for evaluation
python frame_selection_video_synthesis.py \
    --data_dir /path/to/scannet/data \
    --output_path ./evaluation_videos \
    --input_frame 200 \
    --quality high \
    --fps 30
```

### Data Export Use Case
```bash
# Export frames as ZIP files for annotation
python frame_selection_video_synthesis.py \
    --data_dir /path/to/scannet/data \
    --output_path ./exported_frames \
    --input_frame 100 \
    --skip_video \
    --create_zip
```

### Quick Preview Use Case
```bash
# Create low-quality videos for quick preview
python frame_selection_video_synthesis.py \
    --data_dir /path/to/scannet/data \
    --output_path ./preview_videos \
    --input_frame 50 \
    --quality low \
    --fps 15
```

## Error Handling

The tool includes comprehensive error handling for:
- Missing directories or files
- Invalid pose data
- FFmpeg execution errors
- ZIP file creation failures
- Insufficient frames for selection

## Performance Notes

- **Frame Selection**: Very fast, processes scenes in seconds
- **Video Creation**: Depends on resolution and frame count (typically 1-5 minutes per scene)
- **ZIP Creation**: Fast for typical frame counts, scales with total image size
- **Memory Usage**: Minimal, processes frames sequentially

## Troubleshooting

### Common Issues:

1. **"No pose files found"**
   - Ensure pose directory contains `.txt` files
   - Check file naming convention matches frame IDs

2. **"FFmpeg not found"**
   - Install ffmpeg system-wide
   - Add ffmpeg to system PATH

3. **"No images found"**
   - Check image directory contains `.jpg`, `.png`, or `.jpeg` files
   - Verify image file extensions

4. **"Insufficient frames"**
   - Reduce `--input_frame` parameter
   - Check for valid pose-image intersection

## Download Instructions

After processing, ZIP files are available at:
```
output_path/input_frame_XXX/scene_name/scene_name_selected_frames.zip
```

These ZIP files can be:
- Downloaded directly for local use
- Shared with annotation teams
- Used for frame-by-frame analysis
- Imported into other tools

## License

This tool follows the same license as the original VGGT project.