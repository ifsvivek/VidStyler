# Video Stabilization Module

This module provides a simple yet powerful interface for video stabilization using the `vidstab` library, wrapped in a convenient Python class.

## Overview

The video stabilization module reduces camera shake from videos by:

1. Tracking features across video frames
2. Computing transformations between consecutive frames
3. Smoothing the camera trajectory
4. Applying the smoothed transformations to create a stabilized video

## Features

-   Multiple keypoint detection methods:

    -   GFTT (Good Features to Track) - Default, works well for most videos
    -   SIFT - High accuracy but slower
    -   SURF - Good balance of speed and accuracy
    -   ORB - Fast and efficient
    -   BRISK - Fast binary feature detector
    -   FAST - Very fast corner detection

-   Customizable stabilization parameters:

    -   Smoothing radius (window size)
    -   Border handling (black, reflect, replicate)
    -   Border size control

-   Advanced features:
    -   Layer effects (overlay, blend) for artistic motion trails
    -   Save/load transformation matrices
    -   Plot trajectory and transform visualizations
    -   Frame-by-frame processing capability


## UI

![Video Stabilization UI](../../img/VisualAlchemy2-1.png)

## Usage Examples

### Basic Usage

```python
from src.video_stabilization.vidstab import VidStabWrapper

# Initialize with default parameters (GFTT keypoint detector)
stabilizer = VidStabWrapper()

# Stabilize a video
stabilizer.stabilize(
    input_path="shaky_video.mp4",
    output_path="stabilized_video.mp4",
    smoothing_window=30,
    border_type="black"
)
```

### Advanced Usage

```python
from src.video_stabilization.vidstab import VidStabWrapper, get_layer_blend

# Initialize with SIFT keypoint detector
stabilizer = VidStabWrapper(kp_method="SIFT")

# Generate transforms first
stabilizer.gen_transforms(
    input_path="shaky_video.mp4",
    smoothing_window=45
)

# Save transforms for later use
stabilizer.save_transforms("transforms.csv")

# Apply transforms with a blend effect
stabilizer.apply_transforms(
    input_path="shaky_video.mp4",
    output_path="stabilized_with_trails.mp4",
    border_type="reflect",
    border_size=100,
    layer_func=lambda fg, bg: get_layer_blend(fg, bg, alpha=0.7)
)

# Create plots for analysis
trajectory_fig = stabilizer.plot_trajectory()
transforms_fig = stabilizer.plot_transforms()
trajectory_fig.savefig("trajectory.png")
transforms_fig.savefig("transforms.png")
```

## Tips for Best Results

1. **Smoothing Window Size**:

    - Smaller values (10-20): Retains more intentional motion, less stable
    - Medium values (30-50): Good balance for most videos
    - Larger values (60-100): Very smooth but may crop more of the frame

2. **Border Handling**:

    - "black": Best for most cases as it clearly shows the stabilization effect
    - "reflect": Good for nature scenes where black borders would be distracting
    - "replicate": Works well for sky or other uniform backgrounds

3. **Keypoint Methods**:

    - Try GFTT first as it works well in most cases
    - For fast-moving videos, try FAST or ORB
    - For complex scenes with lots of details, SIFT may provide better results

4. **Layer Effects**:
    - Use overlay for artistic "motion trails"
    - Use blend with alpha 0.3-0.7 for soft motion blur effect

## Internal Implementation

The module consists of two main classes:

-   `VidStabWrapper`: A high-level wrapper around the vidstab library
-   Helper functions for layer effects

The implementation handles edge cases like:

-   Videos with few trackable features
-   Error handling for corrupted video files
-   Memory-efficient processing of large videos

## Requirements

-   vidstab
-   OpenCV (cv2)
-   NumPy
-   Matplotlib (for plotting)

## Limitations

-   Very fast camera movements may result in excessive cropping
-   Processing time increases with video resolution and duration
-   Extremely shaky videos may not be perfectly stabilized
