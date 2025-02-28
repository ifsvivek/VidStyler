# Video Stabilization Module

This module provides algorithms for stabilizing shaky videos. It implements both traditional computer vision methods and deep learning approaches to reduce unwanted motion and create smoother videos.

## Features

-   Motion estimation using feature tracking and homography
-   Trajectory smoothing with various filters (moving average, Gaussian, etc.)
-   Motion compensation via frame warping
-   Adaptive cropping to remove border artifacts
-   Support for different stabilization strengths
-   Processing of videos with different resolutions and frame rates

## Implementation Details

The video stabilization process follows these main steps:

1. **Motion Estimation**: Track features (like SIFT or ORB) between consecutive frames to estimate the camera motion
2. **Motion Smoothing**: Apply filters to smooth the estimated motion trajectory
3. **Motion Compensation**: Warp the frames according to the difference between original and smoothed motion
4. **Border Removal**: Crop or fill borders to remove artifacts from the stabilization process

## Stabilization Methods

The module supports different stabilization approaches:

### Feature-based Method

-   Uses feature detection and matching to estimate frame-to-frame transformations
-   Supports both rigid (translation, rotation) and non-rigid transformations
-   Good for videos with distinct features

### Optical Flow Method

-   Uses dense optical flow to estimate motion between frames
-   Better for videos with textureless regions
-   More computationally intensive

### L1-Optimal Method

-   Implements L1-norm optimization for trajectory smoothing
-   Produces more robust results for videos with complex motion

## Usage

```python
from video_stabilization import VideoStabilizer

# Create stabilizer with desired parameters
stabilizer = VideoStabilizer(
    smoothing_radius=30,  # Number of frames for smoothing window
    border_mode='crop',   # How to handle borders: 'crop', 'black', or 'reflect'
    stabilization_method='feature'  # 'feature', 'flow', or 'l1'
)

# Stabilize a video
stabilizer.stabilize(
    input_path="path/to/input_video.mp4",
    output_path="path/to/stabilized_video.mp4"
)
```

## References

-   Grundmann, M., Kwatra, V., & Essa, I. (2011). Auto-directed video stabilization with robust L1 optimal camera paths.
-   Liu, S., Yuan, L., Tan, P., & Sun, J. (2013). Bundled camera paths for video stabilization.
