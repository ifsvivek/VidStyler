import numpy as np
import cv2
import os
import tempfile
import shutil

def layer_overlay(foreground, background):
    """
    Create a simple overlay effect by keeping the brightest pixels
    
    Args:
        foreground: Current frame
        background: Previous frame
        
    Returns:
        result: Combined frame
    """
    if foreground.shape != background.shape:
        return foreground
        
    return np.maximum(foreground, background)

def layer_blend(foreground, background, foreground_alpha=0.7):
    """
    Blend frames together with specified alpha
    
    Args:
        foreground: Current frame
        background: Previous frame
        foreground_alpha: Alpha value for foreground (0.0-1.0)
        
    Returns:
        result: Blended frame
    """
    if foreground.shape != background.shape:
        return foreground
        
    return cv2.addWeighted(foreground, foreground_alpha, 
                          background, 1 - foreground_alpha, 0)

def download_sample_video(output_path, url=None):
    """
    Download a sample video for testing stabilization
    
    Args:
        output_path: Path to save downloaded video
        url: URL to download from (optional)
        
    Returns:
        output_path: Path to downloaded video
    """
    import requests
    from tqdm import tqdm
    
    if url is None:
        # Default sample video URL
        url = "https://github.com/AdamSpannbauer/python_video_stab/blob/master/readme_assets/ostrich.mp4?raw=true"
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kb
    
    with open(output_path, 'wb') as file, tqdm(
        desc="Downloading sample video", 
        total=total_size, 
        unit='B', 
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)
    
    return output_path

def video_to_frames(video_path, output_dir=None):
    """
    Extract frames from video
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames (None for temporary directory)
        
    Returns:
        output_dir: Directory containing frames
        fps: Frames per second of video
    """
    # Create temporary directory if not provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Extract frames
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        cv2.imwrite(os.path.join(output_dir, f"frame_{count:06d}.jpg"), frame)
        count += 1
    
    # Release video
    cap.release()
    
    return output_dir, fps

def frames_to_video(frames_dir, output_path, fps=30):
    """
    Create video from frames
    
    Args:
        frames_dir: Directory containing frames
        output_path: Path to output video
        fps: Frames per second
        
    Returns:
        output_path: Path to output video
    """
    # Get list of frames
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    if len(frames) == 0:
        raise ValueError("No frames found in directory")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame in frames:
        img = cv2.imread(os.path.join(frames_dir, frame))
        writer.write(img)
    
    # Release video writer
    writer.release()
    
    return output_path

def plot_trajectory(trajectory, smoothed_trajectory=None):
    """
    Plot the original and smoothed trajectory
    
    Args:
        trajectory: Original trajectory as numpy array
        smoothed_trajectory: Smoothed trajectory as numpy array (optional)
        
    Returns:
        fig: Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    if trajectory is None or len(trajectory) == 0:
        raise ValueError("No trajectory to plot")
    
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot x trajectory
    ax1.plot(trajectory[:, 0], 'b-', label='Original')
    if smoothed_trajectory is not None:
        ax1.plot(smoothed_trajectory[:, 0], 'r-', label='Smoothed')
    ax1.set_title('X-Trajectory')
    ax1.set_ylabel('dx (pixels)')
    ax1.legend()
    ax1.grid()
    
    # Plot y trajectory
    ax2.plot(trajectory[:, 1], 'b-', label='Original')
    if smoothed_trajectory is not None:
        ax2.plot(smoothed_trajectory[:, 1], 'r-', label='Smoothed')
    ax2.set_title('Y-Trajectory')
    ax2.set_ylabel('dy (pixels)')
    ax2.legend()
    ax2.grid()
    
    # Plot angle trajectory
    ax3.plot(np.rad2deg(trajectory[:, 2]), 'b-', label='Original')
    if smoothed_trajectory is not None:
        ax3.plot(np.rad2deg(smoothed_trajectory[:, 2]), 'r-', label='Smoothed')
    ax3.set_title('Angle Trajectory')
    ax3.set_ylabel('da (degrees)')
    ax3.set_xlabel('Frame Number')
    ax3.legend()
    ax3.grid()
    
    plt.tight_layout()
    return fig

def save_transforms_to_file(transforms, output_path):
    """
    Save transforms to a CSV file
    
    Args:
        transforms: List of transformation matrices
        output_path: Path to output file
        
    Returns:
        output_path: Path to file containing transforms
    """
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for transform in transforms:
            # Flatten transform to 6 values (2x3 matrix)
            writer.writerow(transform.flatten())
    
    return output_path

def read_transforms_from_file(input_path):
    """
    Read transforms from a CSV file
    
    Args:
        input_path: Path to file containing transforms
        
    Returns:
        transforms: List of transformation matrices
    """
    import csv
    transforms = []
    
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Convert string values to float
            values = [float(val) for val in row]
            
            # Convert back to 2x3 matrix
            transform = np.array(values).reshape(2, 3)
            transforms.append(transform)
    
    return transforms