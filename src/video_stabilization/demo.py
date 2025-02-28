import os
import argparse
import matplotlib.pyplot as plt
import tempfile
from .vidstab import VidStab
from .utils import download_sample_video

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Video Stabilization Demo")
    parser.add_argument("--input", "-i", help="Input video path (if not specified, a sample video will be downloaded)")
    parser.add_argument("--output", "-o", help="Output video path", default="stabilized_output.avi")
    parser.add_argument("--smoothing", "-s", help="Smoothing radius", type=int, default=30)
    parser.add_argument("--border", "-b", help="Border type (black, reflect, replicate)", default="black")
    parser.add_argument("--keypoint", "-k", help="Keypoint detection method (GFTT, FAST, ORB, SIFT)", default="GFTT")
    parser.add_argument("--plot", "-p", help="Show plots of trajectory and transforms", action="store_true")
    args = parser.parse_args()
    
    # Get input video
    if args.input:
        input_video = args.input
    else:
        # Download sample video
        print("Downloading sample video...")
        sample_video = os.path.join(tempfile.gettempdir(), "sample_video.mp4")
        input_video = download_sample_video(sample_video)
    
    # Initialize video stabilizer with specified keypoint detection method
    stabilizer = VidStab(kp_method=args.keypoint)
    
    # Stabilize video
    print(f"Stabilizing video with {args.keypoint} keypoint detector and {args.smoothing} frame smoothing window...")
    stabilizer.stabilize(
        input_path=input_video,
        output_path=args.output,
        smoothing_radius=args.smoothing,
        border_type=args.border,
        border_size='auto',
        show_progress=True
    )
    
    # Plot trajectory and transforms if requested
    if args.plot:
        print("Generating trajectory plot...")
        trajectory_plot = stabilizer.plot_trajectory()
        trajectory_plot.savefig('trajectory_plot.png')
        
        print("Generating transform plot...")
        transform_plot = stabilizer.plot_transforms()
        transform_plot.savefig('transforms_plot.png')
        
        plt.show()
    
    print(f"Stabilization complete! Output saved to {args.output}")

if __name__ == "__main__":
    main()
