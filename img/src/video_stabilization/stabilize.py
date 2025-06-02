import tempfile
import traceback
import matplotlib.pyplot as plt
from src.video_stabilization.vidstab import (
    VidStabWrapper,
    get_layer_overlay,
    get_layer_blend,
)

def stabilize_video(
    video_path,
    kp_method,
    smoothing_radius,
    border_type,
    border_size,
    use_layer_effect=False,
    layer_effect_type="overlay",
    layer_alpha=0.5,
    show_plots=True,
):
    """
    Stabilize a video using VidStab

    Args:
        video_path: Path to input video
        kp_method: Keypoint detection method
        smoothing_radius: Radius of smoothing window
        border_type: Type of border handling
        border_size: Size of border
        use_layer_effect: Whether to use layer effects
        layer_effect_type: Type of layer effect
        layer_alpha: Alpha for blend effect
        show_plots: Whether to generate plots

    Returns:
        stabilized_video_path: Path to stabilized video
        trajectory_plot_path: Path to trajectory plot
        transforms_plot_path: Path to transforms plot
        message: Status message
    """
    try:
        if video_path is None:
            return None, None, None, "Please upload a video first."

        # Create temporary output path
        output_path = tempfile.mktemp(suffix=".mp4")

        # Initialize stabilizer with the selected keypoint method
        stabilizer = VidStabWrapper(kp_method=kp_method)

        # Set up layer function if enabled
        layer_func = None
        if use_layer_effect:
            if layer_effect_type == "overlay":
                layer_func = get_layer_overlay
            elif layer_effect_type == "blend":
                # Create a closure to pass the alpha value
                def custom_blend(foreground, background):
                    return get_layer_blend(foreground, background, alpha=layer_alpha)

                layer_func = custom_blend

        # Generate transforms first (separate step for better error handling)
        print(
            f"Generating transforms with {kp_method} keypoints and smoothing window of {smoothing_radius}..."
        )

        # Generate transforms
        success = stabilizer.gen_transforms(
            input_path=video_path, 
            smoothing_window=int(smoothing_radius),
        )

        if not success:
            return (
                None,
                None,
                None,
                "Failed to generate transforms. Try a different keypoint method or check the video file.",
            )

        # Apply transforms to create the stabilized video
        print(
            f"Applying transforms with border type {border_type} and border size {border_size}..."
        )
        
        # Keep border_size as string "auto" or convert to int if it's a numeric string
        processed_border_size = border_size
        if border_size != "auto":
            try:
                processed_border_size = int(border_size)
            except ValueError:
                print(f"Warning: Could not convert border_size '{border_size}' to int, using 'auto'")
                processed_border_size = "auto"
        
        success = stabilizer.apply_transforms(
            input_path=video_path,
            output_path=output_path,
            border_type=border_type,
            border_size=processed_border_size,
            layer_func=layer_func,
        )

        if not success:
            return (
                None,
                None,
                None,
                "Failed to apply transforms. Check logs for details.",
            )

        # Generate plots if requested
        trajectory_plot_path = None
        transforms_plot_path = None
        if show_plots:
            try:
                # Generate trajectory plot
                trajectory_fig = stabilizer.plot_trajectory()
                trajectory_plot_path = tempfile.mktemp(suffix=".png")
                trajectory_fig.savefig(trajectory_plot_path)
                plt.close(trajectory_fig)

                # Generate transforms plot
                transforms_fig = stabilizer.plot_transforms()
                transforms_plot_path = tempfile.mktemp(suffix=".png")
                transforms_fig.savefig(transforms_plot_path)
                plt.close(transforms_fig)
            except Exception as e:
                print(f"Error generating plots: {traceback.format_exc()}")
                return (
                    output_path,
                    None,
                    None,
                    f"Video stabilized, but error generating plots: {str(e)}",
                )

        return (
            output_path,
            trajectory_plot_path,
            transforms_plot_path,
            "Video stabilization completed successfully.",
        )

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error during video stabilization: {error_details}")
        return None, None, None, f"Error during video stabilization: {str(e)}"
