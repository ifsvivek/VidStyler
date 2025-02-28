import gradio as gr
import os
import time
import numpy as np
import cv2
from PIL import Image
import sys
import tempfile
import shutil
import matplotlib.pyplot as plt

# Add the project root to the path so we can import our modules
sys.path.append("/home/sic/Documentations/Vivek_FYP")
from src.style_transfer.neural_style import apply_neural_style_transfer
from src.video_stabilization.vidstab import VidStab
from src.video_stabilization.utils import video_to_frames, frames_to_video


def remove_object(image, mask=None, method="deepfill"):
    """
    Placeholder for object removal function.
    In a real implementation, this would remove the selected object and fill in the area.
    """
    # Simulate processing time
    time.sleep(2)

    # For now, just return the original image as a placeholder
    if mask is not None and image is not None:
        # Apply a simple blurring effect to simulate object removal
        try:
            result = image.copy()
            mask_np = mask.astype(np.uint8) * 255
            blurred = cv2.GaussianBlur(image, (25, 25), 0)
            mask_np = mask_np[:, :, np.newaxis] if len(mask_np.shape) == 2 else mask_np
            result = np.where(mask_np > 0, blurred, image)
            return result
        except Exception as e:
            print(f"Error in remove_object: {e}")
            return image
    return image


def create_style_transfer_tab():
    """Creates the style transfer tab UI elements."""
    with gr.Tab("Style Transfer"):
        gr.Markdown(
            """
        ## Style Transfer
        
        This tool applies the artistic style of one image to the content of another using neural style transfer techniques.
        
        Upload both a content image (the image you want to stylize) and a style image (the image with the artistic style you want to apply).
        """
        )

        with gr.Row():
            with gr.Column():
                neural_content_image = gr.Image(label="Content Image", type="numpy")
                neural_style_image = gr.Image(label="Style Image", type="numpy")

                with gr.Row():
                    style_weight = gr.Slider(
                        minimum=1e4,
                        maximum=1e7,
                        value=1e6,
                        step=1e4,
                        label="Style Weight",
                    )
                    content_weight = gr.Slider(
                        minimum=0.1,
                        maximum=10.0,
                        value=1.0,
                        step=0.1,
                        label="Content Weight",
                    )

                iterations = gr.Slider(
                    minimum=100, maximum=1000, value=300, step=50, label="Iterations"
                )

                neural_apply_button = gr.Button("Apply Neural Style Transfer")

            with gr.Column():
                neural_output_image = gr.Image(label="Stylized Output")
                neural_progress = gr.Textbox(label="Progress")

        # Function to update progress
        def update_progress(step, total_steps, loss):
            return f"Step {step}/{total_steps} - Loss: {loss:.2f}"

        # Function to apply neural style transfer
        def process_neural_style_transfer(
            content_img, style_img, style_w, content_w, iters
        ):
            if content_img is None or style_img is None:
                return None, "Please upload both content and style images."

            try:
                # Apply neural style transfer with progress updates
                def progress_callback(step, total, loss):
                    gr.update(value=f"Step {step}/{total} - Loss: {loss:.2f}")

                # Apply style transfer
                start_time = time.time()
                stylized_img = apply_neural_style_transfer(
                    content_image=content_img,
                    style_image=style_img,
                    style_weight=float(style_w),
                    content_weight=float(content_w),
                    iterations=int(iters),
                )

                process_time = time.time() - start_time
                return (
                    stylized_img,
                    f"Neural style transfer completed in {process_time:.2f} seconds.",
                )

            except Exception as e:
                return None, f"Error applying neural style transfer: {str(e)}"

        neural_apply_button.click(
            fn=process_neural_style_transfer,
            inputs=[
                neural_content_image,
                neural_style_image,
                style_weight,
                content_weight,
                iterations,
            ],
            outputs=[neural_output_image, neural_progress],
        )

        gr.Markdown(
            """
        ### Instructions:
        1. Upload a content image (the image you want to stylize)
        2. Upload a style image (the image with the artistic style you want to apply)
        3. Adjust the style weight (higher values emphasize the style more)
        4. Adjust the content weight (higher values preserve more of the original content)
        5. Set the number of iterations (higher values give better results but take longer)
        6. Click "Apply Neural Style Transfer" and wait for the process to complete
        
        Note: Style transfer is computationally intensive and may take several minutes depending on your hardware and the number of iterations.
        """
        )


def video_stabilization(video_path, smoothing_radius=30, border_type='black', border_size=0, show_plot=False):
    """
    Stabilize a video using VidStab
    
    Args:
        video_path: Path to input video
        smoothing_radius: Radius of smoothing window
        border_type: Type of border handling ('black', 'reflect', 'replicate')
        border_size: Size of border (0 for no border, 'auto' for automatic)
        show_plot: Whether to generate and return trajectory plots
        
    Returns:
        stabilized_video_path: Path to stabilized video
        plot_path: Path to trajectory plot image (if show_plot=True)
        message: Status message
    """
    try:
        if video_path is None:
            return None, None, "Please upload a video first."
        
        # Create temporary output path
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # Initialize stabilizer
        stabilizer = VidStab()
        
        # Generate transforms
        stabilizer.gen_transforms(video_path)
        
        # Apply transforms
        stabilizer.apply_transforms(
            input_path=video_path,
            output_path=output_path,
            smoothing_radius=smoothing_radius,
            border_type=border_type,
            border_size=border_size
        )
        
        plot_path = None
        if show_plot:
            # Generate trajectory plot
            fig = stabilizer.plot_trajectory()
            plot_path = tempfile.mktemp(suffix='.png')
            fig.savefig(plot_path)
            plt.close(fig)
        
        return output_path, plot_path, "Video stabilization completed successfully."
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during video stabilization: {error_details}")
        return None, None, f"Error during video stabilization: {str(e)}"


def create_video_stabilization_tab():
    """Creates the video stabilization tab UI elements."""
    with gr.Tab("Video Stabilization"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Input Video")

                smoothing_radius = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=30,
                    step=5,
                    label="Smoothing Radius (frames)",
                )

                border_type = gr.Radio(
                    choices=["black", "reflect", "replicate"],
                    value="black",
                    label="Border Handling Method",
                )
                
                border_size = gr.Radio(
                    choices=["0", "auto"],
                    value="auto",
                    label="Border Size",
                )
                
                show_plot = gr.Checkbox(
                    label="Show Trajectory Plot",
                    value=True
                )

                stabilize_button = gr.Button("Stabilize Video")

            with gr.Column():
                video_output = gr.Video(label="Stabilized Output")
                trajectory_plot = gr.Image(label="Trajectory Plot", visible=True)
                status_text = gr.Textbox(label="Status")

        stabilize_button.click(
            fn=video_stabilization,
            inputs=[video_input, smoothing_radius, border_type, border_size, show_plot],
            outputs=[video_output, trajectory_plot, status_text],
        )

        gr.Markdown(
            """
        ## Video Stabilization
        
        This tool reduces camera shake and unwanted motion in videos to create smoother footage.
        
        ### Instructions:
        1. Upload a video file
        2. Adjust the smoothing radius (larger values create smoother motion but may crop more)
        3. Select a border handling method:
           - Black: Fills missing areas with black
           - Reflect: Mirrors the image to fill missing areas
           - Replicate: Extends the edge pixels to fill missing areas
        4. Choose a border size:
           - 0: No border (may show edge artifacts)
           - auto: Automatically calculate optimal border size
        5. Choose whether to display a trajectory plot
        6. Click "Stabilize Video" to process the video
        
        Note: Video stabilization works best on videos with unintentional camera movement like handheld footage.
        """
        )


def create_object_removal_tab():
    """Creates the object removal tab UI elements."""
    with gr.Tab("Object Removal"):
        with gr.Row():
            with gr.Column():
                # For older versions of Gradio, use simple Image components
                image_input = gr.Image(label="Input Image", type="numpy")
                mask_input = gr.Image(
                    label="Draw a mask over the object to remove", type="numpy"
                )

                removal_method = gr.Radio(
                    choices=["deepfill", "patchmatch", "generative"],
                    value="deepfill",
                    label="Inpainting Method",
                )

                remove_button = gr.Button("Remove Object")

                gr.Markdown(
                    """
                **Instructions:** 
                1. Upload an image
                2. Draw a mask in another drawing program and upload it as a black and white image 
                   (white areas will be removed)
                3. Choose an inpainting method
                4. Click "Remove Object"
                """
                )

            with gr.Column():
                image_output = gr.Image(label="Result")

        remove_button.click(
            fn=remove_object,
            inputs=[image_input, mask_input, removal_method],
            outputs=image_output,
        )

        gr.Markdown(
            """
        ## Object Removal
        
        This tool allows you to remove unwanted objects from images using intelligent inpainting technology.
        
        ### Instructions:
        1. Upload an image
        2. Upload a mask image where the white areas indicate what to remove
        3. Select an inpainting method:
           - DeepFill: Better for structured scenes
           - PatchMatch: Good for textured backgrounds
           - Generative: Best for complex scenes but slower
        4. Click "Remove Object" to process the image
        """
        )


def create_ui():
    """Creates the complete Gradio UI."""
    with gr.Blocks(title="VisualAlchemy") as app:
        gr.Markdown("# VisualAlchemy")
        gr.Markdown(
            "Transform your media with intelligent editing tools powered by deep learning."
        )

        with gr.Tabs():
            create_style_transfer_tab()
            create_video_stabilization_tab()
            create_object_removal_tab()

        gr.Markdown("### About")
        gr.Markdown(
            """
        This application provides three main functionalities:
        - **Style Transfer**: Apply artistic styles to your images using pre-trained neural networks
        - **Video Stabilization**: Reduce camera shake in videos
        - **Object Removal**: Remove unwanted objects from images
        
        Made with ❤️ using Python, PyTorch, OpenCV, and Gradio.
        """
        )

    return app


if __name__ == "__main__":
    # Launch the app
    app = create_ui()
    app.launch(share=True)
