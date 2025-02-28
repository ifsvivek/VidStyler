import gradio as gr
import sys
import numpy as np
import cv2
import time

# Add the project root to the path so we can import our modules
sys.path.append("/home/sic/Documentations/Vivek_FYP")
from src.video_stabilization.stabilize import stabilize_video
from src.object_removal.inpainting import remove_object
from src.style_transfer.neural_style import apply_neural_style_transfer


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

        # Function to apply neural style transfer
        def process_neural_style_transfer(
            content_img, style_img, style_w, content_w, iters
        ):
            import time

            if content_img is None or style_img is None:
                return None, "Please upload both content and style images."

            try:
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


def create_video_stabilization_tab():
    """Creates the video stabilization tab UI elements."""
    with gr.Tab("Video Stabilization"):
        gr.Markdown(
            """
        ## Video Stabilization
        
        This tool reduces camera shake and unwanted motion in videos to create smoother footage.
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video")

                with gr.Accordion("Basic Settings", open=True):
                    kp_method = gr.Dropdown(
                        choices=["GFTT", "BRISK", "ORB", "FAST", "SIFT", "SURF"],
                        value="GFTT",
                        label="Keypoint Detection Method",
                        info="Algorithm used to detect features across frames",
                    )

                    smoothing_radius = gr.Slider(
                        minimum=5,
                        maximum=100,
                        value=30,
                        step=1,
                        label="Smoothing Radius (frames)",
                        info="Higher values create smoother motion but may increase cropping",
                    )

                    border_type = gr.Radio(
                        choices=["black", "reflect", "replicate"],
                        value="black",
                        label="Border Handling Method",
                        info="Method to fill in missing edges after stabilization",
                    )

                    border_size = gr.Radio(
                        choices=["auto", "0", "50", "100"],
                        value="auto",
                        label="Border Size",
                        info="Size of border in pixels, or 'auto' for automatic sizing",
                    )

                with gr.Accordion("Advanced Effects", open=False):
                    use_layer_effect = gr.Checkbox(
                        label="Use Layer Effects",
                        value=False,
                        info="Apply special layering effects to stabilized video",
                    )

                    layer_effect_type = gr.Radio(
                        choices=["overlay", "blend"],
                        value="overlay",
                        label="Layer Effect Type",
                        info="Overlay creates trails, blend creates motion blur",
                        visible=False,
                    )

                    layer_alpha = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="Blend Alpha",
                        info="Opacity of blended frames (for blend effect only)",
                        visible=False,
                    )

                with gr.Accordion("Visualization", open=True):
                    show_plots = gr.Checkbox(
                        label="Show Trajectory Plots",
                        value=True,
                        info="Generate plots showing the camera motion before and after stabilization",
                    )

                stabilize_button = gr.Button("Stabilize Video", variant="primary")

            with gr.Column(scale=1):
                video_output = gr.Video(label="Stabilized Output")
                with gr.Accordion("Analysis", open=True):
                    with gr.Row():
                        with gr.Column(scale=1):
                            trajectory_plot = gr.Image(
                                label="Trajectory Plot", show_label=True
                            )
                        with gr.Column(scale=1):
                            transforms_plot = gr.Image(
                                label="Transforms Plot", show_label=True
                            )
                status_text = gr.Textbox(label="Status")

        # Set up visibility conditions for layer effect options
        use_layer_effect.change(
            fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
            inputs=[use_layer_effect],
            outputs=[layer_effect_type, layer_alpha],
        )

        stabilize_button.click(
            fn=stabilize_video,
            inputs=[
                video_input,
                kp_method,
                smoothing_radius,
                border_type,
                border_size,
                use_layer_effect,
                layer_effect_type,
                layer_alpha,
                show_plots,
            ],
            outputs=[video_output, trajectory_plot, transforms_plot, status_text],
        )

        gr.Markdown(
            """
        ### Instructions:
        1. Upload a video file
        2. Configure stabilization settings:
           - **Keypoint Detection Method**: Algorithm used to track features across frames
           - **Smoothing Radius**: Higher values create smoother motion but may increase cropping
           - **Border Handling**: Choose how to fill in missing edges after stabilization
           - **Border Size**: Choose border width or let it be determined automatically
        3. Explore advanced effects (optional):
           - **Layer Effects**: Create artistic trail or motion blur effects
        4. Click "Stabilize Video" to process the video
        
        ### Tips for Best Results:
        - For shaky handheld footage, use a smoothing radius of 30-50
        - When objects move close to the frame edge, use 'auto' border size
        - For artistic motion effects, enable layer effects and try both types
        - GFTT and ORB keypoint methods work well for most videos
        """
        )


def create_object_removal_tab():
    """Creates the object removal tab UI elements."""
    with gr.Tab("Object Removal"):
        gr.Markdown(
            """
        ## Object Removal
        
        This tool allows you to remove unwanted objects from images using intelligent inpainting technology.
        Draw over the objects you want to remove and select an inpainting method.
        """
        )

        with gr.Row():
            with gr.Column():
                # Use ImageEditor instead of Image
                input_image = gr.ImageEditor(
                    label="Draw over objects to remove",
                    type="numpy",
                    height=500,
                )

                with gr.Row():
                    removal_method = gr.Radio(
                        choices=["deepfill", "patchmatch", "generative"],
                        value="patchmatch",
                        label="Inpainting Method",
                    )
                    remove_button = gr.Button("Remove Object", variant="primary")

            with gr.Column():
                image_output = gr.Image(label="Result", height=500)
                status = gr.Textbox(label="Status")

        # Process the image removal
        def process_removal(image_data, method):
            if image_data is None:
                return None, "Please upload an image and draw a mask"

            try:
                # Extract the layers from ImageEditor output
                if isinstance(image_data, dict) and "composite" in image_data and "background" in image_data:
                    # Get the original image (background)
                    image = image_data["background"]
                    
                    # Get the image with drawing (composite)
                    composite = image_data["composite"]
                    
                    # Get image dimensions
                    h, w = image.shape[:2]
                    
                    # Create a mask by finding the difference between composite and background
                    diff = cv2.absdiff(composite, image)
                    
                    # Convert to grayscale
                    if len(diff.shape) == 3:
                        mask_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
                    else:
                        mask_gray = diff
                    
                    # Apply threshold to create binary mask
                    _, mask = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
                    
                    # Make sure mask is uint8
                    mask = mask.astype(np.uint8)
                    
                    # Debug info
                    print(f"Image shape: {image.shape}, type: {image.dtype}")
                    print(f"Mask shape: {mask.shape}, type: {mask.dtype}")
                    print(f"Mask values - min: {mask.min()}, max: {mask.max()}")
                    
                    # Apply object removal
                    # Make a copy of the image to ensure we're not modifying the original
                    image_copy = image.copy()
                    
                    # Apply inpainting
                    result = remove_object(image_copy, mask, method)
                    
                    return result, f"Object removed using {method} method"
                else:
                    return None, "Please draw on the image to mark areas for removal"

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"Error processing image: {str(e)}"

        # Connect the remove button
        remove_button.click(
            fn=process_removal,
            inputs=[input_image, removal_method],
            outputs=[image_output, status],
        )

        gr.Markdown(
            """
        ### Instructions:
        1. Upload an image to the editor
        2. Use the drawing tools to mark areas you want to remove
        3. Select an inpainting method
        4. Click "Remove Object" to process the image
        
        ### Tips:
        - For small objects, the PatchMatch method often gives clean results
        - For complex scenes, try the generative method
        - Draw carefully to cover the entire object you want to remove
        - Use a thicker brush for better coverage
        - Make sure to completely cover the object you want to remove
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
        - **Video Stabilization**: Reduce camera shake in videos using advanced tracking algorithms
        - **Object Removal**: Remove unwanted objects from images
        
        Made with ❤️ using Python, PyTorch, OpenCV, and Gradio.
        """
        )

    return app


if __name__ == "__main__":
    # Launch the app
    app = create_ui()
    app.launch()
