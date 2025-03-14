import gradio as gr
import sys
import numpy as np
import cv2
import time
import os

# Add the project root to the path so we can import our modules
sys.path.append("/home/sic/Projects/Vivek_FYP")
from src.video_stabilization.stabilize import stabilize_video
from src.object_removal.inpainting import remove_object
from src.style_transfer.neural_style import apply_neural_style_transfer
from Kolors.sample_inpainting import infer as kolors_infer


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
        Draw over the objects you want to remove with the brush tool.
        """
        )

        with gr.Row():
            with gr.Column():
                # Use ImageEditor for drawing on the image
                input_image = gr.ImageEditor(
                    label="Draw over objects to remove",
                    type="numpy",
                    height=500,
                )

                with gr.Row():
                    inpaint_method = gr.Radio(
                        choices=["Auto (Default)", "DeepFill (ML-based)", "OpenCV"],
                        value="Auto (Default)",
                        label="Inpainting Method",
                        info="Choose the algorithm to use for removing objects",
                    )
                    remove_button = gr.Button("Remove Object", variant="primary")
                    reset_button = gr.Button("Reset Image")

            with gr.Column():
                image_output = gr.Image(label="Result", height=500)
                status = gr.Textbox(label="Status")

        # Process the image removal
        def process_removal(image_data, method):
            if image_data is None:
                return None, "Please upload an image and draw a mask"

            try:
                # Extract the layers from ImageEditor output
                if (
                    isinstance(image_data, dict)
                    and "composite" in image_data
                    and "background" in image_data
                ):
                    # Get the original image (background)
                    image = image_data["background"]

                    # Get the image with drawing (composite)
                    composite = image_data["composite"]

                    # Debug info for input images
                    print(f"Original image shape: {image.shape}, dtype: {image.dtype}")
                    print(
                        f"Composite image shape: {composite.shape}, dtype: {composite.dtype}"
                    )

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

                    # Dilate mask slightly to ensure coverage
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)

                    # Debug info for mask
                    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
                    print(f"Mask values - min: {mask.min()}, max: {mask.max()}")

                    # Map the UI selection to method parameter
                    inpaint_method = "deepfill"  # Default
                    if method == "OpenCV":
                        inpaint_method = "generative"
                    elif method == "DeepFill (ML-based)":
                        inpaint_method = "deepfill"

                    # Apply inpainting using the mask
                    start_time = time.time()
                    result = remove_object(image.copy(), mask, method=inpaint_method)
                    process_time = time.time() - start_time

                    return (
                        result,
                        f"Object removed in {process_time:.2f} seconds using {method}",
                    )
                else:
                    return None, "Please draw on the image to mark areas for removal"

            except Exception as e:
                import traceback

                traceback.print_exc()
                return None, f"Error processing image: {str(e)}"

        # Reset the image (clear drawings)
        def reset_image(image_data):
            if (
                image_data is None
                or not isinstance(image_data, dict)
                or "background" not in image_data
            ):
                return None, "No image to reset"
            return {"image": image_data["background"], "mask": None}, "Image reset"

        # Connect the buttons
        remove_button.click(
            fn=process_removal,
            inputs=[input_image, inpaint_method],
            outputs=[image_output, status],
        )

        reset_button.click(
            fn=reset_image,
            inputs=[input_image],
            outputs=[input_image, status],
        )

        gr.Markdown(
            """
        ### Instructions:
        1. Upload an image to the editor
        2. Draw over the objects you want to remove
        3. Select an inpainting method:
           - **Auto**: Chooses the best method based on your image
           - **DeepFill**: Uses machine learning for more natural results
           - **OpenCV**: Faster but may be less accurate for complex scenes
        4. Click "Remove Object" to process the image
        5. Use "Reset Image" to clear your drawings and start over
        
        ### Tips:
        - Draw carefully to cover the entire object you want to remove
        - For complex objects or textures, try the DeepFill method
        - For simple objects against uniform backgrounds, OpenCV may work well
        - Make sure to completely cover the object you want to remove
        """
        )


def create_kolors_inpainting_tab():
    """Creates the text-guided inpainting tab UI elements."""
    with gr.Tab("Text-Guided Inpainting"):
        gr.Markdown(
            """
        ## Text-Guided Inpainting
        
        This advanced tool allows you to modify specific parts of an image using natural language prompts.
        Upload an image and a corresponding mask, then describe what changes you want to make.
        """
        )

        with gr.Row():
            with gr.Column():
                # Image and mask upload
                input_image = gr.Image(label="Original Image", type="pil")
                mask_image = gr.Image(
                    label="Mask Image (white areas will be modified)", type="pil"
                )

                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="Modification Prompt",
                        placeholder="Example: change the shirt color to red",
                        value="change the shirt color to red",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt (optional)",
                        placeholder="Describe what you don't want to see",
                    )

                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=6.0,
                            step=0.5,
                            label="Guidance Scale",
                            info="How closely to follow the prompt (higher = more literal)",
                        )
                        inference_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=25,
                            step=5,
                            label="Inference Steps",
                            info="More steps = better quality but slower",
                        )

                    with gr.Row():
                        strength = gr.Slider(
                            minimum=0.5,
                            maximum=0.999,
                            value=0.999,
                            step=0.01,
                            label="Effect Strength",
                            info="How strongly to apply the effect",
                        )
                        seed = gr.Slider(
                            minimum=0,
                            maximum=2147483647,
                            value=603,
                            step=1,
                            label="Random Seed",
                            info="For reproducible results",
                        )

                inpaint_button = gr.Button("Generate Inpainting", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Result")
                status = gr.Textbox(label="Status")

        # Function to handle inpainting
        def process_kolors_inpainting(
            input_img,
            mask_img,
            prompt,
            neg_prompt,
            guidance,
            steps,
            strength_val,
            seed_val,
        ):
            if input_img is None or mask_img is None:
                return None, "Please upload both an image and a mask"

            try:
                # Save temporary files for the inpainting script to use
                temp_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp"
                )
                os.makedirs(temp_dir, exist_ok=True)

                input_path = os.path.join(temp_dir, "temp_input.png")
                mask_path = os.path.join(temp_dir, "temp_mask.png")

                input_img.save(input_path)
                mask_img.save(mask_path)

                # Prepare arguments for the inpainting function
                import torch

                args = {
                    "image_path": input_path,
                    "mask_path": mask_path,
                    "prompt": prompt,
                    "negative_prompt": neg_prompt,
                    "guidance_scale": guidance,
                    "num_inference_steps": int(steps),
                    "strength": strength_val,
                    "seed": int(seed_val),
                    "return_image": True,  # Special flag to return image instead of saving
                }

                # Call the modified infer function
                start_time = time.time()

                # Import the function here to avoid loading models until needed
                from Kolors.sample_inpainting import infer

                result_image = infer(**args)

                process_time = time.time() - start_time
                return (
                    result_image,
                    f"Inpainting completed in {process_time:.2f} seconds.",
                )

            except Exception as e:
                import traceback

                traceback.print_exc()
                return None, f"Error processing inpainting: {str(e)}"

        inpaint_button.click(
            fn=process_kolors_inpainting,
            inputs=[
                input_image,
                mask_image,
                prompt_input,
                negative_prompt,
                guidance_scale,
                inference_steps,
                strength,
                seed,
            ],
            outputs=[output_image, status],
        )

        gr.Markdown(
            """
        ### Instructions:
        1. Upload your image
        2. Upload or create a mask image (white areas will be modified)
        3. Enter a text prompt describing what changes you want to make
        4. Optional: adjust advanced settings
        5. Click "Generate Inpainting" to process the image
        
        ### Tips:
        - Use clear, descriptive prompts like "change the shirt color to red" or "replace the background with a beach scene"
        - The white areas in the mask indicate where changes will be applied
        - For best results, ensure the mask covers the entire object you want to modify
        - Adjust guidance scale to control how closely the result follows your text prompt
        - Try different random seeds if you're not satisfied with the initial result
        """
        )


def create_ui():
    """Creates the complete Gradio UI."""
    with gr.Blocks(title="VidStyler") as app:
        gr.Markdown("# VidStyler")
        gr.Markdown(
            "Transform your media with intelligent editing tools powered by deep learning."
        )

        with gr.Tabs():
            create_style_transfer_tab()
            create_video_stabilization_tab()
            create_object_removal_tab()
            create_kolors_inpainting_tab()  # Add the new tab

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
