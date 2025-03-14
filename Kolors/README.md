# Kolors Inpainting

A sample script for image inpainting using the Kolors model. This tool allows you to modify specific parts of an image based on text prompts.

## Features

-   Text-guided image inpainting
-   Leverages ChatGLM text encoder for natural language understanding
-   Based on Stable Diffusion XL architecture
-   High-resolution output support (up to 1024x768)

## Requirements

-   Python 3.8+
-   PyTorch 1.12+
-   diffusers library
-   PIL (Pillow)
-   Required model weights in `weights/Kolors-Inpainting/` directory

## Usage

1. Place the source image as `test.png` in the working directory
2. Create a mask image as `tm.png` (white areas indicate where changes should be applied)
3. Run the script:

```bash
python sample_inpainting.py
```

The output will be saved to the `scripts/outputs/` directory.

## Configuration

You can modify the following parameters in the script:

-   `image_path`: Path to the input image (default: "test.png")
-   `mask_path`: Path to the mask image (default: "tm.png")
-   `prompt`: Text instruction for inpainting (default: "change the shirt color to red")
-   `height` and `width`: Output resolution (default: 1024×768)
-   `guidance_scale`: Controls how closely the image follows the prompt (default: 6.0)
-   `num_inference_steps`: Number of denoising steps (default: 25)
-   `strength`: Strength of the inpainting effect (default: 0.999)

## Advanced Usage

For more customized inpainting, you can modify the script to:

1. Accept command line arguments for image paths and prompts
2. Process batches of images
3. Experiment with different guidance scales and inference steps

## Troubleshooting

-   **CUDA out of memory error**: Reduce image dimensions or batch size
-   **Tokenizer errors**: The script includes a patched tokenizer to handle overflow issues
-   **Missing models**: Ensure all required model files are in the weights directory

## Example Results

Input image + mask + prompt "change the shirt color to red" = Edited output with a red shirt
