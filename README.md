# VidStyler: AI-powered Image and Video Editing Suite

VidStyler is a comprehensive tool that implements intelligent tools to assist users in image and video editing tasks using deep learning and computer vision techniques. The suite offers four main functionalities:

-   **Neural Style Transfer**: Apply artistic styles to images
-   **Video Stabilization**: Reduce shakiness in videos
-   **Object Removal**: Automatically remove objects from images and videos
-   **Inpainting**: Modify specific parts of an image based on text prompts

## Features

### Neural Style Transfer

-   VGG16-based feature extraction for style and content representation
-   Optimization-based style transfer using LBFGS optimizer
-   Gram matrix calculation for style feature representation
-   Adjustable style-content balance via weighting parameters
-   Progress tracking during optimization with callback support

### Video Stabilization

-   Multiple keypoint detection methods (GFTT, SIFT, SURF, ORB, BRISK, FAST)
-   Customizable smoothing radius and border handling
-   Layer effects for artistic motion trails
-   Trajectory and transform visualization
-   Frame-by-frame processing capability

### Object Removal

-   Interactive object removal by drawing over unwanted areas
-   Multiple inpainting methods (OpenCV-based and DeepFill)
-   Support for images with alpha channels
-   Real-time feedback in web interface
-   Memory-efficient processing for large images

### Inpainting

-   Text-guided image inpainting
-   Leverages ChatGLM text encoder for natural language understanding
-   Based on Stable Diffusion XL architecture
-   High-resolution output support (up to 1024x768)
-   Interactive mask creation to specify areas for modification
-   Control over guidance scale and inference steps

## Installation

```bash
git clone https://github.com/ifsvivek/VidStyler.git
cd VidStyler
pip install -r requirements.txt
```

## Usage

Start the Gradio application:

```bash
python run.py
```

Navigate to the provided local URL (typically http://127.0.0.1:7860) in your web browser.

### Style Transfer Usage

1. Upload a content image (the image you want to stylize)
2. Upload a style image (the image with the artistic style you want to apply)
3. Adjust style and content weights to control the balance between preserving content and applying style
4. Set the number of iterations (higher values give better results but take longer)
5. Click "Apply Neural Style Transfer" and wait for the process to complete

### Video Stabilization Usage

1. Upload a video file
2. Configure stabilization settings:
    - Select a keypoint detection method
    - Adjust smoothing radius (30-50 works well for most videos)
    - Choose border handling method and size
3. Optionally enable layer effects for artistic motion trails
4. Click "Stabilize Video" to process the video
5. View trajectory and transforms plots for analysis

### Object Removal Usage

1. Upload an image to the editor
2. Draw over the objects you want to remove with the brush tool
3. Select an inpainting method:
    - Auto: Chooses the best method based on your image
    - DeepFill: Uses machine learning for more natural results
    - OpenCV: Faster but may be less accurate for complex scenes
4. Click "Remove Object" to process the image
5. Use "Reset Image" to clear your drawings and start over

### Inpainting Usage

1. Upload an image to the editor
2. Create a mask by drawing over areas you want to modify
3. Enter a text prompt describing the desired changes (e.g., "change the shirt color to red")
4. Adjust parameters like guidance scale and inference steps if needed
5. Click "Apply Inpainting" to process the image
6. Download the modified result

## How It Works

### Neural Style Transfer Implementation

The style transfer module uses an optimization-based approach:

1. Extract content features from the content image using a pre-trained VGG16 CNN
2. Extract style features (Gram matrices) from the style image
3. Initialize the output image with the content image
4. Iteratively update the output image using LBFGS optimizer to minimize:
    - Content loss: MSE between content features from layer relu3_3
    - Style loss: MSE between Gram matrices from multiple layers
5. Apply denormalization to produce the final stylized image

### Video Stabilization Implementation

The video stabilization process follows these steps:

1. Track features across video frames using the selected keypoint detection method
2. Compute transformations between consecutive frames
3. Smooth the camera trajectory using a moving average over the specified window
4. Apply the smoothed transformations to create a stabilized video
5. Handle borders according to the selected method and size
6. Optionally apply layer effects for artistic results

### Object Removal Implementation

The object removal pipeline consists of:

1. User draws over unwanted objects to create a mask
2. The system processes the mask to ensure proper coverage
3. Depending on the selected method:
    - OpenCV inpainting uses traditional computer vision techniques
    - DeepFill uses a neural network to generate realistic textures
4. The result is presented with the unwanted objects seamlessly removed

### Inpainting Implementation

The inpainting module uses a text-guided approach:

1. Process the input image and mask to identify areas for modification
2. Encode the text prompt using the ChatGLM text encoder
3. Apply the Stable Diffusion XL-based inpainting model to generate the modified content
4. Blend the inpainted region with the original image seamlessly
5. Return the final result with the specified modifications applied

## Project Structure

```
.
├── app/
│   └── app.py           # Gradio interface implementation
├── src/
│   ├── style_transfer/  # Neural style transfer implementation
│   ├── video_stabilization/  # Video stabilization algorithms
│   └── object_removal/  # Object detection and removal system
│   └── inpainting/      # Text-guided inpainting system
├── img/                 # Sample images and UI screenshots
├── run.py               # Application entry point
└── README.md            # Project documentation
```

## Requirements

-   Python 3.8+
-   PyTorch 1.8+
-   OpenCV 4.5+
-   Gradio 2.0+
-   diffusers library
-   vidstab
-   NumPy
-   Matplotlib
-   TensorFlow (for DeepFill inpainting)
-   PIL (Pillow)
-   CUDA-capable GPU (recommended for faster processing)

## Limitations

-   Style transfer is computationally intensive and may take several minutes
-   Very fast camera movements may result in excessive cropping during stabilization
-   Extremely complex object removal may require manual touchup
-   Processing time increases with image/video resolution
-   Inpainting results depend on the clarity and specificity of text prompts
