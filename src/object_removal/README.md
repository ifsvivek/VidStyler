# Object Removal Module

This module implements intelligent object removal from images using inpainting techniques. It provides functionality to remove unwanted objects from images by drawing over them, and then using advanced inpainting algorithms to fill in the removed areas naturally.

## Features

-   Interactive object removal by drawing over unwanted areas
-   Multiple inpainting methods for different use cases
-   Support for images with alpha channels
-   Integration with Gradio UI for easy use

## Implementation Details

The object removal pipeline consists of these main components:

### 1. User Interaction

-   Draw over objects to remove using the brush tool
-   Selection of inpainting method based on image content
-   Real-time feedback in web interface

### 2. Mask Generation

-   Automatic mask generation from user drawings
-   Binary mask creation with thresholding and dilation
-   Preprocessing to ensure compatibility with inpainting algorithms

### 3. Image Inpainting Methods

#### OpenCV Inpainting

-   Fast inpainting using traditional computer vision techniques
-   Good for simple backgrounds and small objects
-   Two methods: Navier-Stokes (NS) and Telea algorithms

#### DeepFill Inpainting

-   Machine learning-based inpainting using TensorFlow
-   Uses pre-trained model for high-quality results
-   Better for complex textures and larger removed areas
-   Adapts to image content for more natural filling

## Usage

### Command Line Interface

```python
import cv2
from src.object_removal.inpainting import remove_object

# Load image and create mask
image = cv2.imread('image.jpg')
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

# Remove object using default method
result = remove_object(image, mask)

# Or specify a method
result_deepfill = remove_object(image, mask, method='deepfill')
result_opencv = remove_object(image, mask, method='generative')

# Save result
cv2.imwrite('result.jpg', result)
```

### Using DeepFillInpainter Directly

```python
from src.object_removal.deepfill_inpainter import DeepFillInpainter

# Initialize the inpainter
inpainter = DeepFillInpainter(model_path='./model/pretrained_model')

# Inpaint an image
result = inpainter.inpaint(image, mask)
```

### Web UI Usage

The module integrates with the Gradio-based web interface:

1. Upload an image to the editor
2. Draw over objects you want to remove
3. Select the inpainting method:
    - Auto (Default): Chooses the best method based on your image
    - DeepFill: Uses machine learning for more natural results
    - OpenCV: Faster but may be less accurate for complex scenes
4. Click "Remove Object" to process the image
5. Use "Reset Image" to clear your drawings and start over


## UI

![UI](../../img/VisualAlchemy3-1.png)

## Technical Details

### DeepFill Model

The DeepFill model uses a neural network architecture with:

-   Encoder-decoder structure
-   Skip connections for preserving detail
-   Supports various input image sizes
-   Pretrained on large datasets of natural images

### Performance Considerations

-   OpenCV methods are faster but may produce less natural results
-   DeepFill provides higher quality but requires more computational resources
-   Images with alpha channels are properly handled by both methods
-   For real-time applications, OpenCV methods are recommended
-   For best quality where speed is not critical, DeepFill is recommended

## Dependencies

-   OpenCV
-   NumPy
-   TensorFlow (for DeepFill method)
