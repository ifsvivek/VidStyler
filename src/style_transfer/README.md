# Neural Style Transfer Module

This module implements neural style transfer techniques to apply artistic styles to images. The implementation follows the optimization-based approach described in the paper by Gatys et al.

## Features

-   VGG16-based feature extraction for style and content representation
-   Optimization-based style transfer using LBFGS optimizer
-   Gram matrix calculation for style feature representation
-   Adjustable style-content balance via weighting parameters
-   Progress tracking during optimization with callback support
-   Support for different image sizes with automatic resizing
-   Device-agnostic implementation (runs on CPU or CUDA)

## Implementation Details

The style transfer uses an optimization-based approach:

1. Extract content features from the content image using a pre-trained VGG16 CNN
2. Extract style features (Gram matrices) from the style image
3. Initialize the output image with the content image
4. Iteratively update the output image using LBFGS optimizer to minimize:
    - Content loss: MSE between content features from layer relu3_3
    - Style loss: MSE between Gram matrices from layers relu1_2, relu2_2, relu3_3, and relu4_3
5. Apply denormalization to produce the final stylized image

## UI

![image](../../img/VisualAlchemy1-1.png)

## Key Components

-   `VGG16`: A pre-trained VGG16 model modified to extract intermediate features
-   `gram_matrix`: Calculates Gram matrices from feature maps for style representation
-   `neural_style_transfer`: Core function that performs the optimization process
-   `apply_neural_style_transfer`: User-friendly wrapper function that handles image format conversion
-   `TransformerNet`: Implementation of a feed-forward network for fast style transfer

## Usage

```python
from src.style_transfer import apply_neural_style_transfer

# For high-quality style transfer
stylized_image = apply_neural_style_transfer(
    content_image=content_img,  # numpy array (RGB or BGR)
    style_image=style_img,      # numpy array (RGB or BGR)
    style_weight=1e6,           # weight for style loss (default: 1e6)
    content_weight=1,           # weight for content loss (default: 1)
    iterations=300,             # number of optimization steps (default: 300)
    progress_callback=None      # optional callback function to track progress
)
```

## Technical Approach

### Content Representation

Content features are extracted from the relu3_3 layer of the VGG16 network, which provides a good balance between high-level structural information and lower-level details.

### Style Representation

Style is represented using Gram matrices calculated from feature maps at multiple layers (relu1_2, relu2_2, relu3_3, and relu4_3). The Gram matrix captures texture information by computing correlations between different feature channels.

### Optimization Process

The style transfer uses LBFGS optimization to minimize a weighted combination of content and style losses. The algorithm typically converges in 300-500 iterations, with each iteration refining the output image to better match both content and style objectives.

### Additional Models

The module includes a `TransformerNet` implementation for feed-forward style transfer, which offers significantly faster stylization once trained. The network architecture includes:

-   Initial convolutional layers to extract features
-   Five residual blocks to maintain structural integrity
-   Upsampling convolutional layers to restore image dimensions
-   Instance normalization for style-agnostic feature normalization

### Utilities

The module provides several utility functions:

-   Image loading and preprocessing with optional resizing
-   Tensor denormalization for visualization
-   Gram matrix calculation for style representation
-   Seed control for reproducibility
