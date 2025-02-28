# Neural Style Transfer Module

This module implements neural style transfer techniques to apply artistic styles to images. The implementation includes both optimization-based methods for high-quality results and feed-forward networks for faster processing.

## Features

-   VGG-based feature extraction for style and content representation
-   Optimization-based style transfer for highest quality results
-   Fast feed-forward style transfer for near real-time processing
-   Multiple pre-trained style models
-   Support for custom style images
-   Adjustable style-content balance

## Implementation Details

### Optimization-Based Approach

The optimization-based approach uses iterative optimization to find an image that matches both content and style features:

1. Extract content features from the content image using a pre-trained CNN
2. Extract style features (Gram matrices) from the style image
3. Initialize the output image (either with random noise or the content image)
4. Iteratively update the output image to minimize content and style losses
5. Apply total variation regularization for smoothness

### Feed-Forward Approach

The feed-forward approach uses a trained transformation network to generate stylized images in a single pass:

1. A transformation network is pre-trained to convert content images to a specific style
2. During inference, the content image is passed through the network to produce a stylized result immediately
3. The network is trained using perceptual losses computed from a pre-trained VGG network

## Usage

```python
from style_transfer import OptimizationStyleTransfer, FeedForwardStyleTransfer

# For high-quality results (slower)
transfer = OptimizationStyleTransfer()
stylized_image = transfer.transfer(content_image="path/to/content.jpg",
                                 style_image="path/to/style.jpg",
                                 num_iterations=1000,
                                 content_weight=1.0,
                                 style_weight=1e6)

# For faster results
fast_transfer = FeedForwardStyleTransfer(model_path="models/starry_night.pth")
stylized_image = fast_transfer.transfer(content_image="path/to/content.jpg")
```

## References

-   Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "Image style transfer using convolutional neural networks."
-   Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution."
