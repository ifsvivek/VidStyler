# AI-powered Image and Video Editing Suite

This project implements intelligent tools that assist users in image and video editing tasks using deep learning and computer vision techniques. The suite offers three main functionalities:

-   **Neural Style Transfer**: Apply artistic styles to images
-   **Video Stabilization**: Reduce shakiness in videos
-   **Object Removal**: Automatically remove objects from images and videos

## Features

-   Intuitive Gradio-based user interface
-   High-quality neural style transfer with multiple style options
-   Advanced video stabilization algorithms
-   Smart object removal with automatic inpainting
-   Support for various image and video formats

## Installation

```bash
git clone https://github.com/yourusername/ai-image-video-editing.git
cd ai-image-video-editing
pip install -r requirements.txt
```

## Usage

Start the Gradio application:

```bash
python app/main.py
```

Navigate to the provided local URL (typically http://127.0.0.1:7860) in your web browser.

## Project Structure

The project is organized into three main components:

-   `src/style_transfer/`: Neural style transfer implementation
-   `src/video_stabilization/`: Video stabilization algorithms
-   `src/object_removal/`: Object detection and removal system
-   `app/`: Gradio interface and integration code

## Requirements

-   Python 3.8+
-   PyTorch 1.8+
-   OpenCV 4.5+
-   Gradio 2.0+
-   CUDA-capable GPU (recommended for faster processing)

See `requirements.txt` for a complete list of dependencies.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

-   Research papers and implementations that inspired this project are listed in INFO.md
