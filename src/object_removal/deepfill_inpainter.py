import os
import numpy as np
import cv2
import time
import tensorflow as tf

# Set TF compatibility mode
if tf.__version__.startswith("2."):
    # TF 2.x compatibility
    tf.compat.v1.disable_eager_execution()
    Session = tf.compat.v1.Session
    placeholder = tf.compat.v1.placeholder
    ConfigProto = tf.compat.v1.ConfigProto
    global_variables_initializer = tf.compat.v1.global_variables_initializer
    Saver = tf.compat.v1.train.Saver
else:
    # TF 1.x
    Session = tf.Session
    placeholder = tf.placeholder
    ConfigProto = tf.ConfigProto
    global_variables_initializer = tf.global_variables_initializer
    Saver = tf.train.Saver

# Import the model (assuming it's in the same directory)
try:
    from src.object_removal.model.model import Model
except ImportError:
    # Alternative import path
    try:
        from model.model import Model
    except ImportError:
        print("Error: Could not import Model. Please check model path.")


class DeepFillInpainter:
    """
    Class for inpainting images using the DeepFill model.
    
    This class loads a pretrained DeepFill model and provides functionality to
    remove objects from images based on a mask.
    """
    
    def __init__(self, model_path=None, size=512):
        """
        Initialize the DeepFillInpainter.
        
        Args:
            model_path: Path to the pretrained model. If None, uses default path.
            size: Size of input images the model expects (will resize if different)
        """
        self.size = size
        self.model_loaded = False
        self.sess = None
        
        # Default model path
        if model_path is None:
            # Try various common paths
            possible_paths = [
                './model/pretrained_model',
                './src/object_removal/model/pretrained_model',
                '/home/sic/Documentations/Vivek_FYP/src/object_removal/model/pretrained_model',
            ]
            
            for path in possible_paths:
                if os.path.exists(path) or os.path.exists(path + '.index') or os.path.exists(path + '.meta'):
                    model_path = path
                    break
        
        self.model_path = model_path
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the TensorFlow model"""
        try:
            print(f"Attempting to load DeepFill model from: {self.model_path}")
            
            # Create TensorFlow session
            config = ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = Session(config=config)
            
            # Set up placeholders
            self.isTraining = placeholder(tf.bool)
            self.images_placeholder = placeholder(tf.float32, shape=[1, self.size, self.size, 3], name="images")
            
            # Build the model
            model = Model()
            self.reconstruction_output = model.build_reconstruction(self.images_placeholder, self.isTraining)
            
            # Load weights
            saver = Saver(max_to_keep=100)
            saver.restore(self.sess, self.model_path)
            
            print("DeepFill model loaded successfully")
            self.model_loaded = True
            
        except Exception as e:
            print(f"Failed to load DeepFill model: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def is_model_loaded(self):
        """Check if the model was loaded successfully"""
        return self.model_loaded
    
    def generate_mask(self, image, strokes):
        """
        Generate a mask from the stroke image.
        
        Args:
            image: Original image (numpy array)
            strokes: Image with black strokes on it (numpy array)
            
        Returns:
            Binary mask where the strokes are
        """
        # Convert to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        if not isinstance(strokes, np.ndarray):
            strokes = np.array(strokes)
        
        # Find difference between original image and stroke image
        diff = cv2.absdiff(image, strokes)
        
        # Convert to grayscale if it's a color image
        if len(diff.shape) == 3:
            mask_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = diff
            
        # Apply threshold to create binary mask
        _, mask = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
        
        # Ensure mask is uint8
        mask = mask.astype(np.uint8)
        
        # Dilate mask slightly to ensure coverage
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def masking_image(self, image, mask):
        """
        Apply the mask to the image for the network input.
        
        Args:
            image: Input image as numpy array (values 0-255)
            mask: Binary mask where white (255) indicates pixels to be removed
            
        Returns:
            Masked image ready for input to the network
        """
        # Normalize image to 0-1 range
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        # Ensure mask is binary and in the right format
        if mask.dtype == np.uint8:
            mask = mask.astype(np.float32) / 255.0
        
        # Convert single channel mask to 3 channels if needed
        if len(mask.shape) == 2:
            mask = np.stack([mask, mask, mask], axis=2)
            
        # Apply mask to image
        masked_image = image * (1 - mask)
        
        return masked_image
    
    def inpaint(self, image, mask=None, strokes=None):
        """
        Remove objects from an image using the trained model.
        
        Args:
            image: Input image as numpy array
            mask: Binary mask where white (255) indicates pixels to inpaint.
                  If None, strokes must be provided.
            strokes: Image with black strokes on areas to be inpainted.
                     Only used if mask is None.
        
        Returns:
            Image with objects removed
        """
        if not self.model_loaded:
            print("Error: Model not loaded. Cannot perform inpainting.")
            return image
        
        if image is None:
            print("Error: Input image is None")
            return None
        
        # Generate mask from strokes if mask not provided
        if mask is None and strokes is not None:
            mask = self.generate_mask(image, strokes)
        elif mask is None:
            print("Error: Either mask or strokes must be provided")
            return image
        
        # Preprocess image and mask
        start_time = time.time()
        
        # Convert to float if uint8
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.copy()
            
        # Make sure image has 3 channels (RGB)
        if len(image_float.shape) == 2:
            image_float = np.stack([image_float, image_float, image_float], axis=2)
        
        # Save original shape for resizing back later
        original_shape = image_float.shape
        
        # Resize to model input size if needed
        if original_shape[0] != self.size or original_shape[1] != self.size:
            image_resized = cv2.resize(image_float, (self.size, self.size))
            mask_resized = cv2.resize(mask, (self.size, self.size))
        else:
            image_resized = image_float
            mask_resized = mask
            
        # Apply mask to image
        input_image = self.masking_image(image_resized, mask_resized)
        
        # Ensure input is in BGR format for the network
        input_image_bgr = input_image[..., ::-1]  # RGB to BGR
        
        # Prepare input tensor
        input_tensor = np.expand_dims(input_image_bgr, 0)  # Add batch dimension
        
        # Run the model
        output_tensor = self.sess.run(
            self.reconstruction_output,
            feed_dict={
                self.images_placeholder: input_tensor,
                self.isTraining: False
            }
        )
        
        # Get result
        result = np.squeeze(output_tensor)  # Remove batch dimension
        result = result[..., ::-1]  # BGR to RGB
        
        # Resize back to original shape if needed
        if original_shape[0] != self.size or original_shape[1] != self.size:
            result = cv2.resize(result, (original_shape[1], original_shape[0]))
        
        # Convert back to uint8 if input was uint8
        if image.dtype == np.uint8:
            result = (result * 255).astype(np.uint8)
            
        process_time = time.time() - start_time
        print(f"DeepFill inpainting completed in {process_time:.2f} seconds")
            
        return result
    
    def __del__(self):
        """Clean up resources"""
        if self.sess is not None:
            self.sess.close()


# Example usage when run as a script
if __name__ == "__main__":
    # Path to the model
    model_path = './model/pretrained_model'
    
    # Initialize the inpainter
    inpainter = DeepFillInpainter(model_path)
    
    if not inpainter.is_model_loaded():
        print("Failed to load model. Exiting.")
        exit(1)
    
    # Test on a sample image
    image_path = './images/test.jpg'
    if os.path.exists(image_path):
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a sample mask (center square)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h//4:3*h//4, w//4:3*w//4] = 255  # Center square
        
        # Perform inpainting
        result = inpainter.inpaint(image, mask)
        
        # Display results
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        
        plt.subplot(132)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        
        plt.subplot(133)
        plt.imshow(result)
        plt.title('Inpainted Result')
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Test image not found at {image_path}")
