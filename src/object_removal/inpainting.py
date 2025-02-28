import time
import numpy as np
import cv2
import os

# Try to import tensorflow - we'll use a fallback if not available
try:
    import tensorflow as tf

    TF_VERSION = tf.__version__
    if TF_VERSION.startswith("2."):
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

    from src.object_removal.model.model import Model

    TF_AVAILABLE = True
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    TF_AVAILABLE = False
except Exception as e:
    print(f"TensorFlow setup error: {e}")
    TF_AVAILABLE = False


def remove_object(image, mask=None, method="deepfill"):
    """
    Remove objects from images using inpainting techniques.

    Args:
        image: Input image as numpy array
        mask: Binary mask where white (1) indicates pixels to be removed
        method: Inpainting method to use ('deepfill', 'patchmatch', 'generative')

    Returns:
        Image with object removed
    """
    if mask is None or image is None:
        print("Error: Image or mask is None")
        return image

    # Make sure image is in the right format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Debug output
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")

    # Make sure mask is binary and properly sized
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    # Ensure mask has same dimensions as image
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Convert to single channel mask if it has 3 channels
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Ensure mask values are either 0 or 255
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Debug output
    print(f"Processed mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Mask values: min={mask.min()}, max={mask.max()}")

    # Choose the appropriate inpainting method
    if method == "deepfill" and TF_AVAILABLE:
        try:
            return deepfill_inpainting(image, mask)
        except Exception as e:
            print(f"DeepFill error: {e}, falling back to OpenCV")
            # Fall back to OpenCV's method
            return cv2_inpainting(image, mask)

    elif method == "patchmatch":
        return patchmatch_inpainting(image, mask)

    elif method == "generative":
        return cv2_inpainting(image, mask, advanced=True)

    # Default fallback
    return cv2_inpainting(image, mask)


def cv2_inpainting(image, mask, advanced=False):
    """Basic inpainting using OpenCV"""
    try:
        # Ensure image is uint8 type
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Ensure mask is uint8 single channel
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Ensure mask values are binary (0 or 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Debug information
        print(f"Final image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Final mask shape: {mask.shape}, dtype: {mask.dtype}")

        # OpenCV inpainting requires continuous non-zero pixels
        # Dilate the mask slightly to ensure better coverage
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # For inpaint API: mask should be 8-bit 1-channel image
        if advanced and hasattr(cv2, "INPAINT_TELEA"):
            # Use more advanced method if available
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        else:
            # Use basic method
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

        print("Inpainting completed successfully")
        return result

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"OpenCV inpainting error: {e}")
        # Return the original image if inpainting fails
        return image


def patchmatch_inpainting(image, mask):
    """Inpainting using PatchMatch algorithm"""
    try:
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Ensure mask is properly formatted for OpenCV
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # If available, use Photo module's inpainting
        if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "inpaint"):
            try:
                return cv2.xphoto.inpaint(image, mask)
            except Exception as e:
                print(f"PatchMatch error: {e}, falling back to standard inpainting")

        # Fall back to standard inpainting with larger radius
        return cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)

    except Exception as e:
        print(f"PatchMatch inpainting error: {e}")
        return image


def deepfill_inpainting(image, mask):
    """Advanced inpainting using DeepFill algorithm"""
    if not TF_AVAILABLE:
        return cv2_inpainting(image, mask)

    # Simplified implementation for now - future enhancement
    # This should be replaced with actual DeepFill implementation when ready
    print("DeepFill requested but using OpenCV inpainting as fallback")
    return cv2_inpainting(image, mask, advanced=True)
