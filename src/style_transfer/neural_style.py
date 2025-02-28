import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import os
import time

from .models import VGG16
from .utils import gram_matrix, denormalize, seed_everything

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_array, max_size=None):
    """
    Load an image from numpy array and convert it to a torch tensor.
    
    Args:
        image_array (numpy.ndarray): Image as numpy array (RGB)
        max_size (int, optional): Maximum size of the image
        
    Returns:
        torch.Tensor: Normalized image tensor
    """
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    
    # Resize image if needed
    if max_size is not None:
        if max(image.size) > max_size:
            size = max_size
            if image.width > image.height:
                size = (max_size, int(image.height * max_size / image.width))
            else:
                size = (int(image.width * max_size / image.height), max_size)
            image = image.resize(size, Image.LANCZOS)
    
    # Define transform pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform and add batch dimension
    image = transform(image).unsqueeze(0).to(device)
    
    return image

def neural_style_transfer(content_img, style_img, num_steps=300, style_weight=1e6, content_weight=1, 
                          progress_callback=None):
    """
    Perform neural style transfer between content and style images.
    
    Args:
        content_img (numpy.ndarray): Content image as numpy array (RGB)
        style_img (numpy.ndarray): Style image as numpy array (RGB)
        num_steps (int): Number of optimization steps
        style_weight (float): Weight for style loss
        content_weight (float): Weight for content loss
        progress_callback (function, optional): Callback function to report progress
        
    Returns:
        numpy.ndarray: Stylized image as numpy array (RGB)
    """
    # Set seed for reproducibility
    seed_everything(42)
    
    # Convert images to tensors
    content_tensor = load_image(content_img, max_size=512)
    style_tensor = load_image(style_img, max_size=512)
    
    # Initialize input image with content image
    input_tensor = content_tensor.clone().requires_grad_(True)
    
    # Load VGG16 model
    vgg = VGG16().to(device).eval()
    
    # Set up optimizer
    optimizer = optim.LBFGS([input_tensor], lr=1, max_iter=20)
    
    # Get features of content and style images
    content_features = vgg(content_tensor)
    style_features = vgg(style_tensor)
    
    # Calculate gram matrices for style features
    style_grams = [gram_matrix(style_feature) for style_feature in style_features]
    
    # Content layer and style layers for loss calculation
    content_layer = 2  # relu3_3
    style_layers = [0, 1, 2, 3]  # relu1_2, relu2_2, relu3_3, relu4_3
    
    # Style layer weights (you can adjust these)
    style_weights = [1.0, 1.0, 1.0, 1.0]
    
    # Track iteration count
    run = [0]
    
    # Function for loss calculation during optimization
    def closure():
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass through VGG16
        features = vgg(input_tensor)
        
        # Calculate content loss
        content_loss = content_weight * nn.MSELoss()(features[content_layer], content_features[content_layer])
        
        # Calculate style loss
        style_loss = 0
        for i, layer in enumerate(style_layers):
            current_feature = features[layer]
            current_gram = gram_matrix(current_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[i] * nn.MSELoss()(current_gram, style_gram)
            style_loss += layer_style_loss
        
        style_loss *= style_weight
        
        # Total loss
        total_loss = content_loss + style_loss
        
        # Backward pass
        total_loss.backward()
        
        # Update progress
        run[0] += 1
        if progress_callback and run[0] % 50 == 0:
            progress_callback(run[0], num_steps, total_loss.item())
        
        return total_loss
    
    # Optimization
    for _ in range(num_steps // 20 + 1):  # Divide by 20 as LBFGS performs 20 iterations per step
        if run[0] < num_steps:
            optimizer.step(closure)
    
    # Get final output
    with torch.no_grad():
        output = denormalize(input_tensor.clone())
        
    # Convert to numpy array
    output_img = output.cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    output_img = (output_img * 255).astype(np.uint8)
    
    return output_img

def apply_neural_style_transfer(content_image, style_image, style_weight=1e6, content_weight=1, 
                                iterations=300, progress_callback=None):
    """
    Apply neural style transfer between content and style images.
    
    Args:
        content_image (numpy.ndarray): Content image as numpy array
        style_image (numpy.ndarray): Style image as numpy array
        style_weight (float): Weight for style loss
        content_weight (float): Weight for content loss
        iterations (int): Number of optimization steps
        progress_callback (function, optional): Callback function to report progress
        
    Returns:
        numpy.ndarray: Stylized image as numpy array
    """
    # Ensure images are in RGB format
    if content_image.shape[2] == 3:  # Check if it's a 3-channel image
        content_rgb = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
        style_rgb = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
    else:
        content_rgb = content_image
        style_rgb = style_image
    
    # Apply style transfer
    start_time = time.time()
    
    result = neural_style_transfer(
        content_img=content_rgb,
        style_img=style_rgb,
        num_steps=iterations,
        style_weight=style_weight,
        content_weight=content_weight,
        progress_callback=progress_callback
    )
    
    elapsed_time = time.time() - start_time
    print(f"Style transfer completed in {elapsed_time:.2f} seconds")
    
    # Convert result back to BGR if input was BGR
    if content_image.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    return result
