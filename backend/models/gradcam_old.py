"""
Grad-CAM Implementation for Explainability
Generates heatmaps showing which regions influence detection
"""
import torch
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    Visualizes which parts of image contribute to prediction
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Layer to generate CAM from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Save activations during forward pass"""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Save gradients during backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_image, target_class=1):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Preprocessed input tensor
            target_class: Class to generate CAM for (1=fake)
            
        Returns:
            Heatmap as numpy array
        """
        # Forward pass
        output = self.model(input_image)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target = output[0, target_class]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image (numpy array, RGB)
            heatmap: Grad-CAM heatmap [0,1]
            alpha: Transparency of overlay
            
        Returns:
            Image with heatmap overlay
        """
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlayed = heatmap_colored * alpha + image * (1 - alpha)
        overlayed = overlayed.astype(np.uint8)
        
        return overlayed