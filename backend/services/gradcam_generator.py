"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Generator
Works with both MesoNet and XceptionNet architectures
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class GradCAM:
    """
    Generate Grad-CAM heatmaps for CNN models
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: The CNN model (PyTorch)
            target_layer: The convolutional layer to visualize
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index (0 for real, 1 for fake)
        
        Returns:
            heatmap: Numpy array of heatmap [H, W]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Calculate weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def create_heatmap_overlay(original_image_path, heatmap, output_path, alpha=0.4):
    """
    Create visualization with heatmap overlaid on original image
    
    Args:
        original_image_path: Path to original image
        heatmap: Grad-CAM heatmap array [H, W]
        output_path: Where to save the result
        alpha: Transparency of heatmap overlay (0-1)
    """
    # Load original image
    img = cv2.imread(original_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Apply colormap (jet colormap: blue=low, red=high)
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlay = (alpha * heatmap_colored + (1 - alpha) * img).astype(np.uint8)
    
    # Create figure with side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Detection Focus)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap='jet'), 
        ax=axes, 
        fraction=0.046, 
        pad=0.04
    )
    cbar.set_label('Attention Intensity', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM visualization saved to {output_path}")


def get_target_layer(model):
    """
    Automatically find the best convolutional layer for Grad-CAM
    
    Args:
        model: PyTorch model
    
    Returns:
        Target layer for Grad-CAM
    """
    # Try common layer names
    if hasattr(model, 'base_model'):
        # For custom models with base_model attribute
        base = model.base_model
        
        # Try ResNet-style layers
        if hasattr(base, 'layer4'):
            return base.layer4[-1]
        
        # Try to find last conv layer
        conv_layers = []
        for name, module in base.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(module)
        
        if conv_layers:
            return conv_layers[-1]
    
    # Direct model search
    if hasattr(model, 'layer4'):
        return model.layer4[-1]
    
    # Find last conv layer in entire model
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)
    
    if conv_layers:
        return conv_layers[-1]
    
    raise ValueError("Could not find suitable convolutional layer for Grad-CAM")


def generate_gradcam_for_image(model, image_path, output_path, target_layer=None):
    """
    Complete pipeline to generate Grad-CAM for an image
    
    Args:
        model: Trained PyTorch model
        image_path: Path to input image
        output_path: Path to save visualization
        target_layer: Specific layer to visualize (auto-detected if None)
    
    Returns:
        heatmap: The generated heatmap array
    """
    from torchvision import transforms
    
    # Get target layer
    if target_layer is None:
        target_layer = get_target_layer(model)
    
    print(f"Using layer: {target_layer.__class__.__name__}")
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    
    # Move to same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Generate heatmap
    heatmap = gradcam.generate(input_tensor)
    
    # Create overlay visualization
    create_heatmap_overlay(image_path, heatmap, output_path)
    
    return heatmap


def generate_gradcam_for_video(model, video_path, output_dir, num_frames=5):
    """
    Generate Grad-CAM for multiple frames in a video
    
    Args:
        model: Trained model
        video_path: Path to video file
        output_dir: Directory to save frame heatmaps
        num_frames: Number of frames to analyze
    
    Returns:
        List of output paths
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    output_paths = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Save frame temporarily
        temp_frame_path = os.path.join(output_dir, f'temp_frame_{i}.jpg')
        cv2.imwrite(temp_frame_path, frame)
        
        # Generate Grad-CAM
        output_path = os.path.join(output_dir, f'gradcam_frame_{i}.jpg')
        
        try:
            generate_gradcam_for_image(model, temp_frame_path, output_path)
            output_paths.append(output_path)
        except Exception as e:
            print(f"Failed to generate Grad-CAM for frame {i}: {e}")
        
        # Clean up temp file
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
    
    cap.release()
    
    print(f"Generated Grad-CAM for {len(output_paths)} frames")
    return output_paths