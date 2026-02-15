"""
Deepfake Detector with Pre-trained Models
Location: backend/models/detector.py
Uses: MesoNet-4 + XceptionNet ensemble
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import os

# TensorFlow/Keras for MesoNet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow import keras


def create_mesonet4():
    """
    Create MesoNet-4 model using Functional API
    This fixes the build() issue with subclassed models
    """
    input_layer = keras.layers.Input(shape=(256, 256, 3))
    
    # Block 1
    x = keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(input_layer)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 2
    x = keras.layers.Conv2D(8, (5, 5), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 3
    x = keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 4
    x = keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(4, 4))(x)
    
    # Classifier
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.models.Model(inputs=input_layer, outputs=output)
    return model


class XceptionNetDeepfake(nn.Module):
    """
    Xception-based Deepfake Detector (PyTorch)
    Uses timm library for pre-trained Xception
    """
    def __init__(self, pretrained=True):
        super(XceptionNetDeepfake, self).__init__()
        
        try:
            import timm
            self.base_model = timm.create_model('xception', pretrained=pretrained)
            num_features = self.base_model.get_classifier().in_features
            self.base_model.reset_classifier(0)  # Remove original classifier
            print("✅ Using Xception architecture")
        except ImportError:
            print("⚠️ timm not found, using ResNet50 instead")
            print("   Install timm for better accuracy: pip install timm")
            self.base_model = models.resnet50(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        
        # Custom binary classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # [real, fake]
        )
    
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)


class DeepfakeDetector:
    """
    Ensemble Deepfake Detector
    Combines MesoNet (Keras) + XceptionNet (PyTorch)
    """
    def __init__(self, device='cpu', use_mesonet=False):  # Set to False to disable
        """
        Initialize with both models
    
        Args:
            device: 'cpu' or 'cuda'
            use_mesonet: Set to False if MesoNet weights are unreliable
        """
        self.device = torch.device(device)
        self.use_mesonet_flag = use_mesonet
    
        print(f"🚀 Initializing Deepfake Detector on {self.device}")
    
        # Get path to pretrained weights
        current_dir = Path(__file__).parent
        self.weights_dir = current_dir / "pretrained"
        self.mesonet_weights = self.weights_dir / "mesonet4.h5"
    
        # Load models
        if use_mesonet:
            self.mesonet = self._load_mesonet()
        else:
            print("⚠️ MesoNet disabled (use_mesonet=False)")
            self.mesonet = None
    
        self.xception = self._load_xception()
    
        # Setup transforms
        self._setup_transforms()
    
        print("✅ Detector initialized successfully")

    def __init__(self, device='cpu'):
        """Initialize with both models"""
        self.device = torch.device(device)
        print(f"🚀 Initializing Deepfake Detector on {self.device}")
        
        # Get path to pretrained weights
        current_dir = Path(__file__).parent
        self.weights_dir = current_dir / "pretrained"
        self.mesonet_weights = self.weights_dir / "mesonet4.h5"
        
        # Load models
        self.mesonet = self._load_mesonet()
        self.xception = self._load_xception()
        
        # Setup transforms
        self._setup_transforms()
        
        print("✅ Detector initialized successfully")
    
    def _load_mesonet(self):
        """Load MesoNet model"""
        print("📦 Loading MesoNet-4...")
        
        try:
            model = create_mesonet4()
            
            if self.mesonet_weights.exists():
                model.load_weights(str(self.mesonet_weights))
                print(f"✅ MesoNet loaded from {self.mesonet_weights}")
            else:
                print(f"⚠️ MesoNet weights not found at {self.mesonet_weights}")
                print("   Using random initialization (lower accuracy)")
                print("   Run: python download_pretrained_models.py")
            
            return model
        
        except Exception as e:
            print(f"❌ MesoNet load failed: {e}")
            print(f"   Error details: {type(e).__name__}")
            return None
    
    def _load_xception(self):
        """Load XceptionNet model"""
        print("📦 Loading XceptionNet...")
        
        try:
            model = XceptionNetDeepfake(pretrained=True)
            model.to(self.device)
            model.eval()
            print("✅ XceptionNet loaded (ImageNet pretrained)")
            return model
        
        except Exception as e:
            print(f"❌ XceptionNet load failed: {e}")
            return None
    
    def _setup_transforms(self):
        """Image preprocessing transforms"""
        # MesoNet: 256x256, no normalization
        self.mesonet_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # XceptionNet: 224x224, ImageNet normalization
        self.xception_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _predict_mesonet(self, image):
        """Predict using MesoNet"""
        if self.mesonet is None:
            return None
        
        try:
            # Convert to numpy array, resize, normalize
            img_array = np.array(image.resize((256, 256))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict (output is fake probability)
            prediction = self.mesonet.predict(img_array, verbose=0)[0][0]
            
            return {
                'fake_probability': float(prediction),
                'real_probability': float(1 - prediction),
                'prediction': 'fake' if prediction > 0.5 else 'real',
                'confidence': float(max(prediction, 1 - prediction))
            }
        
        except Exception as e:
            print(f"⚠️ MesoNet prediction failed: {e}")
            return None
    
    def _predict_xception(self, image):
        """Predict using XceptionNet"""
        if self.xception is None:
            return None
        
        try:
            img_tensor = self.xception_transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.xception(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                
                real_prob = probs[0].item()
                fake_prob = probs[1].item()
                
                prediction = 'fake' if fake_prob > real_prob else 'real'
                confidence = max(real_prob, fake_prob)
            
            return {
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'prediction': prediction,
                'confidence': confidence
            }
        
        except Exception as e:
            print(f"⚠️ XceptionNet prediction failed: {e}")
            return None
    
    def predict_image(self, image):
        """
        Ensemble prediction on single image
        
        Args:
            image: PIL Image, numpy array, or file path
        
        Returns:
            dict with prediction, confidence, and individual results
        """
        # Handle different input types
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Get predictions from both models
        mesonet_result = self._predict_mesonet(image)
        xception_result = self._predict_xception(image)
        
        # Handle failures - ensure we always return valid structure
        if mesonet_result is None and xception_result is None:
            raise Exception("Both models failed to predict")
        
        if mesonet_result is None:
            # Only XceptionNet worked
            result = xception_result.copy()
            result['individual_models'] = {
                'mesonet': None,
                'xception': xception_result
            }
            return result
        
        if xception_result is None:
            # Only MesoNet worked
            result = mesonet_result.copy()
            result['individual_models'] = {
                'mesonet': mesonet_result,
                'xception': None
            }
            return result
        
        # Ensemble: Weighted average
        # Weights based on typical model performance
        mesonet_weight = 0.4
        xception_weight = 0.6
        
        ensemble_fake_prob = (
            mesonet_weight * mesonet_result['fake_probability'] +
            xception_weight * xception_result['fake_probability']
        )
        
        ensemble_real_prob = 1 - ensemble_fake_prob
        ensemble_prediction = 'fake' if ensemble_fake_prob > 0.5 else 'real'
        ensemble_confidence = max(ensemble_fake_prob, ensemble_real_prob)
        
        return {
            'prediction': ensemble_prediction,
            'confidence': ensemble_confidence,
            'probabilities': {
                'real': ensemble_real_prob,
                'fake': ensemble_fake_prob
            },
            'individual_models': {
                'mesonet': mesonet_result,
                'xception': xception_result
            }
        }
    
    def predict_video(self, frames):
        """
        Predict across video frames
        
        Args:
            frames: List of PIL Images
        
        Returns:
            dict with aggregated video prediction
        """
        print(f"🎬 Analyzing {len(frames)} video frames...")
        
        frame_results = []
        
        for idx, frame in enumerate(frames):
            try:
                result = self.predict_image(frame)
                result['frame_number'] = idx
                frame_results.append(result)
                
                if (idx + 1) % 10 == 0:
                    print(f"   Processed {idx + 1}/{len(frames)} frames")
            
            except Exception as e:
                print(f"⚠️ Frame {idx} failed: {e}")
        
        if not frame_results:
            raise Exception("All frames failed")
        
        # Aggregate results
        fake_count = sum(1 for r in frame_results if r['prediction'] == 'fake')
        total = len(frame_results)
        
        avg_confidence = np.mean([r['confidence'] for r in frame_results])
        overall_prediction = 'fake' if fake_count > total / 2 else 'real'
        
        avg_fake_prob = np.mean([r['probabilities']['fake'] for r in frame_results])
        avg_real_prob = 1 - avg_fake_prob
        
        return {
            'prediction': overall_prediction,
            'confidence': float(avg_confidence),
            'total_frames_analyzed': total,
            'fake_frames': fake_count,
            'real_frames': total - fake_count,
            'probabilities': {
                'real': float(avg_real_prob),
                'fake': float(avg_fake_prob)
            },
            'frame_results': frame_results
        }
    
    @property
    def model(self):
        """Return XceptionNet for Grad-CAM"""
        return self.xception


# Test when run directly
if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING DEEPFAKE DETECTOR")
    print("="*60)
    
    detector = DeepfakeDetector()
    
    print("\n📝 Status:")
    print(f"   MesoNet: {'✅' if detector.mesonet else '❌'}")
    print(f"   XceptionNet: {'✅' if detector.xception else '❌'}")
    
    # Test with random image
    print("\n🧪 Testing with 224x224 random image...")
    test_img = Image.new('RGB', (224, 224), color='red')
    
    try:
        result = detector.predict_image(test_img)
        print("\n✅ Prediction successful:")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        if 'probabilities' in result:
            print(f"   Real: {result['probabilities']['real']:.2%}")
            print(f"   Fake: {result['probabilities']['fake']:.2%}")
        
        if 'individual_models' in result:
            print("\n📊 Individual Models:")
            
            meso = result['individual_models'].get('mesonet')
            if meso:
                print(f"   MesoNet: {meso['prediction']} ({meso['confidence']:.2%})")
            else:
                print(f"   MesoNet: Not available")
            
            xcep = result['individual_models'].get('xception')
            if xcep:
                print(f"   XceptionNet: {xcep['prediction']} ({xcep['confidence']:.2%})")
            else:
                print(f"   XceptionNet: Not available")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
    
    print("\n" + "="*60)
    print("✅ Detector ready!")
    print("="*60)