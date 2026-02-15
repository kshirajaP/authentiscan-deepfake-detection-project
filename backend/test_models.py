"""
Model Testing and Debugging Script
Run this to verify both models are working correctly
Location: backend/test_models.py

Usage: python test_models.py [path_to_test_image]
"""
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.detector import DeepfakeDetector


def test_model_outputs(image_path=None):
    """Test both models with detailed output"""
    
    print("="*70)
    print("🔬 DEEPFAKE DETECTOR MODEL DIAGNOSTICS")
    print("="*70)
    
    # Initialize detector
    print("\n1️⃣ Initializing detector...")
    detector = DeepfakeDetector()
    
    # Check model status
    print("\n2️⃣ Model Status:")
    print(f"   MesoNet loaded: {'✅ YES' if detector.mesonet else '❌ NO'}")
    print(f"   XceptionNet loaded: {'✅ YES' if detector.xception else '❌ NO'}")
    
    if not detector.mesonet and not detector.xception:
        print("\n❌ CRITICAL: Both models failed to load!")
        return
    
    # Create or load test image
    if image_path and Path(image_path).exists():
        print(f"\n3️⃣ Loading test image: {image_path}")
        test_image = Image.open(image_path).convert('RGB')
    else:
        print("\n3️⃣ Creating synthetic test image (224x224)...")
        # Create a test image with some structure
        test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)
    
    print(f"   Image size: {test_image.size}")
    print(f"   Image mode: {test_image.mode}")
    
    # Test MesoNet individually
    print("\n4️⃣ Testing MesoNet-4:")
    print("-" * 70)
    if detector.mesonet:
        try:
            meso_result = detector._predict_mesonet(test_image)
            if meso_result:
                print(f"   ✅ MesoNet prediction successful")
                print(f"   Prediction: {meso_result['prediction']}")
                print(f"   Confidence: {meso_result['confidence']:.4f} ({meso_result['confidence']*100:.2f}%)")
                print(f"   Real probability: {meso_result['real_probability']:.4f}")
                print(f"   Fake probability: {meso_result['fake_probability']:.4f}")
                
                # Sanity checks
                if meso_result['confidence'] > 0.99:
                    print("   ⚠️ WARNING: Confidence is suspiciously high (>99%)")
                    print("   This suggests the model may not be trained properly")
                elif meso_result['confidence'] < 0.51:
                    print("   ⚠️ WARNING: Confidence is very low (<51%)")
                    print("   Model is uncertain about the prediction")
            else:
                print("   ❌ MesoNet prediction failed")
        except Exception as e:
            print(f"   ❌ MesoNet error: {e}")
    else:
        print("   ⚠️ MesoNet not loaded")
    
    # Test XceptionNet individually
    print("\n5️⃣ Testing XceptionNet:")
    print("-" * 70)
    if detector.xception:
        try:
            xcep_result = detector._predict_xception(test_image)
            if xcep_result:
                print(f"   ✅ XceptionNet prediction successful")
                print(f"   Prediction: {xcep_result['prediction']}")
                print(f"   Confidence: {xcep_result['confidence']:.4f} ({xcep_result['confidence']*100:.2f}%)")
                print(f"   Real probability: {xcep_result['real_probability']:.4f}")
                print(f"   Fake probability: {xcep_result['fake_probability']:.4f}")
                
                # Sanity checks
                if xcep_result['confidence'] > 0.99:
                    print("   ⚠️ WARNING: Confidence is suspiciously high (>99%)")
                    print("   This suggests the model may not be trained properly")
                elif xcep_result['confidence'] < 0.51:
                    print("   ⚠️ WARNING: Confidence is very low (<51%)")
                    print("   Model is uncertain about the prediction")
            else:
                print("   ❌ XceptionNet prediction failed")
        except Exception as e:
            print(f"   ❌ XceptionNet error: {e}")
    else:
        print("   ⚠️ XceptionNet not loaded")
    
    # Test ensemble
    print("\n6️⃣ Testing Ensemble:")
    print("-" * 70)
    try:
        ensemble_result = detector.predict_image(test_image)
        print(f"   ✅ Ensemble prediction successful")
        print(f"   Final Prediction: {ensemble_result['prediction']}")
        print(f"   Final Confidence: {ensemble_result['confidence']:.4f} ({ensemble_result['confidence']*100:.2f}%)")
        print(f"   Real probability: {ensemble_result['probabilities']['real']:.4f}")
        print(f"   Fake probability: {ensemble_result['probabilities']['fake']:.4f}")
        
        # Show individual contributions
        if 'individual_models' in ensemble_result:
            print("\n   Individual Model Contributions:")
            
            meso = ensemble_result['individual_models'].get('mesonet')
            if meso:
                print(f"      MesoNet: {meso['prediction']} ({meso['confidence']*100:.2f}%)")
            else:
                print(f"      MesoNet: Not available")
            
            xcep = ensemble_result['individual_models'].get('xception')
            if xcep:
                print(f"      XceptionNet: {xcep['prediction']} ({xcep['confidence']*100:.2f}%)")
            else:
                print(f"      XceptionNet: Not available")
    except Exception as e:
        print(f"   ❌ Ensemble error: {e}")
    
    # Diagnosis
    print("\n7️⃣ Diagnosis:")
    print("-" * 70)
    
    issues_found = []
    
    if not detector.mesonet:
        issues_found.append("MesoNet failed to load - download weights")
    elif meso_result and meso_result['confidence'] > 0.99:
        issues_found.append("MesoNet shows 100% confidence - likely untrained/random weights")
    
    if not detector.xception:
        issues_found.append("XceptionNet failed to load")
    
    if issues_found:
        print("   ❌ Issues found:")
        for issue in issues_found:
            print(f"      • {issue}")
    else:
        print("   ✅ All models appear to be working normally")
    
    print("\n" + "="*70)
    print("📊 RECOMMENDATIONS:")
    print("="*70)
    
    if not detector.mesonet:
        print("1. Run: python download_pretrained_models.py")
        print("   To download MesoNet weights")
    
    if detector.mesonet and meso_result and meso_result['confidence'] > 0.99:
        print("1. MesoNet weights may be corrupted or incompatible")
        print("   Try re-downloading: python download_pretrained_models.py")
    
    if detector.xception and xcep_result and xcep_result['confidence'] < 0.6:
        print("2. XceptionNet is using ImageNet weights (not trained on deepfakes)")
        print("   This is normal - it will have lower accuracy on deepfakes")
        print("   For better results, fine-tune on deepfake dataset")
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    # Check for command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_model_outputs(image_path)
    else:
        print("Usage: python test_models.py [path_to_image]")
        print("Running with synthetic test image...\n")
        test_model_outputs()