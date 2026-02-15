"""
Download Pre-trained Deepfake Detection Models
Save to: backend/models/pretrained/
Run from project root: python backend/download_pretrained_models.py
"""
import os
import requests
from pathlib import Path

def download_file(url, destination):
    """Download file from URL with progress"""
    print(f"Downloading {destination.name}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"Progress: {percent:.1f}%", end='\r')
    
    print(f"\n✅ Downloaded {destination.name}")

def setup_models_directory():
    """Create backend/models/pretrained/ directory"""
    # Get backend directory (parent of this script)
    backend_dir = Path(__file__).parent
    models_dir = backend_dir / "models" / "pretrained"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Models directory: {models_dir}")
    return models_dir

def download_mesonet(models_dir):
    """Download MesoNet pre-trained weights"""
    print("\n" + "="*50)
    print("📦 Downloading MesoNet-4")
    print("="*50)
    
    mesonet_url = "https://github.com/DariusAf/MesoNet/raw/master/weights/Meso4_DF.h5"
    mesonet_path = models_dir / "mesonet4.h5"
    
    if mesonet_path.exists():
        print(f"✅ MesoNet already exists at {mesonet_path}")
        return mesonet_path
    
    try:
        download_file(mesonet_url, mesonet_path)
        return mesonet_path
    except Exception as e:
        print(f"❌ Failed to download MesoNet: {e}")
        print("💡 Download manually from: https://github.com/DariusAf/MesoNet/tree/master/weights")
        print(f"   Save to: {mesonet_path}")
        return None

def download_xceptionnet(models_dir):
    """Setup XceptionNet (will use timm's pretrained)"""
    print("\n" + "="*50)
    print("📦 XceptionNet Setup")
    print("="*50)
    
    print("XceptionNet will be downloaded automatically via timm library")
    print("Make sure you have timm installed: pip install timm")
    print("✅ No manual download needed")
    return True

def create_requirements():
    """Create requirements.txt in backend/"""
    backend_dir = Path(__file__).parent
    req_file = backend_dir / "requirements.txt"
    
    if req_file.exists():
        print(f"\n✅ requirements.txt already exists at {req_file}")
        return
    
    requirements = """# Deep Learning Frameworks
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.12.0
keras>=2.12.0

# Model utilities
timm>=0.9.0

# Image/Video Processing
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0

# Flask Backend
Flask>=2.3.0
flask-cors>=4.0.0
werkzeug>=2.3.0

# Utilities
requests>=2.31.0

# Visualization (for Grad-CAM)
matplotlib>=3.7.0
"""
    
    with open(req_file, "w") as f:
        f.write(requirements)
    
    print(f"\n✅ Created {req_file}")

def main():
    """Main download script"""
    print("="*60)
    print("🚀 AUTHENTISCAN - MODEL SETUP")
    print("="*60)
    
    print("\nThis will download pre-trained models to:")
    print("backend/models/pretrained/")
    print("\nModels:")
    print("1. MesoNet-4 (~2MB)")
    print("2. XceptionNet (auto via timm)")
    print()
    
    # Setup directory
    models_dir = setup_models_directory()
    
    # Create requirements if needed
    create_requirements()
    
    # Download models
    mesonet_path = download_mesonet(models_dir)
    xception_ok = download_xceptionnet(models_dir)
    
    print("\n" + "="*60)
    print("📊 SETUP SUMMARY")
    print("="*60)
    
    if mesonet_path and mesonet_path.exists():
        print(f"✅ MesoNet: {mesonet_path}")
    else:
        print("⚠️ MesoNet: Download manually (optional)")
    
    if xception_ok:
        print("✅ XceptionNet: Will auto-download via timm")
    
    print("\n" + "="*60)
    print("🎯 NEXT STEPS")
    print("="*60)
    print("1. Install dependencies:")
    print("   cd backend")
    print("   pip install -r requirements.txt")
    print()
    print("2. Test detector:")
    print("   python -m models.detector")
    print()
    print("3. Start backend:")
    print("   python app.py")
    print()
    print("✅ Setup complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")