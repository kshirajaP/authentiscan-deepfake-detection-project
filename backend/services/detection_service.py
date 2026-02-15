"""
Detection Service
Orchestrates video processing and detection workflow with REAL ensemble
Location: backend/services/detection_service.py
"""
import cv2
import numpy as np
from PIL import Image
import os

class DetectionService:
    """
    High-level service for detection workflows
    """
    
    def __init__(self, detector):
        """
        Initialize service with detector
        
        Args:
            detector: DeepfakeDetector instance with real models
        """
        self.detector = detector
    
    def extract_frames(self, video_path, sample_rate=10):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame
            
        Returns:
            List of PIL Images
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            
            frame_count += 1
        
        cap.release()
        
        print(f"Extracted {len(frames)} frames from {total_frames} total frames")
        return frames
    
    def analyze_image(self, image_path):
        """
        Analyze single image with REAL ensemble
        
        Args:
            image_path: Path to image file
            
        Returns:
            Detection results dict
        """
        image = Image.open(image_path).convert('RGB')
        result = self.detector.predict_image(image)
        
        return {
            'type': 'image',
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'individual_models': result.get('individual_models', {})
        }
    
    def analyze_video(self, video_path, sample_rate=10):
        """
        Analyze video file with REAL ensemble
        
        Args:
            video_path: Path to video file
            sample_rate: Frame sampling rate
            
        Returns:
            Detection results dict
        """
        print(f"Analyzing video: {video_path}")
        
        # Extract frames
        frames = self.extract_frames(video_path, sample_rate)
        
        if not frames:
            return {'error': 'Could not extract frames from video'}
        
        # Run REAL detection (not simulated)
        result = self.detector.predict_video(frames)
        
        # Add video-specific info
        result['type'] = 'video'
        result['sample_rate'] = sample_rate
        
        return result
    
    def analyze_file(self, filepath):
        """
        Analyze file (auto-detect type)
        
        Args:
            filepath: Path to media file
            
        Returns:
            Detection results dict
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.mp4', '.avi', '.mov']:
            return self.analyze_video(filepath)
        elif ext in ['.jpg', '.jpeg', '.png']:
            return self.analyze_image(filepath)
        else:
            return {'error': f'Unsupported file type: {ext}'}
    
    def analyze_with_comparison(self, filepath):
        """
        Analyze file and show REAL ensemble vs. individual model comparison
        DIFFERENTIATOR: Shows actual advantage of multi-model approach
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.mp4', '.avi', '.mov']:
            # Video analysis
            frames = self.extract_frames(filepath)
            
            if not frames:
                return {'error': 'Could not extract frames from video'}
            
            # Get REAL ensemble results
            ensemble_result = self.detector.predict_video(frames)
            
            # Extract individual model performances from frame results
            mesonet_confidences = []
            xception_confidences = []
            
            for frame_result in ensemble_result.get('frame_results', []):
                if 'individual_models' in frame_result:
                    meso = frame_result['individual_models'].get('mesonet')
                    xcep = frame_result['individual_models'].get('xception')
                    
                    if meso:
                        mesonet_confidences.append(meso['confidence'])
                    if xcep:
                        xception_confidences.append(xcep['confidence'])
            
            mesonet_avg = np.mean(mesonet_confidences) if mesonet_confidences else 0
            xception_avg = np.mean(xception_confidences) if xception_confidences else 0
            
            # Calculate improvement over BEST single model
            best_single_model = max(mesonet_avg, xception_avg)
            if best_single_model > 0:
                improvement = ((ensemble_result['confidence'] - best_single_model) / best_single_model * 100)
            else:
                improvement = 0
            
            return {
                **ensemble_result,
                'comparison': {
                    'model1': {
                        'name': 'MesoNet-4',
                        'prediction': ensemble_result['prediction'],
                        'confidence': float(mesonet_avg)
                    },
                    'model2': {
                        'name': 'XceptionNet',
                        'prediction': ensemble_result['prediction'],
                        'confidence': float(xception_avg)
                    },
                    'ensemble': {
                        'name': 'Weighted Ensemble',
                        'prediction': ensemble_result['prediction'],
                        'confidence': float(ensemble_result['confidence'])
                    },
                    'improvement': f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
                }
            }
        
        elif ext in ['.jpg', '.jpeg', '.png']:
            # Image analysis
            image = Image.open(filepath).convert('RGB')
            result = self.detector.predict_image(image)
            
            # Extract individual model results
            mesonet_result = result.get('individual_models', {}).get('mesonet')
            xception_result = result.get('individual_models', {}).get('xception')
            
            # Get confidences, handle None case
            mesonet_conf = mesonet_result.get('confidence', 0) if mesonet_result else 0
            xception_conf = xception_result.get('confidence', 0) if xception_result else 0
            
            # Calculate improvement over BEST single model
            best_single_model = max(mesonet_conf, xception_conf)
            if best_single_model > 0:
                improvement = ((result['confidence'] - best_single_model) / best_single_model * 100)
            else:
                improvement = 0
            
            # Get predictions, handle None case
            mesonet_pred = mesonet_result.get('prediction', 'unavailable') if mesonet_result else 'unavailable'
            xception_pred = xception_result.get('prediction', 'unavailable') if xception_result else 'unavailable'
            
            return {
                'type': 'image',
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'comparison': {
                    'model1': {
                        'name': 'MesoNet-4',
                        'prediction': mesonet_pred,
                        'confidence': float(mesonet_conf)
                    },
                    'model2': {
                        'name': 'XceptionNet',
                        'prediction': xception_pred,
                        'confidence': float(xception_conf)
                    },
                    'ensemble': {
                        'name': 'Weighted Ensemble',
                        'prediction': result['prediction'],
                        'confidence': float(result['confidence'])
                    },
                    'improvement': f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
                }
            }
        
        else:
            return {'error': f'Unsupported file type: {ext}'}