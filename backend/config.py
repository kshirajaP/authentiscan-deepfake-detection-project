"""
Configuration file for AuthentiScan backend
Contains all configurable parameters
"""
import os

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # File upload configuration
    UPLOAD_FOLDER = 'uploads'
    RESULTS_FOLDER = 'results'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}
    
    # Model configuration
    CONFIDENCE_THRESHOLD = 0.5  # Threshold for fake classification
    FRAME_SAMPLE_RATE = 10  # Process every Nth frame
    
    # API configuration
    API_VERSION = 'v1.0.0'
    API_TITLE = 'AuthentiScan API'