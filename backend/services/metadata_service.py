"""
Metadata Extraction Service
Extracts and analyzes file metadata for inconsistencies
"""
from PIL import Image
from PIL.ExifTags import TAGS
import os
from datetime import datetime

class MetadataService:
    """
    Extracts and validates media file metadata
    """
    
    @staticmethod
    def extract_exif(image_path):
        """
        Extract EXIF data from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with EXIF data
        """
        try:
            image = Image.open(image_path)
            exif_data = image.getexif()
            
            if not exif_data:
                return {'status': 'no_exif', 'data': {}}
            
            exif_dict = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_dict[tag_name] = str(value)
            
            return {'status': 'success', 'data': exif_dict}
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def detect_inconsistencies(exif_data):
        """
        Detect suspicious patterns in EXIF data
        
        Args:
            exif_data: Extracted EXIF dictionary
            
        Returns:
            List of inconsistency warnings
        """
        inconsistencies = []
        
        if not exif_data or exif_data.get('status') != 'success':
            inconsistencies.append("No EXIF data found (suspicious for modern cameras)")
            return inconsistencies
        
        data = exif_data.get('data', {})
        
        # Check for missing critical fields
        if 'Make' not in data and 'Model' not in data:
            inconsistencies.append("Missing camera make/model information")
        
        if 'DateTime' not in data:
            inconsistencies.append("Missing DateTime metadata")
        else:
            # Check timestamp validity
            try:
                dt_str = data['DateTime']
                # Format: "YYYY:MM:DD HH:MM:SS"
                dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                
                if dt > datetime.now():
                    inconsistencies.append("DateTime is in the future")
                
                if dt.year < 2000:
                    inconsistencies.append("Unusually old timestamp")
            except:
                inconsistencies.append("Invalid DateTime format")
        
        # Check for software editing
        if 'Software' in data:
            software = data['Software'].lower()
            editing_software = ['photoshop', 'gimp', 'paint.net', 'pixlr']
            if any(s in software for s in editing_software):
                inconsistencies.append(f"Image edited with {data['Software']}")
        
        return inconsistencies
    
    @staticmethod
    def analyze_file(filepath):
        """
        Complete metadata analysis
        
        Args:
            filepath: Path to file
            
        Returns:
            dict with metadata and inconsistencies
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            exif_data = MetadataService.extract_exif(filepath)
            inconsistencies = MetadataService.detect_inconsistencies(exif_data)
            
            return {
                'exif': exif_data,
                'inconsistencies': inconsistencies,
                'risk_level': 'high' if len(inconsistencies) > 2 else 'medium' if inconsistencies else 'low'
            }
        else:
            return {
                'message': 'Metadata extraction only supported for images',
                'exif': {},
                'inconsistencies': [],
                'risk_level': 'unknown'
            }