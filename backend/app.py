"""
AuthentiScan Backend API
Main Flask application with all endpoints
Location: backend/app.py
"""
from flask import Flask, request, jsonify, send_from_directory
from security.security_middleware import security
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime

from config import Config
from models.detector import DeepfakeDetector
from services.detection_service import DetectionService
from services.metadata_service import MetadataService
from services.c2pa_service import C2PAService

# Initialize Flask app
app = Flask(__name__)

# Add security headers to all responses
@app.after_request
def add_security_headers(response):
    return security.add_security_headers(response)

app.config.from_object(Config)
CORS(app)  # Enable CORS for frontend

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize detector (global, loaded once)
print("Loading detection model...")
detector = DeepfakeDetector(device='cpu')
detection_service = DetectionService(detector)
print("Detection service ready")

# In-memory job storage (in production, use database)
jobs_db = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def home():
    """API root endpoint"""
    return jsonify({
        'service': app.config['API_TITLE'],
        'version': app.config['API_VERSION'],
        'status': 'online',
        'endpoints': {
            'upload': '/api/upload',
            'detect': '/api/detect/compare/<job_id>',
            'status': '/api/status/<job_id>',
            'metadata': '/api/metadata/<job_id>',
            'gradcam': '/api/gradcam/<job_id>',
            'c2pa': '/api/c2pa/<job_id>'
        }
    })


@app.route('/api/upload', methods=['POST'])
@security.rate_limit('upload')
def upload_file():
    """
    Upload media file for analysis
    Returns job_id for tracking
    """
    # Validate request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: MP4, AVI, MOV, JPG, PNG'}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': 'File too large. Maximum size: 100MB'}), 400
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save file with UUID filename
    ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
    saved_filename = f"{job_id}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
    file.save(filepath)

    # Content verification
    is_valid, verification_result = security.verify_file_content(filepath)
    if not is_valid:
        os.remove(filepath)  # Delete invalid file
        return jsonify({'error': verification_result}), 400
    
    # Calculate file hash for integrity
    file_hash = security.calculate_file_hash(filepath)
    
    # Create job record
    jobs_db[job_id] = {
        'job_id': job_id,
        'original_filename': file.filename,
        'filepath': filepath,
        'file_type': ext,
        'file_size': file_size,
        'file_hash': file_hash,
        'mime_type': verification_result,
        'status': 'uploaded',
        'upload_time': datetime.now().isoformat(),
        'result': None
    }
    
    print(f"File uploaded: {job_id} ({file.filename})")
    
    return jsonify({
        'job_id': job_id,
        'filename': file.filename,
        'file_size': file_size,
        'status': 'uploaded',
        'message': 'File uploaded successfully. Use /api/detect/compare/<job_id> to analyze.'
    }), 200


@app.route('/api/detect/compare/<job_id>', methods=['POST'])
@security.rate_limit('detect')
def detect_with_comparison(job_id):
    """
    Detection with ensemble comparison
    DIFFERENTIATOR: Shows multi-model advantage
    """
    if job_id not in jobs_db:
        return jsonify({'error': 'Job ID not found'}), 404
    
    job = jobs_db[job_id]
    
    if job['status'] != 'uploaded':
        return jsonify({'error': f'Job status is {job["status"]}'}), 400
    
    job['status'] = 'processing'
    
    try:
        result = detection_service.analyze_with_comparison(job['filepath'])
        
        job['result'] = result
        job['status'] = 'completed'
        job['completed_time'] = datetime.now().isoformat()
        
        return jsonify({
            'job_id': job_id,
            'status': 'completed',
            'result': result
        }), 200
    
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        print(f"Detection error for {job_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get job status and results"""
    if job_id not in jobs_db:
        return jsonify({'error': 'Job ID not found'}), 404
    
    job = jobs_db[job_id]
    
    return jsonify({
        'job_id': job_id,
        'filename': job['original_filename'],
        'status': job['status'],
        'upload_time': job['upload_time'],
        'result': job.get('result')
    }), 200


@app.route('/api/metadata/<job_id>', methods=['GET'])
def get_metadata(job_id):
    """Extract and analyze file metadata"""
    if job_id not in jobs_db:
        return jsonify({'error': 'Job ID not found'}), 404
    
    job = jobs_db[job_id]
    
    try:
        metadata = MetadataService.analyze_file(job['filepath'])
        
        return jsonify({
            'job_id': job_id,
            'metadata': metadata
        }), 200
    
    except Exception as e:
        return jsonify({
            'job_id': job_id,
            'error': str(e)
        }), 500


@app.route('/api/gradcam/<job_id>', methods=['GET'])
def get_gradcam(job_id):
    """
    Generate and return Grad-CAM heatmap
    """
    if job_id not in jobs_db:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs_db[job_id]
    filepath = job['filepath']
    file_type = job['file_type']
    
    try:
        # Import here to avoid circular imports
        from services.gradcam_generator import generate_gradcam_for_image, generate_gradcam_for_video
        
        # Create gradcam output directory
        gradcam_dir = os.path.join(app.config['RESULTS_FOLDER'], 'gradcam')
        os.makedirs(gradcam_dir, exist_ok=True)
        
        if file_type in ['jpg', 'jpeg', 'png']:
            # Generate for image
            output_path = os.path.join(gradcam_dir, f'{job_id}_gradcam.jpg')
            
            # Use the detector's model (XceptionNet)
            generate_gradcam_for_image(
                model=detector.model,
                image_path=filepath,
                output_path=output_path
            )
            
            return jsonify({
                'job_id': job_id,
                'type': 'image',
                'heatmap_url': f'/api/gradcam/image/{job_id}',
                'message': 'Grad-CAM heatmap generated successfully'
            }), 200
            
        elif file_type in ['mp4', 'avi', 'mov']:
            # Generate for video (multiple frames)
            video_gradcam_dir = os.path.join(gradcam_dir, job_id)
            
            output_paths = generate_gradcam_for_video(
                model=detector.model,
                video_path=filepath,
                output_dir=video_gradcam_dir,
                num_frames=5
            )
            
            return jsonify({
                'job_id': job_id,
                'type': 'video',
                'num_frames': len(output_paths),
                'heatmap_urls': [f'/api/gradcam/video/{job_id}/{i}' for i in range(len(output_paths))],
                'message': f'Grad-CAM generated for {len(output_paths)} frames'
            }), 200
        
        else:
            return jsonify({'error': 'Unsupported file type for Grad-CAM'}), 400
    
    except Exception as e:
        print(f"Grad-CAM generation error: {str(e)}")
        return jsonify({
            'error': f'Failed to generate Grad-CAM: {str(e)}'
        }), 500


@app.route('/api/gradcam/image/<job_id>', methods=['GET'])
def serve_gradcam_image(job_id):
    """
    Serve the Grad-CAM image file
    """
    gradcam_path = os.path.join(
        app.config['RESULTS_FOLDER'], 
        'gradcam', 
        f'{job_id}_gradcam.jpg'
    )
    
    if not os.path.exists(gradcam_path):
        return jsonify({'error': 'Grad-CAM image not found'}), 404
    
    return send_from_directory(
        os.path.join(app.config['RESULTS_FOLDER'], 'gradcam'),
        f'{job_id}_gradcam.jpg',
        mimetype='image/jpeg'
    )


@app.route('/api/gradcam/video/<job_id>/<int:frame_idx>', methods=['GET'])
def serve_gradcam_video_frame(job_id, frame_idx):
    """
    Serve a specific Grad-CAM frame from video analysis
    """
    gradcam_path = os.path.join(
        app.config['RESULTS_FOLDER'],
        'gradcam',
        job_id,
        f'gradcam_frame_{frame_idx}.jpg'
    )
    
    if not os.path.exists(gradcam_path):
        return jsonify({'error': 'Grad-CAM frame not found'}), 404
    
    return send_from_directory(
        os.path.join(app.config['RESULTS_FOLDER'], 'gradcam', job_id),
        f'gradcam_frame_{frame_idx}.jpg',
        mimetype='image/jpeg'
    )


@app.route('/api/c2pa/<job_id>', methods=['GET'])
def verify_c2pa(job_id):
    """
    C2PA content credential verification
    DIFFERENTIATOR: Combines AI detection with cryptographic authentication
    """
    if job_id not in jobs_db:
        return jsonify({'error': 'Job ID not found'}), 404
    
    job = jobs_db[job_id]
    
    try:
        verification = C2PAService.verify_credentials(job['filepath'])
        provenance = C2PAService.get_provenance_chain(job['filepath'])
        
        return jsonify({
            'job_id': job_id,
            'c2pa_verification': verification,
            'provenance_chain': provenance
        }), 200
    
    except Exception as e:
        return jsonify({
            'job_id': job_id,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    total_jobs = len(jobs_db)
    completed = sum(1 for j in jobs_db.values() if j['status'] == 'completed')
    processing = sum(1 for j in jobs_db.values() if j['status'] == 'processing')
    
    if completed > 0:
        results = [j['result'] for j in jobs_db.values() if j.get('result')]
        fake_count = sum(1 for r in results if r.get('prediction') == 'fake')
        real_count = sum(1 for r in results if r.get('prediction') == 'real')
    else:
        fake_count = real_count = 0
    
    return jsonify({
        'total_jobs': total_jobs,
        'completed': completed,
        'processing': processing,
        'detections': {
            'fake': fake_count,
            'real': real_count
        }
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  AuthentiScan Backend API Starting")
    print("=" * 50)
    print(f"Version: {app.config['API_VERSION']}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)