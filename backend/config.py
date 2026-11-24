"""
Configuration file for AuthentiScan backend
Contains all configurable parameters
"""
import os


def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ('1', 'true', 'yes', 'on')


class Config:
    # Flask configuration.
    # SECRET_KEY: must be overridden via env var for any non-local deployment.
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # DEBUG defaults to False. Only set FLASK_DEBUG=true for local development —
    # Flask's debug mode exposes an interactive debugger (with code execution)
    # to anyone who can trigger a 500 error, so it must never be on when the
    # server is reachable beyond localhost.
    DEBUG = _env_bool('FLASK_DEBUG', False)

    # HOST defaults to loopback-only. Set FLASK_HOST=0.0.0.0 explicitly (and only
    # with DEBUG=False) if the server needs to be reachable from other machines,
    # e.g. behind a reverse proxy.
    HOST = os.environ.get('FLASK_HOST', '127.0.0.1')
    PORT = int(os.environ.get('FLASK_PORT', 5000))

    # CORS: comma-separated list of allowed origins. Defaults to the local
    # Vite dev server only — do NOT default this to '*' in any deployed setting.
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:5173').split(',')

    # File upload configuration
    UPLOAD_FOLDER = 'uploads'
    RESULTS_FOLDER = 'results'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}

    # Secure deletion: uploaded files and their derived results (gradcam, etc.)
    # older than this are purged by the cleanup job in app.py.
    FILE_RETENTION_SECONDS = int(os.environ.get('FILE_RETENTION_SECONDS', 24 * 3600))

    # Model configuration
    CONFIDENCE_THRESHOLD = 0.5  # Threshold for fake classification
    FRAME_SAMPLE_RATE = 10  # Process every Nth frame
    
    # API configuration
    API_VERSION = 'v1.0.0'
    API_TITLE = 'AuthentiScan API'