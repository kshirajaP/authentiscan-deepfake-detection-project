"""
Security Middleware
Production-grade security features that differentiate AuthentiScan
"""
from functools import wraps
from flask import request, jsonify
import time
from collections import defaultdict
import hashlib

class SecurityMiddleware:
    """
    Implements production-grade security features:
    - Rate limiting
    - Request validation
    - Security headers
    - File content verification
    """
    
    def __init__(self):
        # Rate limiting storage (IP -> [timestamps])
        self.rate_limit_storage = defaultdict(list)
        
        # Rate limits
        self.UPLOAD_LIMIT = 10  # per hour
        self.DETECT_LIMIT = 20  # per hour
        
    def rate_limit(self, limit_type='default'):
        """
        Rate limiting decorator
        Prevents abuse and DoS attacks
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Get client IP
                client_ip = request.remote_addr
                current_time = time.time()
                
                # Clean old entries (older than 1 hour)
                self.rate_limit_storage[client_ip] = [
                    t for t in self.rate_limit_storage[client_ip]
                    if current_time - t < 3600
                ]
                
                # Check limit
                limit = self.UPLOAD_LIMIT if limit_type == 'upload' else self.DETECT_LIMIT
                if len(self.rate_limit_storage[client_ip]) >= limit:
                    return jsonify({
                        'error': f'Rate limit exceeded. Max {limit} requests per hour.',
                        'retry_after': 3600
                    }), 429
                
                # Add current request
                self.rate_limit_storage[client_ip].append(current_time)
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    @staticmethod
    def verify_file_content(file_path):
        """
        Deep file content verification
        Checks actual file content, not just extension
        """
        try:
            import magic
            mime = magic.from_file(file_path, mime=True)
            
            allowed_mimes = {
                'video/mp4', 'video/x-msvideo', 'video/quicktime',
                'image/jpeg', 'image/png'
            }
            
            if mime not in allowed_mimes:
                return False, f"Invalid file type detected: {mime}"
            
            return True, mime
        except Exception as e:
            return False, f"File verification failed: {str(e)}"
    
    @staticmethod
    def add_security_headers(response):
        """
        Add security headers to response
        Prevents common web vulnerabilities
        """
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'
        
        # Prevent MIME sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # Enable XSS filter
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Referrer policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Content Security Policy
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        
        return response
    
    @staticmethod
    def sanitize_filename(filename):
        """
        Sanitize filename to prevent directory traversal
        """
        import re
        # Remove any path separators
        filename = filename.replace('/', '').replace('\\', '')
        # Remove dangerous characters
        filename = re.sub(r'[^\w\s.-]', '', filename)
        return filename
    
    @staticmethod
    def calculate_file_hash(file_path):
        """
        Calculate file hash for integrity verification
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


# Initialize global security middleware
security = SecurityMiddleware()