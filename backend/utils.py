import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for the application"""
    directories = [
        'data',  # For storing datasets
        'models',  # For storing trained models
        'temp',  # For temporary files during processing
        'logs',  # For log files
        'static'  # For static files like images
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def get_config():
    """Get application configuration"""
    config = {
        'MODEL_PATH': os.getenv('MODEL_PATH', 'final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5'),
        'SCALER_PATH': os.getenv('SCALER_PATH', 'thesis_bidirectionallstm_scaler.pkl'),
        'LABEL_ENCODER_PATH': os.getenv('LABEL_ENCODER_PATH', 'thesis_bidirectionallstm_label_encoder.pkl'),
        'TEMP_DIR': os.getenv('TEMP_DIR', 'temp'),
        'MAX_FILE_SIZE': int(os.getenv('MAX_FILE_SIZE', 50 * 1024 * 1024)),  # 50MB
        'ALLOWED_EXTENSIONS': {'mp4', 'mov', 'avi', 'm4v', 'jpg', 'jpeg', 'png'},
        'WINDOW_SIZE': int(os.getenv('WINDOW_SIZE', 30)),  # Number of frames for model analysis
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO')
    }
    
    return config

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def validate_file_size(file_path, max_size):
    """Validate that file size does not exceed the limit"""
    if os.path.getsize(file_path) > max_size:
        return False
    return True

def get_timestamp():
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()