import os
import cv2
import numpy as np
import tempfile
from typing import List, Tuple, Dict, Any
from datetime import datetime
import time
import logging

from pose_analyzer import PoseAnalyzer
from utils import get_config, allowed_file, validate_file_size, get_timestamp
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from reference_maker import FormAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FormAnalysisService:
    """Service class to handle form analysis business logic"""
    
    def __init__(self):
        self.config = get_config()
        self.pose_analyzer = PoseAnalyzer()
        self.lstm_model = None
        self.scaler = None
        self.label_encoder = None
        self.exercise_classes = []
        
        # Initialize new form analyzer with reference system
        self.form_analyzer = FormAnalyzer()
        
        # Load models
        self.load_models()
        
        # Load reference models
        self.load_reference_models()
    
    def load_models(self):
        """Load trained models for exercise classification and form analysis"""
        try:
            model_path = self.config['MODEL_PATH']
            scaler_path = self.config['SCALER_PATH']
            label_encoder_path = self.config['LABEL_ENCODER_PATH']
            
            # Check if model files exist
            if os.path.exists(model_path):
                self.lstm_model = load_model(model_path)
                logger.info(f"Loaded LSTM model from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")
                
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
            else:
                logger.warning(f"Scaler file not found: {scaler_path}")
                
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
                self.exercise_classes = self.label_encoder.classes_
                logger.info(f"Loaded label encoder from {label_encoder_path}")
            else:
                logger.warning(f"Label encoder file not found: {label_encoder_path}")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def load_reference_models(self):
        """Load reference models for form comparison"""
        # Load reference models that were created using the reference_maker
        reference_dir = "models/references"
        if os.path.exists(reference_dir):
            for file_name in os.listdir(reference_dir):
                if file_name.endswith('.pkl') and file_name.startswith('reference_'):
                    # Extract exercise type from filename
                    # Format: reference_exerciseType.pkl or reference_exercise-type.pkl
                    # Remove 'reference_' prefix and '.pkl' suffix
                    name_without_prefix = file_name[10:-4]  # Remove 'reference_' (10 chars) and '.pkl' (4 chars)
                    
                    # Handle special cases like push-up which has hyphen
                    if name_without_prefix == 'push-up':
                        exercise_type = 'push-up'
                    elif name_without_prefix == 'bicep_curl':
                        exercise_type = 'bicep_curl'
                    elif name_without_prefix == 'shoulder_press':
                        exercise_type = 'shoulder_press'
                    elif name_without_prefix == 'squat':
                        exercise_type = 'squat'
                    else:
                        # For simple cases, just use the name as-is
                        exercise_type = name_without_prefix
                    
                    file_path = os.path.join(reference_dir, file_name)
                    try:
                        self.form_analyzer.load_reference(exercise_type, file_path)
                        logger.info(f"Loaded reference model for {exercise_type}")
                    except Exception as e:
                        logger.error(f"Error loading reference {file_path}: {str(e)}")
    
    def analyze_video_form(self, video_file_path: str) -> Dict[str, Any]:
        """Analyze form from video file"""
        start_time = time.time()
        
        try:
            cap = cv2.VideoCapture(video_file_path)
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Process frames to detect exercise and analyze form
            frame_count = 0
            landmarks_window = []
            window_size = self.config['WINDOW_SIZE']
            current_prediction = "unknown"
            all_form_feedbacks = []
            
            # Process all frames to get comprehensive analysis
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # If we have reference models, use the new analysis method
                if self.form_analyzer.references:
                    # Analyze form against reference models
                    for exercise_type in self.form_analyzer.references.keys():
                        form_analysis = self.form_analyzer.analyze_form(frame, exercise_type)
                        
                        # If we find a good match based on form score or confidence
                        if form_analysis.get('form_score', 0) > 70:  # Threshold for considering this exercise
                            current_prediction = exercise_type
                            all_form_feedbacks.append(form_analysis)
                            break  # Break to avoid analyzing for other exercise types
                        elif not all_form_feedbacks:
                            all_form_feedbacks.append(form_analysis)
                
                # Fallback to old method if no reference models
                else:
                    landmarks = self.pose_analyzer.preprocess_frame(frame)
                    if len(landmarks) == len(self.pose_analyzer.RELEVANT_LANDMARKS_INDICES) * 3:
                        features = self.pose_analyzer.extract_features(landmarks)
                        if len(features) == 22:
                            landmarks_window.append(features)
                            frame_count += 1
                            
                            # Process window when we have enough frames
                            if len(landmarks_window) == window_size:
                                # Use the model to predict exercise type
                                if self.lstm_model is not None and self.scaler is not None:
                                    landmarks_window_np = np.array(landmarks_window).flatten().reshape(1, -1)
                                    scaled_landmarks_window = self.scaler.transform(landmarks_window_np)
                                    scaled_landmarks_window = scaled_landmarks_window.reshape(1, window_size, 22)

                                    prediction = self.lstm_model.predict(scaled_landmarks_window)
                                    predicted_class = np.argmax(prediction, axis=1)[0]
                                    
                                    if predicted_class < len(self.exercise_classes):
                                        current_prediction = self.exercise_classes[predicted_class]
                                
                                # Clear window
                                landmarks_window = []
            
            cap.release()
            
            # Prepare response based on analysis method used
            if all_form_feedbacks:
                # Use the best scoring analysis
                best_feedback = max(all_form_feedbacks, key=lambda x: x.get('form_score', 0))
                
                # Convert to compatible format
                detected_errors = []
                for error in best_feedback.get('detected_errors', []):
                    detected_errors.append({
                        "error_type": error.get('feature', 'unknown'),
                        "confidence": min(1.0, error.get('deviation', 1.0) / 10.0),  # Normalize deviation to confidence
                        "message": error.get('message', 'Form error detected'),
                        "severity": "high" if error.get('deviation', 0) > 3.0 else "medium" if error.get('deviation', 0) > 2.0 else "low"
                    })
                
                form_feedback = {
                    "timestamp": get_timestamp(),
                    "exercise_type": best_feedback.get('exercise_type', 'unknown'),
                    "detected_errors": detected_errors,
                    "overall_score": best_feedback.get('form_score', 0.0),
                    "recommendations": best_feedback.get('recommendations', [])
                }
            else:
                # Fallback to old method if no landmarks detected
                form_feedback = {
                    "timestamp": get_timestamp(),
                    "exercise_type": current_prediction,
                    "detected_errors": [],
                    "overall_score": 0.0,
                    "recommendations": ["Could not detect body landmarks. Please ensure you are clearly visible in the frame."]
                }
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "form_feedback": form_feedback,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_video_form: {str(e)}")
            raise e
    
    def analyze_image_form(self, image_file_path: str) -> Dict[str, Any]:
        """Analyze form from single image"""
        start_time = time.time()
        
        try:
            # Read image using OpenCV
            frame = cv2.imread(image_file_path)
            
            if frame is None:
                raise ValueError("Could not decode image file")
            
            # If we have reference models, use the new analysis method
            if self.form_analyzer.references:
                # Analyze form against all reference models and pick the best match
                best_analysis = None
                best_score = 0
                
                for exercise_type in self.form_analyzer.references.keys():
                    form_analysis = self.form_analyzer.analyze_form(frame, exercise_type)
                    score = form_analysis.get('form_score', 0)
                    
                    if score > best_score:
                        best_score = score
                        best_analysis = form_analysis
                
                # If we have a good match
                if best_analysis and best_score > 30:  # Threshold for considering valid
                    # Convert to compatible format
                    detected_errors = []
                    for error in best_analysis.get('detected_errors', []):
                        detected_errors.append({
                            "error_type": error.get('feature', 'unknown'),
                            "confidence": min(1.0, error.get('deviation', 1.0) / 10.0),  # Normalize deviation to confidence
                            "message": error.get('message', 'Form error detected'),
                            "severity": "high" if error.get('deviation', 0) > 3.0 else "medium" if error.get('deviation', 0) > 2.0 else "low"
                        })
                    
                    form_feedback = {
                        "timestamp": get_timestamp(),
                        "exercise_type": best_analysis.get('exercise_type', 'unknown'),
                        "detected_errors": detected_errors,
                        "overall_score": best_analysis.get('form_score', 0.0),
                        "recommendations": best_analysis.get('recommendations', [])
                    }
                else:
                    # If no good match found, try to predict exercise type
                    landmarks = self.pose_analyzer.preprocess_frame(frame)
                    predicted_exercise = self.pose_analyzer.predict_exercise_from_landmarks(landmarks) if len(landmarks) == len(self.pose_analyzer.RELEVANT_LANDMARKS_INDICES) * 3 else "unknown"
                    
                    form_feedback = {
                        "timestamp": get_timestamp(),
                        "exercise_type": predicted_exercise,
                        "detected_errors": [],
                        "overall_score": 0.0,
                        "recommendations": ["Could not match form to reference models. Please try a different exercise or check lighting conditions."]
                    }
            
            else:
                # Fallback to old method if no reference models
                landmarks = self.pose_analyzer.preprocess_frame(frame)
                
                # Predict exercise type using landmarks
                if len(landmarks) == len(self.pose_analyzer.RELEVANT_LANDMARKS_INDICES) * 3:
                    # Simplified exercise classification based on landmark positions
                    current_prediction = self.pose_analyzer.predict_exercise_from_landmarks(landmarks)
                    
                    detected_errors, recommendations, overall_score = self.pose_analyzer.analyze_exercise_form(
                        landmarks, current_prediction
                    )
                    
                    form_feedback = {
                        "timestamp": get_timestamp(),
                        "exercise_type": current_prediction,
                        "detected_errors": detected_errors,
                        "overall_score": overall_score,
                        "recommendations": recommendations
                    }
                else:
                    form_feedback = {
                        "timestamp": get_timestamp(),
                        "exercise_type": "unknown",
                        "detected_errors": [],
                        "overall_score": 0.0,
                        "recommendations": ["Could not detect body landmarks. Please ensure you are clearly visible in the frame."]
                    }
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "form_feedback": form_feedback,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_image_form: {str(e)}")
            raise e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models"""
        return {
            "model_loaded": self.lstm_model is not None,
            "scaler_loaded": self.scaler is not None,
            "label_encoder_loaded": self.label_encoder is not None,
            "reference_models_loaded": len(self.form_analyzer.references) > 0,
            "reference_exercises": list(self.form_analyzer.references.keys()),
            "exercise_classes": list(self.exercise_classes) if self.exercise_classes is not None else [],
            "timestamp": get_timestamp()
        }
    
    def validate_file_upload(self, file_path: str) -> bool:
        """Validate uploaded file"""
        file_extension = file_path.split('.')[-1].lower()
        
        # Check file extension
        if not allowed_file(file_path, self.config['ALLOWED_EXTENSIONS']):
            return False
        
        # Check file size
        if not validate_file_size(file_path, self.config['MAX_FILE_SIZE']):
            return False
        
        return True