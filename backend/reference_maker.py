import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import pickle
import json
from pathlib import Path
from datetime import datetime

@dataclass
class ReferenceModel:
    """Data class to store reference model information"""
    exercise_type: str
    feature_ranges: Dict[str, Tuple[float, float]]  # min, max for each feature
    feature_means: Dict[str, float]  # mean values per feature
    feature_stds: Dict[str, float]   # std values per feature
    phase_ranges: Dict[str, Tuple[Tuple[float, float], ...]]  # phase-specific ranges
    metadata: Dict = None

class SkeletonNormalizer:
    """Class to normalize skeleton poses for comparison"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, 
                                      enable_segmentation=False, min_detection_confidence=0.5)
        
        # Define relevant landmarks indices
        self.relevant_landmarks_indices = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]
    
    def calculate_angle(self, a: Tuple[float, float, float], b: Tuple[float, float, float], 
                       c: Tuple[float, float, float]) -> float:
        """Calculate the angle between three points"""
        if np.any(np.array([a, b, c]) == 0):
            return -1.0  # Placeholder for missing landmarks
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def calculate_distance(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        if np.any(np.array([a, b]) == 0):
            return -1.0  # Placeholder for missing landmarks
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)

    def preprocess_frame(self, frame: np.ndarray) -> List[float]:
        """Process frame with MediaPipe pose to extract landmarks"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        landmarks = []
        if results.pose_landmarks:
            for idx in self.relevant_landmarks_indices:
                landmark = results.pose_landmarks.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks

    def normalize_skeleton(self, landmarks: List[float]) -> List[float]:
        """
        Normalize skeleton by:
        1. Translating to origin (using hip midpoint as root)
        2. Scaling by body proxy (shoulder-to-hip distance)
        3. Rotating to canonical orientation
        """
        if len(landmarks) != len(self.relevant_landmarks_indices) * 3:
            return landmarks  # Return as-is if not enough landmarks

        # Reshape landmarks to (n_keypoints, 3)
        points = np.array(landmarks).reshape(-1, 3)

        # Calculate hip midpoint as root
        left_hip_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_HIP.value)
        right_hip_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        
        root_point = (points[left_hip_idx] + points[right_hip_idx]) / 2.0

        # Translate all points so root is at origin
        translated_points = points - root_point

        # Calculate scale factor (shoulder-to-hip distance)
        left_shoulder_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        
        left_shoulder = translated_points[left_shoulder_idx]
        left_hip = translated_points[left_hip_idx]
        
        scale_factor = np.linalg.norm(left_shoulder - left_hip)
        
        if scale_factor != 0:
            # Scale all points
            scaled_points = translated_points / scale_factor
        else:
            scaled_points = translated_points

        # Return flattened normalized landmarks
        return scaled_points.flatten().tolist()

class FeatureExtractor:
    """Class to extract robust features from normalized skeletons"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.relevant_landmarks_indices = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]

    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate the angle between three points"""
        if np.any(np.array([a, b, c]) == 0):
            return -1.0  # Placeholder for missing landmarks
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def calculate_distance(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        if np.any(np.array([a, b]) == 0):
            return -1.0  # Placeholder for missing landmarks
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)

    def calculate_y_distance(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """Calculate Y-coordinate distance between two points"""
        if np.any(np.array([a, b]) == 0):
            return -1.0  # Placeholder for missing landmarks
        return np.abs(a[1] - b[1])

    def extract_features(self, normalized_landmarks: List[float]) -> Dict[str, float]:
        """Extract robust features from normalized landmarks"""
        if len(normalized_landmarks) != len(self.relevant_landmarks_indices) * 3:
            return {}  # Return empty dict if not enough landmarks

        # Reshape landmarks to (n_keypoints, 3)
        points = np.array(normalized_landmarks).reshape(-1, 3)

        # Define landmark index mapping
        shoulder_l_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        shoulder_r_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        elbow_l_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_ELBOW.value)
        elbow_r_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        wrist_l_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_WRIST.value)
        wrist_r_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_WRIST.value)
        hip_l_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_HIP.value)
        hip_r_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        knee_l_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        knee_r_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
        ankle_l_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        ankle_r_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        # Get landmark coordinates
        shoulder_l = points[shoulder_l_idx][:2]  # Only x,y for 2D angles
        shoulder_r = points[shoulder_r_idx][:2]
        elbow_l = points[elbow_l_idx][:2]
        elbow_r = points[elbow_r_idx][:2]
        wrist_l = points[wrist_l_idx][:2]
        wrist_r = points[wrist_r_idx][:2]
        hip_l = points[hip_l_idx][:2]
        hip_r = points[hip_r_idx][:2]
        knee_l = points[knee_l_idx][:2]
        knee_r = points[knee_r_idx][:2]
        ankle_l = points[ankle_l_idx][:2]
        ankle_r = points[ankle_r_idx][:2]

        features = {}

        # Joint angles
        features['left_knee_angle'] = self.calculate_angle(hip_l, knee_l, ankle_l)
        features['right_knee_angle'] = self.calculate_angle(hip_r, knee_r, ankle_r)
        features['left_hip_angle'] = self.calculate_angle(shoulder_l, hip_l, knee_l)
        features['right_hip_angle'] = self.calculate_angle(shoulder_r, hip_r, knee_r)
        features['left_elbow_angle'] = self.calculate_angle(shoulder_l, elbow_l, wrist_l)
        features['right_elbow_angle'] = self.calculate_angle(shoulder_r, elbow_r, wrist_r)
        features['trunk_angle'] = self.calculate_angle(shoulder_l, hip_l, ankle_l)

        # Segment ratios (normalized distances)
        features['left_knee_toe_ratio'] = self.calculate_distance(knee_l, ankle_l) / self.calculate_distance(hip_l, ankle_l) if self.calculate_distance(hip_l, ankle_l) > 0 else 0
        features['right_knee_toe_ratio'] = self.calculate_distance(knee_r, ankle_r) / self.calculate_distance(hip_r, ankle_r) if self.calculate_distance(hip_r, ankle_r) > 0 else 0
        features['left_elbow_shoulder_offset'] = self.calculate_distance(elbow_l, shoulder_l) / self.calculate_distance(shoulder_l, hip_l) if self.calculate_distance(shoulder_l, hip_l) > 0 else 0
        features['right_elbow_shoulder_offset'] = self.calculate_distance(elbow_r, shoulder_r) / self.calculate_distance(shoulder_r, hip_r) if self.calculate_distance(shoulder_r, hip_r) > 0 else 0

        # Alignment metrics
        features['hip_shoulder_alignment'] = self.calculate_y_distance(hip_l, shoulder_l) + self.calculate_y_distance(hip_r, shoulder_r)
        features['ankle_shoulder_alignment'] = self.calculate_y_distance(ankle_l, shoulder_l) + self.calculate_y_distance(ankle_r, shoulder_r)

        # Frontal plane projection angle (FPPA) - knee valgus detection
        # Simplified: angle between hip-knee-ankle in frontal plane
        features['left_fppa'] = self.calculate_angle(hip_l, knee_l, ankle_l)
        features['right_fppa'] = self.calculate_angle(hip_r, knee_r, ankle_r)

        return features

class PhaseSegmenter:
    """Class to segment exercises by phase"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.relevant_landmarks_indices = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]

    def calculate_distance(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        if np.any(np.array([a, b]) == 0):
            return -1.0  # Placeholder for missing landmarks
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)

    def calculate_y_distance(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """Calculate Y-coordinate distance between two points"""
        if np.any(np.array([a, b]) == 0):
            return -1.0  # Placeholder for missing landmarks
        return np.abs(a[1] - b[1])

    def segment_by_exercise_phase(self, landmarks: List[float], exercise_type: str) -> float:
        """Map frame to exercise phase [0.0, 1.0]"""
        if len(landmarks) != len(self.relevant_landmarks_indices) * 3:
            return 0.5  # Default to middle phase if not enough landmarks

        # Reshape landmarks to (n_keypoints, 3)
        points = np.array(landmarks).reshape(-1, 3)

        # Define landmark index mapping
        shoulder_l_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        shoulder_r_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        hip_l_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_HIP.value)
        hip_r_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        knee_l_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        knee_r_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
        ankle_l_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        ankle_r_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        elbow_l_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_ELBOW.value)
        elbow_r_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        wrist_l_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.LEFT_WRIST.value)  # Added missing variable
        wrist_r_idx = self.relevant_landmarks_indices.index(self.mp_pose.PoseLandmark.RIGHT_WRIST.value)  # Added missing variable

        # Get landmark coordinates
        shoulder_l = points[shoulder_l_idx]
        hip_l = points[hip_l_idx]
        knee_l = points[knee_l_idx]
        ankle_l = points[ankle_l_idx]
        elbow_l = points[elbow_l_idx]
        wrist_l = points[wrist_l_idx]  # Added missing variable

        if exercise_type.lower() == 'squat':
            # Phase based on hip height (lower = deeper squat = phase 0.5)
            # Calculate hip height relative to ankle
            hip_height = hip_l[1]  # Y coordinate (vertical)
            ankle_height = ankle_l[1]
            
            # Normalize to phase [0, 1]
            # Assuming max hip height = standing (phase 0), min hip height = deepest squat (phase 0.5)
            # For now, use a simple range based on observed values
            # This would need calibration with actual exercise videos
            max_hip_height = 0.0  # Standing position
            min_hip_height = 0.7  # Deep squat position (normalized)
            
            # Calculate phase based on hip height (inverted: lower = deeper)
            if (max_hip_height - min_hip_height) != 0:
                phase = 1.0 - max(0, min(1, (hip_height - min_hip_height) / (max_hip_height - min_hip_height)))
            else:
                phase = 0.5
            return phase

        elif exercise_type.lower() == 'push-up':
            # Phase based on elbow angle or sternum height
            # Use elbow angle: straight = 0 (top), bent = 180 (bottom)
            shoulder_pos = points[shoulder_l_idx]
            elbow_pos = points[elbow_l_idx]
            wrist_pos = points[wrist_l_idx]
            
            # Calculate elbow angle
            from math import atan2
            angle1 = atan2(shoulder_pos[1] - elbow_pos[1], shoulder_pos[0] - elbow_pos[0])
            angle2 = atan2(wrist_pos[1] - elbow_pos[1], wrist_pos[0] - elbow_pos[0])
            elbow_angle = abs(angle1 - angle2)
            elbow_angle = elbow_angle * 180 / 3.14159  # Convert to degrees
            
            if elbow_angle > 180:
                elbow_angle = 360 - elbow_angle
            
            # Normalize phase based on elbow angle
            # Straight arm ~180° (top of pushup) -> phase 0
            # Bent arm ~0° (bottom of pushup) -> phase 0.5
            max_elbow_angle = 180.0
            min_elbow_angle = 90.0  # Minimum during pushup
            
            if (max_elbow_angle - min_elbow_angle) != 0:
                phase = max(0, min(1, (elbow_angle - min_elbow_angle) / (max_elbow_angle - min_elbow_angle)))
            else:
                phase = 0.5
            return phase

        elif exercise_type.lower() == 'bicep_curl':
            # Phase based on elbow angle: extended = 0° (bottom), flexed = 180° (top)
            shoulder_pos = points[shoulder_l_idx]
            elbow_pos = points[elbow_l_idx]
            wrist_pos = points[wrist_l_idx]
            
            # Calculate elbow angle
            from math import atan2
            angle1 = atan2(shoulder_pos[1] - elbow_pos[1], shoulder_pos[0] - elbow_pos[0])
            angle2 = atan2(wrist_pos[1] - elbow_pos[1], wrist_pos[0] - elbow_pos[0])
            elbow_angle = abs(angle1 - angle2)
            elbow_angle = elbow_angle * 180 / 3.14159  # Convert to degrees
            
            if elbow_angle > 180:
                elbow_angle = 360 - elbow_angle
            
            # Normalize phase based on elbow angle
            # Extended arm ~180° (bottom of curl) -> phase 0
            # Flexed arm ~0° (top of curl) -> phase 0.5
            max_elbow_angle = 180.0
            min_elbow_angle = 30.0  # Minimum during curl
            
            if (max_elbow_angle - min_elbow_angle) != 0:
                phase = max(0, min(1, (elbow_angle - min_elbow_angle) / (max_elbow_angle - min_elbow_angle)))
            else:
                phase = 0.5
            return phase

        else:  # Default to static pose (plank, etc.)
            return 0.5  # Middle phase for static exercises

class ReferenceBuilder:
    """Class to build reference models from video data"""
    
    def __init__(self):
        self.normalizer = SkeletonNormalizer()
        self.feature_extractor = FeatureExtractor()
        self.phase_segmenter = PhaseSegmenter()
        self.references = {}  # Store all reference models

    def load_video_data(self, video_path: str) -> List[np.ndarray]:
        """Load video frames from path"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames

    def build_reference_from_video(self, video_path: str, exercise_type: str, 
                                 n_bins: int = 10) -> ReferenceModel:
        """Build a reference model from a video of correct form"""
        
        print(f"Building reference model for {exercise_type} from {video_path}")
        
        # Load video frames
        frames = self.load_video_data(video_path)
        
        # Process each frame
        all_features = []
        all_phases = []
        
        for i, frame in enumerate(frames):
            if i % 5 == 0:  # Process every 5th frame to reduce computation
                # Extract landmarks
                landmarks = self.normalizer.preprocess_frame(frame)
                if len(landmarks) == 0:
                    continue
                
                # Normalize skeleton
                normalized_landmarks = self.normalizer.normalize_skeleton(landmarks)
                
                # Extract features
                features = self.feature_extractor.extract_features(normalized_landmarks)
                if not features:
                    continue
                
                # Segment by phase
                phase = self.phase_segmenter.segment_by_exercise_phase(landmarks, exercise_type)
                
                all_features.append(features)
                all_phases.append(phase)
        
        if not all_features:
            raise ValueError("No valid frames found in video")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(all_features)
        phases = np.array(all_phases)
        
        # Create phase bins
        phase_bins = np.linspace(0, 1, n_bins + 1)
        phase_indices = np.digitize(phases, phase_bins) - 1
        phase_indices = np.clip(phase_indices, 0, n_bins - 1)  # Ensure indices are valid
        
        # Calculate statistics per phase per feature
        feature_ranges = {}
        feature_means = {}
        feature_stds = {}
        phase_ranges = {}
        
        for feature_name in df.columns:
            feature_data = df[feature_name].values
            
            # Calculate overall statistics
            feature_means[feature_name] = float(np.nanmean(feature_data))
            feature_stds[feature_name] = float(np.nanstd(feature_data))
            
            # Calculate min/max ranges
            valid_data = feature_data[~np.isnan(feature_data)]
            if len(valid_data) > 0:
                feature_ranges[feature_name] = (float(np.nanmin(valid_data)), float(np.nanmax(valid_data)))
            else:
                feature_ranges[feature_name] = (0.0, 0.0)
            
            # Calculate phase-specific ranges
            phase_feature_ranges = []
            for phase_idx in range(n_bins):
                phase_mask = phase_indices == phase_idx
                phase_feature_data = feature_data[phase_mask]
                if len(phase_feature_data) > 0 and not np.all(np.isnan(phase_feature_data)):
                    phase_min = float(np.nanmin(phase_feature_data))
                    phase_max = float(np.nanmax(phase_feature_data))
                    phase_feature_ranges.append((phase_min, phase_max))
                else:
                    phase_feature_ranges.append((feature_means[feature_name] - feature_stds[feature_name],
                                                feature_means[feature_name] + feature_stds[feature_name]))
            
            phase_ranges[feature_name] = tuple(phase_feature_ranges)
        
        # Create reference model
        reference = ReferenceModel(
            exercise_type=exercise_type,
            feature_ranges=feature_ranges,
            feature_means=feature_means,
            feature_stds=feature_stds,
            phase_ranges=phase_ranges,
            metadata={
                'n_frames': len(all_features),
                'n_bins': n_bins,
                'built_at': datetime.now().isoformat(),
                'video_path': video_path
            }
        )
        
        # Store in references dictionary
        self.references[exercise_type] = reference
        
        print(f"Successfully built reference model for {exercise_type}")
        return reference

    def save_reference(self, reference: ReferenceModel, file_path: str):
        """Save reference model to file"""
        with open(file_path, 'wb') as f:
            pickle.dump(reference, f)
        print(f"Reference model saved to {file_path}")

    def load_reference(self, file_path: str) -> ReferenceModel:
        """Load reference model from file"""
        with open(file_path, 'rb') as f:
            reference = pickle.load(f)
        print(f"Reference model loaded from {file_path}")
        return reference

class FormAnalyzer:
    """Class to analyze form against reference models"""
    
    def __init__(self):
        self.normalizer = SkeletonNormalizer()
        self.feature_extractor = FeatureExtractor()
        self.phase_segmenter = PhaseSegmenter()
        self.references = {}

    def load_reference(self, exercise_type: str, file_path: str):
        """Load a reference model for a specific exercise type"""
        with open(file_path, 'rb') as f:
            reference = pickle.load(f)
        self.references[exercise_type] = reference

    def analyze_form(self, frame: np.ndarray, exercise_type: str) -> Dict:
        """Analyze form in a single frame against reference"""
        
        if exercise_type not in self.references:
            return {
                'error': f'No reference model found for exercise type: {exercise_type}',
                'feedback': 'Unknown exercise type'
            }
        
        reference = self.references[exercise_type]
        
        # Extract landmarks
        landmarks = self.normalizer.preprocess_frame(frame)
        if len(landmarks) == 0:
            return {
                'error': 'No landmarks detected in frame',
                'feedback': 'Could not detect body pose'
            }
        
        # Normalize skeleton
        normalized_landmarks = self.normalizer.normalize_skeleton(landmarks)
        
        # Extract features
        features = self.feature_extractor.extract_features(normalized_landmarks)
        if not features:
            return {
                'error': 'Could not extract features from landmarks',
                'feedback': 'Invalid pose detected'
            }
        
        # Segment by phase
        phase = self.phase_segmenter.segment_by_exercise_phase(landmarks, exercise_type)
        
        # Calculate phase bin
        n_bins = reference.metadata['n_bins']
        phase_bins = np.linspace(0, 1, n_bins + 1)
        phase_idx = min(int(phase * n_bins), n_bins - 1)
        
        # Calculate deviations from reference
        deviations = {}
        for feature_name, value in features.items():
            if feature_name in reference.feature_means:
                mean_val = reference.feature_means[feature_name]
                std_val = reference.feature_stds[feature_name]
                
                if std_val != 0:
                    z_score = abs(value - mean_val) / std_val
                    deviations[feature_name] = {
                        'value': value,
                        'reference_mean': mean_val,
                        'reference_std': std_val,
                        'z_score': z_score,
                        'is_outlier': z_score > 2.0  # More than 2 standard deviations
                    }
        
        # Identify significant errors
        error_feedback = []
        for feature_name, dev_info in deviations.items():
            if dev_info['is_outlier']:
                error_feedback.append({
                    'feature': feature_name,
                    'value': dev_info['value'],
                    'reference_mean': dev_info['reference_mean'],
                    'deviation': dev_info['z_score'],
                    'message': self._get_error_message(feature_name, dev_info['value'], dev_info['reference_mean'])
                })
        
        # Calculate overall form score
        total_z_scores = [dev['z_score'] for dev in deviations.values()]
        if total_z_scores:
            avg_z_score = np.mean(total_z_scores)
            # Convert to percentage score (lower z-score = better form)
            form_score = max(0, 100 - (avg_z_score * 10))  # Adjust multiplier as needed
        else:
            form_score = 100  # Perfect score if no data
        
        return {
            'exercise_type': exercise_type,
            'phase': phase,
            'form_score': form_score,
            'deviations': deviations,
            'detected_errors': error_feedback,
            'recommendations': self._generate_recommendations(error_feedback),
            'timestamp': datetime.now().isoformat()
        }

    def _get_error_message(self, feature_name: str, value: float, reference_mean: float) -> str:
        """Generate error message for a specific feature deviation"""
        
        # Common error messages based on features
        if 'knee_angle' in feature_name:
            if value < reference_mean:
                side = 'left' if 'left' in feature_name else 'right'
                return f"Too acute {side} knee angle - knee might be collapsing inward"
            else:
                side = 'left' if 'left' in feature_name else 'right'
                return f"Too obtuse {side} knee angle - knee might not be bending enough"
        
        elif 'hip_angle' in feature_name:
            if value < reference_mean:
                side = 'left' if 'left' in feature_name else 'right'
                return f"Too acute {side} hip angle - hip might not be bending properly"
            else:
                side = 'left' if 'left' in feature_name else 'right'
                return f"Too obtuse {side} hip angle - hip might be hyperextending"
        
        elif 'elbow_angle' in feature_name:
            if value < reference_mean:
                side = 'left' if 'left' in feature_name else 'right'
                return f"Too acute {side} elbow angle - arm might not be extending fully"
            else:
                side = 'left' if 'left' in feature_name else 'right'
                return f"Too obtuse {side} elbow angle - arm might be hyperextending"
        
        elif 'fppa' in feature_name:  # Frontal Plane Projection Angle (knee valgus)
            side = 'left' if 'left' in feature_name else 'right'
            return f"Possible {side} knee valgus (knee caving inward)"
        
        elif 'alignment' in feature_name:
            return f"Body alignment issue detected"
        
        else:
            return f"{feature_name.replace('_', ' ')} deviation from reference"
    
    def _generate_recommendations(self, error_feedback: List[Dict]) -> List[str]:
        """Generate recommendations based on detected errors"""
        
        recommendations = []
        for error in error_feedback:
            feature = error['feature']
            value = error['value']
            ref_mean = error['reference_mean']
            
            # Generate specific recommendations based on features
            if 'knee_angle' in feature:
                if value < ref_mean:
                    recommendations.append("Focus on keeping knees aligned over toes")
                else:
                    recommendations.append("Make sure to bend knees properly through the movement")
            
            elif 'hip_angle' in feature:
                if value < ref_mean:
                    recommendations.append("Engage your core and maintain proper hip hinge")
                else:
                    recommendations.append("Lower your hips more during the movement")
            
            elif 'elbow_angle' in feature:
                if value < ref_mean:
                    recommendations.append("Fully extend your arms at the appropriate points")
                else:
                    recommendations.append("Make sure to bend your elbows properly")
            
            elif 'alignment' in feature:
                recommendations.append("Focus on maintaining a straight line from head to heels")
        
        return recommendations

def build_references_from_dataset(dataset_path: str, output_dir: str):
    """Build reference models from a dataset of videos - one reference per exercise type"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    builder = ReferenceBuilder()
    
    # Group videos by exercise type
    exercise_videos = {}
    
    # Walk through the dataset directory to find video files
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.m4v', '.mkv')):
                file_path = os.path.join(root, file)
                
                # Try to determine exercise type from directory name or filename
                dir_name = os.path.basename(root).lower()
                if 'squat' in dir_name:
                    exercise_type = 'squat'
                elif 'pushup' in dir_name or 'push_up' in dir_name or 'push-up' in dir_name:
                    exercise_type = 'push-up'
                elif 'curl' in dir_name or 'bicep' in dir_name or 'hammer' in dir_name:
                    exercise_type = 'bicep_curl'
                elif 'press' in dir_name:
                    exercise_type = 'shoulder_press'
                elif 'plank' in dir_name:
                    exercise_type = 'plank'
                else:
                    # If can't determine from directory, try from filename
                    filename = file.lower()
                    if 'squat' in filename:
                        exercise_type = 'squat'
                    elif 'pushup' in filename or 'push_up' in filename or 'push-up' in filename:
                        exercise_type = 'push-up'
                    elif 'curl' in filename or 'bicep' in filename or 'hammer' in filename:
                        exercise_type = 'bicep_curl'
                    elif 'press' in filename:
                        exercise_type = 'shoulder_press'
                    elif 'plank' in filename:
                        exercise_type = 'plank'
                    else:
                        print(f"Could not determine exercise type for {file_path}, skipping...")
                        continue
                
                # Add to the exercise type group
                if exercise_type not in exercise_videos:
                    exercise_videos[exercise_type] = []
                exercise_videos[exercise_type].append(file_path)
    
    # Process each exercise type as a group
    for exercise_type, video_paths in exercise_videos.items():
        print(f"Building reference model for {exercise_type} using {len(video_paths)} videos...")
        
        # Process all videos for this exercise type to aggregate features
        all_features = []
        all_phases = []
        
        for video_path in video_paths:
            try:
                print(f"Processing {os.path.basename(video_path)}...")
                
                # Load video frames
                frames = builder.load_video_data(video_path)
                
                # Process each frame
                for i, frame in enumerate(frames):
                    if i % 5 == 0:  # Process every 5th frame to reduce computation
                        # Extract landmarks
                        landmarks = builder.normalizer.preprocess_frame(frame)
                        if len(landmarks) == 0:
                            continue
                        
                        # Normalize skeleton
                        normalized_landmarks = builder.normalizer.normalize_skeleton(landmarks)
                        
                        # Extract features
                        features = builder.feature_extractor.extract_features(normalized_landmarks)
                        if not features:
                            continue
                        
                        # Segment by phase
                        phase = builder.phase_segmenter.segment_by_exercise_phase(landmarks, exercise_type)
                        
                        all_features.append(features)
                        all_phases.append(phase)
                        
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                continue
        
        if not all_features:
            print(f"No valid frames found for {exercise_type}, skipping...")
            continue
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(all_features)
        phases = np.array(all_phases)
        
        # Create phase bins
        n_bins = 10  # Using 10 bins as default
        phase_bins = np.linspace(0, 1, n_bins + 1)
        phase_indices = np.digitize(phases, phase_bins) - 1
        phase_indices = np.clip(phase_indices, 0, n_bins - 1)  # Ensure indices are valid
        
        # Calculate statistics per phase per feature for this exercise type
        feature_ranges = {}
        feature_means = {}
        feature_stds = {}
        phase_ranges = {}
        
        for feature_name in df.columns:
            feature_data = df[feature_name].values
            
            # Calculate overall statistics for this exercise type
            feature_means[feature_name] = float(np.nanmean(feature_data))
            feature_stds[feature_name] = float(np.nanstd(feature_data))
            
            # Calculate min/max ranges
            valid_data = feature_data[~np.isnan(feature_data)]
            if len(valid_data) > 0:
                feature_ranges[feature_name] = (float(np.nanmin(valid_data)), float(np.nanmax(valid_data)))
            else:
                feature_ranges[feature_name] = (0.0, 0.0)
            
            # Calculate phase-specific ranges for this exercise type
            phase_feature_ranges = []
            for phase_idx in range(n_bins):
                phase_mask = phase_indices == phase_idx
                phase_feature_data = feature_data[phase_mask]
                if len(phase_feature_data) > 0 and not np.all(np.isnan(phase_feature_data)):
                    phase_min = float(np.nanmin(phase_feature_data))
                    phase_max = float(np.nanmax(phase_feature_data))
                    phase_feature_ranges.append((phase_min, phase_max))
                else:
                    # Use overall mean ± std if no data for this phase
                    phase_feature_ranges.append((feature_means[feature_name] - feature_stds[feature_name],
                                                feature_means[feature_name] + feature_stds[feature_name]))
            
            phase_ranges[feature_name] = tuple(phase_feature_ranges)
        
        # Create reference model for this exercise type
        reference = ReferenceModel(
            exercise_type=exercise_type,
            feature_ranges=feature_ranges,
            feature_means=feature_means,
            feature_stds=feature_stds,
            phase_ranges=phase_ranges,
            metadata={
                'n_frames': len(all_features),
                'n_bins': n_bins,
                'built_at': datetime.now().isoformat(),
                'video_paths': video_paths,
                'n_videos': len(video_paths)
            }
        )
        
        # Save the reference model for this exercise type
        output_path = os.path.join(output_dir, f"reference_{exercise_type.replace(' ', '_')}.pkl")
        builder.save_reference(reference, output_path)
        
        # Store in builder's references for potential use
        builder.references[exercise_type] = reference
        
        print(f"Successfully built and saved reference model for {exercise_type} using {len(video_paths)} videos")
    
    print(f"Reference building complete. Models saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    dataset_path = "data/final_kaggle_with_additional_video"
    output_dir = "models/references"
    
    # Build references from dataset
    build_references_from_dataset(dataset_path, output_dir)