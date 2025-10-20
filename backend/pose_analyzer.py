import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple
import logging

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define relevant landmarks indices
RELEVANT_LANDMARKS_INDICES = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value
]

class PoseAnalyzer:
    """Class to handle pose estimation, feature extraction, and form analysis"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def preprocess_frame(self, frame: np.ndarray) -> List[float]:
        """Process frame with MediaPipe pose to extract landmarks"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        landmarks = []
        if results.pose_landmarks:
            for idx in RELEVANT_LANDMARKS_INDICES:
                landmark = results.pose_landmarks.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks

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

    def extract_features(self, landmarks: List[float]) -> List[float]:
        """Extract feature vectors from pose landmarks for model input"""
        features = []
        if len(landmarks) == len(RELEVANT_LANDMARKS_INDICES) * 3:
            # Angles
            features.append(self.calculate_angle(landmarks[0:3], landmarks[6:9], landmarks[12:15]))  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
            features.append(self.calculate_angle(landmarks[3:6], landmarks[9:12], landmarks[15:18]))  # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
            features.append(self.calculate_angle(landmarks[18:21], landmarks[24:27], landmarks[30:33]))  # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
            features.append(self.calculate_angle(landmarks[21:24], landmarks[27:30], landmarks[33:36]))  # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
            features.append(self.calculate_angle(landmarks[0:3], landmarks[18:21], landmarks[24:27]))  # LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE
            features.append(self.calculate_angle(landmarks[3:6], landmarks[21:24], landmarks[27:30]))  # RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE

            # New angles
            features.append(self.calculate_angle(landmarks[18:21], landmarks[0:3], landmarks[6:9]))  # LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW
            features.append(self.calculate_angle(landmarks[21:24], landmarks[3:6], landmarks[9:12]))  # RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW

            # Distances
            distances = [
                self.calculate_distance(landmarks[0:3], landmarks[3:6]),  # LEFT_SHOULDER, RIGHT_SHOULDER
                self.calculate_distance(landmarks[18:21], landmarks[21:24]),  # LEFT_HIP, RIGHT_HIP
                self.calculate_distance(landmarks[18:21], landmarks[24:27]),  # LEFT_HIP, LEFT_KNEE
                self.calculate_distance(landmarks[21:24], landmarks[27:30]),  # RIGHT_HIP, RIGHT_KNEE
                self.calculate_distance(landmarks[0:3], landmarks[18:21]),  # LEFT_SHOULDER, LEFT_HIP
                self.calculate_distance(landmarks[3:6], landmarks[21:24]),  # RIGHT_SHOULDER, RIGHT_HIP
                self.calculate_distance(landmarks[6:9], landmarks[24:27]),  # LEFT_ELBOW, LEFT_KNEE
                self.calculate_distance(landmarks[9:12], landmarks[27:30]),  # RIGHT_ELBOW, RIGHT_KNEE
                self.calculate_distance(landmarks[12:15], landmarks[0:3]),  # LEFT_WRIST, LEFT_SHOULDER
                self.calculate_distance(landmarks[15:18], landmarks[3:6]),  # RIGHT_WRIST, RIGHT_SHOULDER
                self.calculate_distance(landmarks[12:15], landmarks[18:21]),  # LEFT_WRIST, LEFT_HIP
                self.calculate_distance(landmarks[15:18], landmarks[21:24])   # RIGHT_WRIST, RIGHT_HIP
            ]

            # Y-coordinate distances
            y_distances = [
                self.calculate_y_distance(landmarks[6:9], landmarks[0:3]),  # LEFT_ELBOW, LEFT_SHOULDER
                self.calculate_y_distance(landmarks[9:12], landmarks[3:6])   # RIGHT_ELBOW, RIGHT_SHOULDER
            ]

            # Normalization factor based on shoulder-hip or hip-knee distance
            normalization_factor = -1
            distances_to_check = [
                self.calculate_distance(landmarks[0:3], landmarks[18:21]),  # LEFT_SHOULDER, LEFT_HIP
                self.calculate_distance(landmarks[3:6], landmarks[21:24]),  # RIGHT_SHOULDER, RIGHT_HIP
                self.calculate_distance(landmarks[18:21], landmarks[24:27]),  # LEFT_HIP, LEFT_KNEE
                self.calculate_distance(landmarks[21:24], landmarks[27:30])   # RIGHT_HIP, RIGHT_KNEE
            ]

            for distance in distances_to_check:
                if distance > 0:
                    normalization_factor = distance
                    break
            
            if normalization_factor == -1:
                normalization_factor = 0.5  # Fallback normalization factor
            
            # Normalize distances
            normalized_distances = [d / normalization_factor if d != -1.0 else d for d in distances]
            normalized_y_distances = [d / normalization_factor if d != -1.0 else d for d in y_distances]

            # Combine features
            features.extend(normalized_distances)
            features.extend(normalized_y_distances)

        else:
            logging.warning(f"Insufficient landmarks: expected {len(RELEVANT_LANDMARKS_INDICES)}, got {len(landmarks)//3}")
            features = [-1.0] * 22  # Placeholder for missing landmarks
        return features

    def analyze_exercise_form(self, landmarks: List[float], exercise_type: str) -> Tuple[List[dict], List[str], float]:
        """Analyze exercise form for common errors"""
        feedback_items = []
        recommendations = []
        
        if exercise_type == 'squat':
            # Check for common squat errors
            landmarks_dict = {i: (landmarks[i*3], landmarks[i*3+1], landmarks[i*3+2]) 
                             for i in range(len(landmarks)//3)}
            
            # Knee alignment check (knees going past toes)
            left_knee = landmarks[24:27]  # LEFT_KNEE
            left_ankle = landmarks[30:33]  # LEFT_ANKLE
            left_toe_angle = self.calculate_angle(left_knee, left_ankle, (landmarks[30], landmarks[31], landmarks[32]))
            
            if left_toe_angle > 160:
                feedback_items.append({
                    "error_type": "knee_misalignment",
                    "confidence": 0.85,
                    "message": "Knees going beyond toes during squat",
                    "severity": "high"
                })
                recommendations.append("Keep knees behind toes during the movement")
            
            # Hip alignment check (hip drop)
            left_hip = landmarks[18:21]  # LEFT_HIP
            right_hip = landmarks[21:24]  # RIGHT_HIP
            hip_distance = self.calculate_y_distance(left_hip, right_hip)
            
            if hip_distance > 0.1:
                feedback_items.append({
                    "error_type": "hip_alignment",
                    "confidence": 0.75,
                    "message": "Hip misalignment detected",
                    "severity": "medium"
                })
                recommendations.append("Maintain even hip level throughout the movement")
        
        elif exercise_type == 'push-up':
            # Check for common push-up errors
            left_shoulder = landmarks[0:3]  # LEFT_SHOULDER
            right_shoulder = landmarks[3:6]  # RIGHT_SHOULDER
            left_hip = landmarks[18:21]  # LEFT_HIP
            right_hip = landmarks[21:24]  # RIGHT_HIP
            left_ankle = landmarks[30:33]  # LEFT_ANKLE
            right_ankle = landmarks[33:36]  # RIGHT_ANKLE
            
            # Check for hip drop
            hip_shoulder_distance = self.calculate_y_distance(left_hip, left_shoulder)
            if hip_shoulder_distance > 0.1:
                feedback_items.append({
                    "error_type": "hip_drop",
                    "confidence": 0.80,
                    "message": "Hip drop detected during push-up",
                    "severity": "high"
                })
                recommendations.append("Keep your body in a straight line from head to heels")
        
        elif exercise_type == 'bicep_curl':
            # Check for common bicep curl errors
            left_shoulder = landmarks[0:3]  # LEFT_SHOULDER
            left_elbow = landmarks[6:9]  # LEFT_ELBOW
            left_wrist = landmarks[12:15]  # LEFT_WRIST
            
            # Check for elbow movement (shoulder involvement)
            elbow_shoulder_distance = self.calculate_y_distance(left_elbow, left_shoulder)
            if elbow_shoulder_distance > 0.05:
                feedback_items.append({
                    "error_type": "elbow_movement",
                    "confidence": 0.70,
                    "message": "Elbow moving during bicep curl",
                    "severity": "medium"
                })
                recommendations.append("Keep your upper arm still during the curl")
        
        elif exercise_type == 'shoulder_press':
            # Check for common shoulder press errors
            left_shoulder = landmarks[0:3]  # LEFT_SHOULDER
            left_elbow = landmarks[6:9]  # LEFT_ELBOW
            left_wrist = landmarks[12:15]  # LEFT_WRIST
            
            # Check for bar path
            shoulder_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            if shoulder_elbow_angle < 90:
                feedback_items.append({
                    "error_type": "bar_path",
                    "confidence": 0.75,
                    "message": "Incorrect bar path during shoulder press",
                    "severity": "medium"
                })
                recommendations.append("Keep the bar in line with your shoulders")
        
        # Calculate overall score based on number of errors
        error_count = len(feedback_items)
        max_errors = 5  # arbitrary max for scoring
        overall_score = max(0, 100 - (error_count * 20))
        
        return feedback_items, recommendations, overall_score

    def predict_exercise_from_landmarks(self, landmarks: List[float]) -> str:
        """
        Simplified exercise prediction based on landmark positions
        In a full implementation, this would use a trained model
        """
        # Get key landmark positions
        left_shoulder = landmarks[0:3]
        right_shoulder = landmarks[3:6]
        left_hip = landmarks[18:21]
        right_hip = landmarks[21:24]
        left_knee = landmarks[24:27]
        right_knee = landmarks[27:30]
        left_ankle = landmarks[30:33]
        right_ankle = landmarks[33:36]
        left_elbow = landmarks[6:9]
        right_elbow = landmarks[9:12]
        left_wrist = landmarks[12:15]
        right_wrist = landmarks[15:18]
        
        # Calculate distances and angles between key points
        shoulder_distance = self.calculate_distance(left_shoulder, right_shoulder)
        hip_distance = self.calculate_distance(left_hip, right_hip)
        
        # Hip-knee-shin angle (for squats and similar movements)
        left_hip_knee_ankle_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_hip_knee_ankle_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        # Shoulder-elbow-wrist angle (for curls and similar movements)
        left_shoulder_elbow_wrist_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_shoulder_elbow_wrist_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Determine exercise based on body positions
        if left_hip_knee_ankle_angle < 120 and right_hip_knee_ankle_angle < 120:
            # Likely a squat if knees are bent
            return "squat"
        elif left_shoulder_elbow_wrist_angle < 120 or right_shoulder_elbow_wrist_angle < 120:
            # Likely a bicep curl if arm is bent
            return "bicep_curl"
        elif (left_shoulder[1] > left_elbow[1] and right_shoulder[1] > right_elbow[1] and 
              left_elbow[1] > left_wrist[1] and right_elbow[1] > right_wrist[1]):
            # Likely push-up if arms are extended down
            return "push-up"
        else:
            # Default to unknown if no clear pattern
            return "unknown"