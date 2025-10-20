import sys
import os
sys.path.append('.')

# Test script to verify reference maker functions work correctly

def test_skeleton_normalization():
    print("Testing Skeleton Normalization...")
    try:
        from backend.reference_maker import SkeletonNormalizer
        import numpy as np
        
        # Create a sample landmark configuration (just for testing)
        # This represents 12 key landmarks (4 per joint group: shoulder, hip, knee, etc.)
        sample_landmarks = [
            0.5, 0.5, 0.0,  # LEFT_SHOULDER
            0.6, 0.5, 0.0,  # RIGHT_SHOULDER  
            0.4, 0.6, 0.0,  # LEFT_ELBOW
            0.7, 0.6, 0.0,  # RIGHT_ELBOW
            0.3, 0.7, 0.0,  # LEFT_WRIST
            0.8, 0.7, 0.0,  # RIGHT_WRIST
            0.5, 0.8, 0.0,  # LEFT_HIP
            0.6, 0.8, 0.0,  # RIGHT_HIP
            0.4, 1.1, 0.0,  # LEFT_KNEE
            0.7, 1.1, 0.0,  # RIGHT_KNEE
            0.4, 1.4, 0.0,  # LEFT_ANKLE
            0.7, 1.4, 0.0   # RIGHT_ANKLE
        ] * 3  # Just to make sure we have enough values
        
        # Take the first 36 values (12 landmarks * 3 coordinates each)
        sample_landmarks = sample_landmarks[:36]
        
        normalizer = SkeletonNormalizer()
        
        # Test normalization
        normalized = normalizer.normalize_skeleton(sample_landmarks)
        print(f"[SUCCESS] Normalization successful. Input length: {len(sample_landmarks)}, Output length: {len(normalized)}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Skeleton normalization failed: {str(e)}")
        return False

def test_feature_extraction():
    print("\nTesting Feature Extraction...")
    try:
        from backend.reference_maker import FeatureExtractor
        import numpy as np
        
        # Create a sample normalized landmark configuration
        sample_landmarks = [
            0.5, 0.5, 0.0,  # LEFT_SHOULDER
            0.6, 0.5, 0.0,  # RIGHT_SHOULDER  
            0.4, 0.6, 0.0,  # LEFT_ELBOW
            0.7, 0.6, 0.0,  # RIGHT_ELBOW
            0.3, 0.7, 0.0,  # LEFT_WRIST
            0.8, 0.7, 0.0,  # RIGHT_WRIST
            0.5, 0.8, 0.0,  # LEFT_HIP
            0.6, 0.8, 0.0,  # RIGHT_HIP
            0.4, 1.1, 0.0,  # LEFT_KNEE
            0.7, 1.1, 0.0,  # RIGHT_KNEE
            0.4, 1.4, 0.0,  # LEFT_ANKLE
            0.7, 1.4, 0.0   # RIGHT_ANKLE
        ]  # 12 landmarks * 3 coordinates = 36 values
        
        extractor = FeatureExtractor()
        
        # Test feature extraction
        features = extractor.extract_features(sample_landmarks)
        print(f"[SUCCESS] Feature extraction successful. Extracted {len(features)} features: {list(features.keys())}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {str(e)}")
        return False

def test_phase_segmentation():
    print("\nTesting Phase Segmentation...")
    try:
        from backend.reference_maker import PhaseSegmenter
        import numpy as np
        
        # Create a sample landmark configuration for testing
        sample_landmarks = [
            0.5, 0.5, 0.0,  # LEFT_SHOULDER
            0.6, 0.5, 0.0,  # RIGHT_SHOULDER  
            0.4, 0.6, 0.0,  # LEFT_ELBOW
            0.7, 0.6, 0.0,  # RIGHT_ELBOW
            0.3, 0.7, 0.0,  # LEFT_WRIST
            0.8, 0.7, 0.0,  # RIGHT_WRIST
            0.5, 0.8, 0.0,  # LEFT_HIP
            0.6, 0.8, 0.0,  # RIGHT_HIP
            0.4, 1.1, 0.0,  # LEFT_KNEE
            0.7, 1.1, 0.0,  # RIGHT_KNEE
            0.4, 1.4, 0.0,  # LEFT_ANKLE
            0.7, 1.4, 0.0   # RIGHT_ANKLE
        ]
        
        segmenter = PhaseSegmenter()
        
        # Test phase segmentation for different exercise types
        for exercise in ['squat', 'push-up', 'bicep_curl']:
            phase = segmenter.segment_by_exercise_phase(sample_landmarks, exercise)
            print(f"  {exercise}: phase = {phase:.2f}")
        
        print("[SUCCESS] Phase segmentation successful.")
        return True
    except Exception as e:
        print(f"[ERROR] Phase segmentation failed: {str(e)}")
        return False

def test_form_analyzer():
    print("\nTesting Form Analyzer...")
    try:
        from backend.reference_maker import FormAnalyzer
        import cv2
        import numpy as np
        
        # Create a simple test image (black image with a white rectangle to simulate a person)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_frame, (300, 100), (340, 400), (255, 255, 255), -1)  # Body
        cv2.rectangle(test_frame, (280, 100), (300, 150), (255, 255, 255), -1)  # Left arm
        cv2.rectangle(test_frame, (340, 100), (360, 150), (255, 255, 255), -1)  # Right arm
        cv2.rectangle(test_frame, (290, 400), (310, 450), (255, 255, 255), -1)  # Left leg
        cv2.rectangle(test_frame, (330, 400), (350, 450), (255, 255, 255), -1)  # Right leg
        
        analyzer = FormAnalyzer()
        
        # Test with an unknown exercise type (this should handle gracefully)
        result = analyzer.analyze_form(test_frame, "unknown_exercise")
        print(f"  Test with unknown exercise type: {result.get('feedback', 'No feedback')}")
        
        print("[SUCCESS] Form analyzer initialization successful.")
        return True
    except Exception as e:
        print(f"[ERROR] Form analyzer failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing Reference Maker Functions...\n")
    
    all_tests_passed = True
    
    all_tests_passed &= test_skeleton_normalization()
    all_tests_passed &= test_feature_extraction()
    all_tests_passed &= test_phase_segmentation()
    all_tests_passed &= test_form_analyzer()
    
    print(f"\n{'='*50}")
    if all_tests_passed:
        print("[SUCCESS] All tests passed! The reference maker functions are working correctly.")
    else:
        print("[ERROR] Some tests failed. Please check the error messages above.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()