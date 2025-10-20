import sys
import os
import cv2
import numpy as np
from datetime import datetime

# Add the backend path to import the reference maker
sys.path.append('.')

def test_demo_video_analysis():
    """Test the reference-based analysis on the demo_2.mp4 file"""
    print("="*60)
    print("TESTING REFERENCE-BASED ANALYSIS ON demo_2.mp4")
    print("="*60)
    
    # Import the reference maker
    from backend.reference_maker import FormAnalyzer
    
    # Initialize the form analyzer
    analyzer = FormAnalyzer()
    
    # Load reference models
    reference_dir = "backend/models/references"
    if os.path.exists(reference_dir):
        for file_name in os.listdir(reference_dir):
            if file_name.endswith('.pkl') and file_name.startswith('reference_'):
                # Extract exercise type from filename
                parts = file_name.replace('.pkl', '').split('_')
                if len(parts) >= 2:
                    if parts[1] == 'push-up':
                        exercise_type = 'push-up'
                    else:
                        exercise_type = parts[1]  # e.g., 'squat', 'bicep_curl', 'shoulder_press'
                    file_path = os.path.join(reference_dir, file_name)
                    try:
                        # Load the reference model directly with pickle to avoid module path issues
                        import pickle
                        import sys
                        import backend.reference_maker
                        # Temporarily map the old module name to the new location
                        sys.modules['reference_maker'] = backend.reference_maker
                        with open(file_path, 'rb') as f:
                            reference = pickle.load(f)
                        analyzer.references[exercise_type] = reference
                        print(f"[SUCCESS] Loaded reference model for {exercise_type}")
                    except Exception as e:
                        print(f"[ERROR] Error loading reference {file_path}: {str(e)}")
    else:
        print("Reference models directory not found!")
        return
    
    # Check if demo video exists
    demo_video_path = "demo_2.mp4"
    if not os.path.exists(demo_video_path):
        print(f"Demo video not found: {demo_video_path}")
        return
    
    print(f"Found demo video: {demo_video_path}")
    
    # Open the demo video
    cap = cv2.VideoCapture(demo_video_path)
    if not cap.isOpened():
        print("Could not open demo video")
        return
    
    # Process the video frame by frame
    print("\nProcessing video frames...")
    frame_count = 0
    all_analysis_results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:  # Process every 10th frame to speed up
            print(f"Processing frame {frame_count}...")
            
            # Analyze form against all available exercise types
            best_analysis = None
            best_score = 0
            
            for exercise_type in analyzer.references.keys():
                form_analysis = analyzer.analyze_form(frame, exercise_type)
                
                if form_analysis.get('form_score', 0) > best_score:
                    best_score = form_analysis.get('form_score', 0)
                    best_analysis = form_analysis
                    best_analysis['frame_number'] = frame_count
            
            if best_analysis:
                all_analysis_results.append(best_analysis)
                print(f"  Frame {frame_count}: Best match - {best_analysis['exercise_type']}, Score: {best_analysis['form_score']:.2f}")
                
                # Print detected errors if any
                if best_analysis.get('detected_errors'):
                    print(f"  Detected errors ({len(best_analysis['detected_errors'])}):")
                    for error in best_analysis['detected_errors'][:2]:  # Show top 2 errors
                        print(f"    - {error['message']}")
                else:
                    print(f"  No significant form errors detected")
    
    cap.release()
    
    if not all_analysis_results:
        print("No frames were successfully analyzed")
        return
    
    print(f"\nProcessed {len(all_analysis_results)} frames successfully")
    
    # Analyze the results
    exercise_types_found = {}
    scores_by_exercise = {}
    
    for result in all_analysis_results:
        ex_type = result['exercise_type']
        score = result['form_score']
        
        if ex_type not in exercise_types_found:
            exercise_types_found[ex_type] = 0
            scores_by_exercise[ex_type] = []
        
        exercise_types_found[ex_type] += 1
        scores_by_exercise[ex_type].append(score)
    
    print("\n" + "-"*40)
    print("ANALYSIS SUMMARY")
    print("-"*40)
    
    for ex_type, count in exercise_types_found.items():
        avg_score = np.mean(scores_by_exercise[ex_type])
        print(f"{ex_type.upper()}:")
        print(f"  Frames analyzed: {count}")
        print(f"  Average form score: {avg_score:.2f}/100")
        print(f"  Range: {min(scores_by_exercise[ex_type]):.2f} - {max(scores_by_exercise[ex_type]):.2f}")
    
    # Show sample analysis results
    if all_analysis_results:
        print(f"\nSAMPLE ANALYSIS FROM FRAME {all_analysis_results[0]['frame_number']}:")
        print("-" * 40)
        sample = all_analysis_results[0]
        print(f"Exercise: {sample['exercise_type']}")
        print(f"Score: {sample['form_score']:.2f}/100")
        print(f"Phase: {sample['phase']:.2f}")
        print(f"Timestamp: {sample['timestamp']}")
        
        if sample['detected_errors']:
            print("Detected Errors:")
            for i, error in enumerate(sample['detected_errors'][:3]):  # Top 3 errors
                print(f"  {i+1}. {error['message']}")
                print(f"     Feature: {error['feature']}")
                print(f"     Deviation: {error['deviation']:.2f}")
        else:
            print("No form errors detected in this frame")
    
    print("\n" + "="*60)
    print("REFERENCE-BASED ANALYSIS VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_demo_video_analysis()