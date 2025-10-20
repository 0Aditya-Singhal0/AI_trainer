import streamlit as st
import requests
import tempfile
import os
from typing import Dict, Any

# Set page configuration
st.set_page_config(
    page_title="Genyx Fitness AI Trainer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def analyze_form_with_user_exercise(video_file, exercise_type: str) -> Dict[str, Any]:
    """
    Call the backend API endpoint /analyze-form-user-specified to analyze form
    """
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as temp_file:
        temp_file.write(video_file.read())
        temp_file_path = temp_file.name
    
    try:
        # Prepare the files and data for the request
        with open(temp_file_path, 'rb') as f:
            files = {'file': (video_file.name, f, video_file.type)}
            data = {'exercise_type': exercise_type}
            
            # Make the API call to the backend
            response = requests.post(
                "http://localhost:8000/analyze-form-user-specified",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Please ensure the FastAPI server is running.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def display_analysis_results(results: Dict[str, Any]):
    """
    Display the form analysis results in a user-friendly format
    """
    if not results:
        st.error("No results to display.")
        return
    
    form_feedback = results.get('form_feedback', {})
    
    # Display exercise type and timestamp
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Exercise: {form_feedback.get('exercise_type', 'Unknown').title()}")
    with col2:
        st.subheader(f"Score: {form_feedback.get('overall_score', 0):.1f}/100")
    
    # Display analysis timestamp
    st.caption(f"Analysis timestamp: {form_feedback.get('timestamp', 'N/A')}")
    
    st.markdown("---")
    
    # Display detected errors in an expander
    detected_errors = form_feedback.get('detected_errors', [])
    if detected_errors:
        with st.expander(f"⚠️ Detected Form Issues ({len(detected_errors)})", expanded=True):
            for i, error in enumerate(detected_errors):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{error.get('message', 'No description')}**")
                        st.write(f"Error type: `{error.get('error_type', 'Unknown')}`")
                    with col2:
                        confidence = error.get('confidence', 0)
                        severity = error.get('severity', 'unknown')
                        
                        # Display confidence as percentage
                        st.write(f"Confidence: {confidence:.1%}")
                        
                        # Display severity with color coding
                        if severity.lower() == 'high':
                            st.markdown(f"**:red[Severity: {severity.title()}]**")
                        elif severity.lower() == 'medium':
                            st.markdown(f"**:orange[Severity: {severity.title()}]**")
                        else:
                            st.markdown(f"**:blue[Severity: {severity.title()}]**")
                    
                    st.markdown("---")
    else:
        # Display success message if no errors detected
        st.success("✅ Great form! No significant errors detected.")
    
    # Display recommendations
    recommendations = form_feedback.get('recommendations', [])
    if recommendations:
        with st.expander("💡 Recommendations", expanded=True):
            for i, rec in enumerate(recommendations):
                st.write(f"{i+1}. {rec}")
    
    # Display processing time
    processing_time = results.get('processing_time', 0)
    st.caption(f"Processing time: {processing_time:.2f} seconds")

def main():
    st.title("🏋️ Genyx Fitness AI Trainer")
    st.subheader("Advanced Exercise Form Analysis")
    
    st.markdown("""
    Upload your exercise video and select the exercise type to get detailed form analysis 
    with specific feedback and recommendations for improvement.
    """)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Exercise selection
        exercise_type = st.selectbox(
            "Select Exercise Type",
            options=[
                "squat",
                "push-up", 
                "bicep_curl",
                "shoulder_press"
            ],
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Choose the exercise you performed in the video"
        )
        
        # Video upload
        video_file = st.file_uploader(
            "Upload Exercise Video",
            type=['mp4', 'mov', 'avi', 'm4v'],
            help="Upload a video of yourself performing the selected exercise"
        )
        
        # Button to trigger analysis
        analyze_button = st.button(
            "Analyze Form",
            disabled=not video_file,
            type="primary"
        )
    
    with col2:
        if video_file is not None:
            st.video(video_file)
            st.caption("Preview of uploaded video")
    
    # Display processing message
    if analyze_button and video_file:
        with st.spinner(f"Analyzing {exercise_type.replace('_', ' ')} form..."):
            results = analyze_form_with_user_exercise(video_file, exercise_type)
        
        if results:
            st.success("Analysis complete!")
            display_analysis_results(results)
        else:
            st.error("Failed to analyze the video. Please make sure the backend API is running and try again.")
    
    # Add information about the system
    with st.expander("ℹ️ About This System", expanded=False):
        st.markdown("""
        ### How It Works
        
        1. **Upload**: Select your exercise type and upload a video of yourself performing it
        2. **Analysis**: Our AI analyzes your form by comparing it to expert reference models
        3. **Feedback**: Receive detailed feedback on form errors and personalized recommendations
        
        ### Supported Exercises
        - Squats: Analysis of knee alignment, hip movement, depth, and stability
        - Push-ups: Evaluation of body position, arm extension, and form maintenance
        - Bicep Curls: Examination of elbow positioning, movement range, and stability
        - Shoulder Press: Assessment of bar path, positioning, and core stability
        
        ### Understanding Results
        - **Overall Score**: 0-100 scale indicating similarity to expert form
        - **Error Confidence**: Percentage indicating how significant the deviation is
        - **Severity**: Low, medium, or high based on deviation from reference
        """)
    
    # Add footer
    st.markdown("---")
    st.caption("💡 Tip: Ensure good lighting and a clear view of your full body for best analysis results.")

if __name__ == "__main__":
    main()