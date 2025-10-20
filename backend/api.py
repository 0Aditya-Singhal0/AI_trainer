from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import os
import tempfile
from datetime import datetime

from models import FeedbackItem, FormFeedback, VideoUploadResponse
from services import FormAnalysisService

app = FastAPI(title="Genyx Fitness AI Trainer API", 
              description="Real-time workout form error detection and analysis microservice",
              version="1.0.0")

# Initialize service
service = FormAnalysisService()

@app.post("/analyze-form", response_model=VideoUploadResponse)
async def analyze_form(file: UploadFile = File(...)):
    """
    Analyze workout form from video file upload with automatic exercise detection
    """
    # Create temporary file to store uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name
    
    try:
        # Validate file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ['mp4', 'mov', 'avi', 'm4v']:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a video file.")
        
        # Analyze form using the service with auto-detection
        result = service.analyze_video_form(temp_file_path, auto_detect=True)
        
        # Convert to response format
        if result["form_feedback"]["detected_errors"]:
            detected_errors = [FeedbackItem(**error) for error in result["form_feedback"]["detected_errors"]]
        else:
            detected_errors = []
        
        form_feedback = FormFeedback(
            timestamp=result["form_feedback"]["timestamp"],
            exercise_type=result["form_feedback"]["exercise_type"],
            detected_errors=detected_errors,
            overall_score=result["form_feedback"]["overall_score"],
            recommendations=result["form_feedback"]["recommendations"]
        )
        
        return VideoUploadResponse(
            status=result["status"],
            form_feedback=form_feedback,
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        print(f"Error in analyze_form: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/analyze-form-user-specified", response_model=VideoUploadResponse)
async def analyze_form_user_specified(exercise_type: str = Form(...), file: UploadFile = File(...)):
    """
    Analyze workout form from video file upload with user-specified exercise
    """
    # Create temporary file to store uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name
    
    try:
        # Validate file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ['mp4', 'mov', 'avi', 'm4v']:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a video file.")
        
        # Validate exercise type
        valid_exercises = ['push-up', 'squat', 'bicep_curl', 'shoulder_press', 'plank']
        if exercise_type.lower() not in valid_exercises:
            raise HTTPException(status_code=400, detail=f"Invalid exercise type. Valid types: {valid_exercises}")
        
        # Analyze form using the service with user-specified exercise
        result = service.analyze_video_form(temp_file_path, auto_detect=False, specified_exercise=exercise_type.lower())
        
        # Convert to response format
        if result["form_feedback"]["detected_errors"]:
            detected_errors = [FeedbackItem(**error) for error in result["form_feedback"]["detected_errors"]]
        else:
            detected_errors = []
        
        form_feedback = FormFeedback(
            timestamp=result["form_feedback"]["timestamp"],
            exercise_type=result["form_feedback"]["exercise_type"],
            detected_errors=detected_errors,
            overall_score=result["form_feedback"]["overall_score"],
            recommendations=result["form_feedback"]["recommendations"]
        )
        
        return VideoUploadResponse(
            status=result["status"],
            form_feedback=form_feedback,
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        print(f"Error in analyze_form_user_specified: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/analyze-frame", response_model=FormFeedback)
async def analyze_frame(file: UploadFile = File(...)):
    """
    Analyze workout form from single image frame with automatic exercise detection
    """
    # Create temporary file to store uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name
    
    try:
        # Validate file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ['jpg', 'jpeg', 'png']:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image file.")
        
        # Analyze form using the service with auto-detection
        result = service.analyze_image_form(temp_file_path, auto_detect=True)
        
        # Convert to response format
        if result["form_feedback"]["detected_errors"]:
            detected_errors = [FeedbackItem(**error) for error in result["form_feedback"]["detected_errors"]]
        else:
            detected_errors = []
        
        form_feedback = FormFeedback(
            timestamp=result["form_feedback"]["timestamp"],
            exercise_type=result["form_feedback"]["exercise_type"],
            detected_errors=detected_errors,
            overall_score=result["form_feedback"]["overall_score"],
            recommendations=result["form_feedback"]["recommendations"]
        )
        
        return form_feedback
        
    except Exception as e:
        print(f"Error in analyze_frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/analyze-frame-user-specified", response_model=FormFeedback)
async def analyze_frame_user_specified(exercise_type: str = Form(...), file: UploadFile = File(...)):
    """
    Analyze workout form from single image frame with user-specified exercise
    """
    # Create temporary file to store uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name
    
    try:
        # Validate file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ['jpg', 'jpeg', 'png']:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image file.")
        
        # Validate exercise type
        valid_exercises = ['push-up', 'squat', 'bicep_curl', 'shoulder_press', 'plank']
        if exercise_type.lower() not in valid_exercises:
            raise HTTPException(status_code=400, detail=f"Invalid exercise type. Valid types: {valid_exercises}")
        
        # Analyze form using the service with user-specified exercise
        result = service.analyze_image_form(temp_file_path, auto_detect=False, specified_exercise=exercise_type.lower())
        
        # Convert to response format
        if result["form_feedback"]["detected_errors"]:
            detected_errors = [FeedbackItem(**error) for error in result["form_feedback"]["detected_errors"]]
        else:
            detected_errors = []
        
        form_feedback = FormFeedback(
            timestamp=result["form_feedback"]["timestamp"],
            exercise_type=result["form_feedback"]["exercise_type"],
            detected_errors=detected_errors,
            overall_score=result["form_feedback"]["overall_score"],
            recommendations=result["form_feedback"]["recommendations"]
        )
        
        return form_feedback
        
    except Exception as e:
        print(f"Error in analyze_frame_user_specified: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/model-info")
async def model_info():
    """Get information about the loaded models"""
    return service.get_model_info()