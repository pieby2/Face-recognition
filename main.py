from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from FRS_Microservice.ml_pipeline import FaceRecognitionPipeline
from FRS_Microservice.database import add_identity, list_identities
from FRS_Microservice.matching_service import MatchingService
from typing import List, Dict
import os
import shutil
import uuid
import numpy as np

# --- Configuration ---
UPLOAD_DIR = "FRS_Microservice/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Initialization ---
app = FastAPI(
    title="Face Recognition Service (FRS) Microservice",
    description="Production-ready FRS microservice for face detection and identity recognition, optimized for CPU inference.",
    version="1.0.0"
)

# Initialize ML and Matching components
# Note: These are initialized once when the app starts
try:
    pipeline = FaceRecognitionPipeline()
    matcher = MatchingService(threshold=0.6, top_k=1)
except Exception as e:
    print(f"Error initializing ML components: {e}")
    pipeline = None
    matcher = None

# --- Utility Functions ---
def save_upload_file(upload_file: UploadFile) -> str:
    """Saves the uploaded file to the UPLOAD_DIR and returns the path."""
    try:
        # Generate a unique filename to prevent conflicts
        file_extension = os.path.splitext(upload_file.filename)[1]
        unique_filename = str(uuid.uuid4()) + file_extension
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint for service health check."""
    return {"message": "FRS Microservice is running. Visit /docs for API documentation."}

@app.post("/add_identity")
async def add_new_identity(
    name: str,
    file: UploadFile = File(...)
):
    """
    Adds a new identity to the face gallery.
    The image should contain a single, clear face.
    """
    if not pipeline or not matcher:
        raise HTTPException(status_code=503, detail="ML service is not initialized.")
        
    file_path = save_upload_file(file)
    
    try:
        # 1. Process image to get embedding
        results = pipeline.process_image(file_path)
        
        if not results:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image.")
        
        if len(results) > 1:
            # The MTCNN is set to keep_all=True, so we take the first one, 
            # but warn the user that multiple faces were found.
            print(f"Warning: Multiple faces ({len(results)}) detected. Using the first one.")
            
        embedding = np.array(results[0]["embedding"], dtype=np.float32)
        
        # 2. Add to database
        identity_id = add_identity(name, embedding, file_path)
        
        # 3. Reload the gallery in the matching service
        matcher.load_gallery()
        
        return JSONResponse(content={
            "status": "success",
            "identity_id": identity_id,
            "name": name,
            "message": "Identity successfully added and gallery reloaded."
        })
        
    finally:
        # Clean up the temporary file if it's not the one we want to keep (optional, 
        # but for this assignment, we'll keep the file path in the DB for simplicity)
        # For a real system, we would move the file to a permanent storage (e.g., S3)
        pass


@app.post("/detect")
async def detect_faces(
    file: UploadFile = File(...)
) -> List[Dict]:
    """
    Detects faces in an image and returns the bounding boxes.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="ML service is not initialized.")
        
    file_path = save_upload_file(file)
    
    try:
        # Use the pipeline's detect_and_align function, but we only need the boxes
        _, boxes = pipeline.detect_and_align(file_path)
        
        if boxes is None:
            return []
            
        # Convert numpy array of boxes to list of lists for JSON serialization
        return [{"box": box.tolist()} for box in boxes]
        
    finally:
        os.remove(file_path)


@app.post("/recognize")
async def recognize_identity(
    file: UploadFile = File(...)
) -> List[Dict]:
    """
    Detects faces, extracts features, and attempts to recognize identities 
    from the face gallery.
    """
    if not pipeline or not matcher:
        raise HTTPException(status_code=503, detail="ML service is not initialized.")
        
    file_path = save_upload_file(file)
    
    try:
        # 1. Process image to get all detected faces and their embeddings
        results = pipeline.process_image(file_path)
        
        if not results:
            return []
            
        recognition_results = []
        
        # 2. Match each detected face
        for face_result in results:
            embedding = np.array(face_result["embedding"], dtype=np.float32)
            
            # Perform matching
            matches = matcher.match_identity(embedding)
            
            # Format the output
            recognition_results.append({
                "bounding_box": face_result["box"],
                "matches": matches,
                "identity": matches[0]["name"] if matches else "Unknown",
                "confidence": matches[0]["confidence"] if matches else 0.0
            })
            
        return recognition_results
        
    finally:
        os.remove(file_path)


@app.get("/list_identities")
async def get_identities():
    """
    Lists all registered identities in the face gallery.
    """
    return list_identities()

# --- Cleanup (Optional, for a real system) ---
@app.on_event("shutdown")
def shutdown_event():
    # Clean up the uploads directory on shutdown
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    print(f"Cleaned up upload directory: {UPLOAD_DIR}")
    
    # Clean up the database file (optional, but good for a clean sandbox environment)
    db_path = "FRS_Microservice/face_gallery.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Cleaned up database file: {db_path}")
