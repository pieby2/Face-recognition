import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import os
import cv2

# Set device to CPU as required by the assignment
DEVICE = torch.device('cpu')

class FaceRecognitionPipeline:
    def __init__(self, min_face_size=20, threshold=0.6, keep_all=True):
        """
        Initializes the Face Recognition Pipeline with MTCNN for detection/alignment
        and InceptionResnetV1 (FaceNet) for feature extraction.
        """
        print(f"Initializing models on device: {DEVICE}")
        
        # 1. Face Detection and Alignment (MTCNN)
        # MTCNN is used for detection and also handles the cropping/alignment
        # 'image_size' is the size of the output face crop (e.g., 160 for FaceNet)
        # 'margin' is the margin added to the bounding box
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=min_face_size, 
            thresholds=[0.6, 0.7, 0.7], 
            factor=0.709, 
            post_process=True, # Performs alignment and normalization
            device=DEVICE,
            keep_all=keep_all # Keep all detected faces
        )

        # 2. Feature Extractor (FaceNet - InceptionResnetV1)
        # Pretrained on VGGFace2
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    def detect_and_align(self, img_path):
        """
        Detects faces in an image and returns the aligned face tensors and bounding boxes.
        
        Args:
            img_path (str): Path to the input image.
            
        Returns:
            tuple: (aligned_faces, boxes) where aligned_faces is a torch.Tensor 
                   and boxes is a numpy array of bounding boxes.
        """
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            return None, None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None, None

        # MTCNN returns a tensor of aligned faces (N, 3, 160, 160) and the bounding boxes
        # The 'post_process=True' in MTCNN handles the 5-point alignment and normalization.
        aligned_faces, boxes = self.mtcnn(img, save_path=None, return_prob=False)
        
        if aligned_faces is None:
            return None, None
        
        # Convert boxes to numpy array for easier handling
        boxes = boxes.astype(int) if boxes is not None else None
        
        return aligned_faces, boxes

    @torch.no_grad()
    def extract_features(self, aligned_faces):
        """
        Extracts 512-dimensional embeddings from aligned face tensors.
        
        Args:
            aligned_faces (torch.Tensor): Tensor of aligned faces (N, 3, 160, 160).
            
        Returns:
            numpy.ndarray: Array of face embeddings (N, 512).
        """
        if aligned_faces is None:
            return None
            
        # Generate embeddings
        embeddings = self.resnet(aligned_faces.to(DEVICE)).cpu().numpy()
        return embeddings

    def process_image(self, img_path):
        """
        Performs detection, alignment, and feature extraction on a single image.
        """
        aligned_faces, boxes = self.detect_and_align(img_path)
        
        if aligned_faces is None:
            return [], []

        embeddings = self.extract_features(aligned_faces)
        
        # Combine boxes and embeddings for output
        results = []
        for box, embedding in zip(boxes, embeddings):
            results.append({
                "box": box.tolist(),
                "embedding": embedding.tolist()
            })
            
        return results

if __name__ == '__main__':
    # Simple test case: Create a dummy image for testing
    dummy_img_path = "FRS_Microservice/test_face.jpg"
    
    # Create a dummy image file (a simple black image) for testing the pipeline
    # In a real scenario, this would be a real image with a face.
    # Since I cannot download a real image, I will just create a placeholder
    # and rely on the user to provide a real image for actual testing.
    # For now, I will just test the initialization.
    
    # Create a placeholder image to avoid FileNotFoundError during the test run
    if not os.path.exists(dummy_img_path):
        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(dummy_img_path, dummy_img)
        print(f"Created placeholder image at {dummy_img_path}")

    print("--- Testing Face Recognition Pipeline Initialization ---")
    pipeline = FaceRecognitionPipeline()
    print("Pipeline initialized successfully.")
    
    print("\n--- Testing Image Processing (will likely fail to detect a face on placeholder) ---")
    results = pipeline.process_image(dummy_img_path)
    
    if results:
        print(f"Detected {len(results)} face(s).")
        print(f"First face bounding box: {results[0]['box']}")
        print(f"First face embedding shape: {np.array(results[0]['embedding']).shape}")
    else:
        print("No faces detected (expected for placeholder image).")
        
    # Clean up placeholder
    os.remove(dummy_img_path)
    print(f"Cleaned up placeholder image at {dummy_img_path}")
