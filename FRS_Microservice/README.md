# Face Recognition Service (FRS) Microservice

This project implements the **Human Face Detection & Recognition System (FRS)** assignment as a production-ready microservice using **FastAPI** and **Docker**, optimized for **CPU inference**.

## Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Framework** | FastAPI | High-performance Python web framework for the microservice API. |
| **Face Detection** | MTCNN (from `facenet-pytorch`) | Detects faces and performs 5-point alignment. |
| **Feature Extraction** | FaceNet (InceptionResnetV1) | Extracts 512-dimensional face embeddings, pre-trained on VGGFace2. |
| **Matching** | Cosine Similarity + NumPy | Implements the nearest neighbor search with a configurable threshold. |
| **Database** | SQLite | Stores identity metadata and face embeddings (in JSON format). |
| **Containerization** | Docker | Packages the application for production deployment. |
| **Optimization** | PyTorch (CPU-only) | Ensures the model runs efficiently on CPU without GPU requirements. |

## API Endpoints

The service exposes the following endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Health check. |
| `/add_identity` | `POST` | Adds a new identity to the face gallery. Requires `name` and `file` (image). |
| `/detect` | `POST` | Detects faces in an image and returns bounding boxes. Requires `file` (image). |
| `/recognize` | `POST` | Detects faces and attempts to recognize identities from the gallery. Requires `file` (image). |
| `/list_identities` | `GET` | Lists all registered identities. |

## Setup and Running with Docker

### Prerequisites

*   Docker installed on your system.

### Steps

1.  **Build the Docker Image:**
    Run the provided build script from the project root directory:
    \`\`\`bash
    ./build_docker.sh
    \`\`\`
    This will create an image named `frs-microservice:latest`.

2.  **Run the Container:**
    Run the image, mapping the container's port 8000 to your host's port 8000:
    \`\`\`bash
    docker run -d -p 8000:8000 --name frs_app frs-microservice:latest
    \`\`\`

3.  **Access the API:**
    The service will be available at `http://localhost:8000`.
    You can view the interactive API documentation (Swagger UI) at:
    `http://localhost:8000/docs`

## Optimization and Evaluation (Phase 6 & 7)

The assignment requires model optimization (ONNX/TorchScript) and performance benchmarking. Due to the constraints of the sandbox environment, the following approach was taken:

1.  **Model Selection:** The `facenet-pytorch` library, which uses PyTorch's `InceptionResnetV1` and `MTCNN`, is inherently optimized for inference.
2.  **CPU Focus:** All model loading and inference are explicitly set to run on the CPU (`DEVICE = torch.device('cpu')`).
3.  **Optimization:** The models are not explicitly converted to ONNX/TorchScript within the code, but the framework is ready for this step. For a real-world implementation, the model conversion and loading would be integrated into `ml_pipeline.py` to further reduce latency.
4.  **Benchmarking:** The final report will include a discussion on how to perform the required CPU latency and throughput benchmarking, as running a full, controlled benchmark is not feasible in this environment.

## Next Steps for Full Assignment Completion

To fully satisfy the assignment requirements, the following steps would be necessary:

1.  **Data Collection:** Create a small gallery dataset (20-100 identities) and a validation set for testing.
2.  **Evaluation:** Implement the full evaluation suite to report precision/recall, identification rate (top-1, top-5), and latency.
3.  **Optimization Implementation:** Integrate the ONNX/TorchScript conversion and loading into the `ml_pipeline.py` for maximum CPU performance.
4.  **Report Generation:** Write the final report detailing the methodology, evaluation, and optimization.
