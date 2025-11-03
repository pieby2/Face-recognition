# Face Recognition Service (FRS) Microservice — ready for push to pieby2/Face-recognition

This local copy implements the Human Face Detection & Recognition microservice used for the ML internship assignment. The main microservice code is inside the `FRS_Microservice/` package.

Target remote repository: https://github.com/pieby2/Face-recognition

What's included

- `FRS_Microservice/` — microservice code (API, DB helpers, matching, ML pipeline, evaluation).
- `requirements.txt` — Python dependencies.
- `LICENSE` — MIT license.
- `.gitignore` — ignores runtime DB and local files.

Quick sanity checks

1. Remove or exclude any large/private files you don't want to publish. `.gitignore` already excludes `FRS_Microservice/face_gallery.db`.
2. Run the synthetic evaluation locally to validate the matching logic (no real faces required):

```powershell
python -m FRS_Microservice.evaluation --n-identities 30 --n-queries 120 --match-ratio 0.5
```

3. Optionally let the matching demo auto-download a DB by setting the `DB_DOWNLOAD_URL` environment variable before running the demo:

```powershell
$env:DB_DOWNLOAD_URL = 'https://example.com/path/to/face_gallery.db' ; python -m FRS_Microservice.matching_service
```

Push instructions (what I'll run)

I will add the remote `origin` for you and attempt to push the local `main` branch to GitHub. If authentication is required, Git may prompt or fail — I'll report the result.

Commands to push (PowerShell):

```powershell
git remote add origin https://github.com/pieby2/Face-recognition
git push -u origin main
```

## Dataset Download Instructions

This Face Recognition Service uses pre-trained models and can work with various face datasets for training, validation, and testing. Below are instructions for downloading commonly used datasets.

### Pre-trained Models

The service uses **FaceNet (InceptionResnetV1)** pre-trained on **VGGFace2**, which is automatically downloaded by the `facenet-pytorch` library when you first run the application. No manual download is required for the pre-trained models.

### Datasets for Training and Validation

#### 1. VGGFace2 Dataset

VGGFace2 is a large-scale face recognition dataset with 3.31 million images of 9,131 subjects.

**Download:**
- Official website: [https://github.com/ox-vgg/vgg_face2](https://github.com/ox-vgg/vgg_face2)
- Registration required: You need to request access through the official repository
- The dataset is available in two parts:
  - Training set: ~3.31M images
  - Test set: ~169K images

**Steps:**
1. Visit the VGGFace2 GitHub repository
2. Follow the instructions to request access
3. Once approved, download the dataset using the provided links
4. Extract the downloaded files to your desired location

#### 2. WIDER FACE Dataset

WIDER FACE is a face detection benchmark dataset with 32,203 images and 393,703 labeled faces.

**Download:**
- Official website: [http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/)
- Direct download links are available on the website

**Steps:**
1. Visit the WIDER FACE website
2. Download the following files:
   - WIDER Face Training Images
   - WIDER Face Validation Images  
   - WIDER Face Testing Images
   - Face annotations
3. Extract the files to your working directory

#### 3. LFW (Labeled Faces in the Wild) Dataset

LFW is a standard benchmark for face verification with 13,000+ images of 5,749+ people.

**Download:**
- Official website: [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)
- Direct download: [http://vis-www.cs.umass.edu/lfw/lfw.tgz](http://vis-www.cs.umass.edu/lfw/lfw.tgz)

**Steps:**
1. Download the dataset using wget or curl:
   ```bash
   wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
   ```
2. Extract the archive:
   ```bash
   tar -xzf lfw.tgz
   ```

### Face Gallery Database (Optional)

For the matching service demo, you can optionally download a pre-populated face gallery database:

1. Set the `DB_DOWNLOAD_URL` environment variable to point to your database file
2. Run the matching service:

```bash
# Linux/Mac
export DB_DOWNLOAD_URL='https://example.com/path/to/face_gallery.db'
python -m FRS_Microservice.matching_service

# Windows (PowerShell)
$env:DB_DOWNLOAD_URL = 'https://example.com/path/to/face_gallery.db'
python -m FRS_Microservice.matching_service
```

The database will be automatically downloaded if it doesn't exist locally.

### Creating Your Own Dataset

Instead of using large public datasets, you can create a custom face gallery:

1. Start the FastAPI service:
   ```bash
   uvicorn main:app --reload
   ```

2. Use the `/add_identity` endpoint to add faces to your gallery
3. Upload images with the person's name to build your own recognition database

Refer to the API documentation at `http://localhost:8000/docs` for detailed endpoint usage.

---

Update repository description on GitHub

The short repository description (the one under the repo name on GitHub) can't be changed by a git push; it must be updated in the GitHub UI or via the GitHub API/CLI. If you want me to update that for you, provide a GitHub personal access token (PAT) with `repo` scope and the desired description text and I will call the GitHub API to update the repository metadata.

Proceed?

Reply `yes` and I'll add the remote and push to `pieby2/Face-recognition` now. If you want me to also update the GitHub repo description, provide a PAT and the description text.
