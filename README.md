# Face Recognition Service (FRS) Microservice

This repository contains the Face Recognition Service (FRS) Microservice used for the "Implementing ML Internship 2025" assignment. The main microservice code lives in the `FRS_Microservice/` folder.

What you'll find here:

- `FRS_Microservice/` — the microservice package (FastAPI endpoints, database utilities, matching code, ML pipeline, and evaluation).
- `requirements.txt` — Python dependencies.
- `LICENSE` — MIT License.
- `.gitignore` — sensible defaults for Python projects (ignores `FRS_Microservice/face_gallery.db`, virtualenvs, etc.).

Quick checklist before pushing to GitHub

1. Remove any large or private files you don't want in the repository (for example the local SQLite DB `FRS_Microservice/face_gallery.db`). The provided `.gitignore` already ignores `FRS_Microservice/face_gallery.db`.
2. Verify code runs locally. You can run the synthetic evaluation to sanity-check the matching logic (no real images required):

```powershell
python -m FRS_Microservice.evaluation --n-identities 30 --n-queries 120 --match-ratio 0.5
```

3. If you'd like the demo to pull a pre-built DB automatically, set an environment variable `DB_DOWNLOAD_URL` to a valid HTTP/HTTPS URL pointing at a `.db` file before running the matching demo:

```powershell
$env:DB_DOWNLOAD_URL = 'https://example.com/path/to/face_gallery.db' ; python -m FRS_Microservice.matching_service
```

How to push this folder to GitHub (PowerShell)

```powershell
git init
git add .
git commit -m "Initial project import: Face Recognition Service (FRS) Microservice"
# create a remote repo on GitHub and then add it (replace URL with your repo)
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

Notes

- The repository is set up so you can add real gallery images and run the pipeline in `ml_pipeline.py` to build a production-like evaluation. Real model inference requires the packages listed in `requirements.txt` and may be slow on CPU.
- If you'd like, I can also add templates for CONTRIBUTING.md, issue/PR templates, or a richer CI workflow that runs unit tests or static checks. Tell me which you'd like next.
