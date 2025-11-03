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

Update repository description on GitHub

The short repository description (the one under the repo name on GitHub) can't be changed by a git push; it must be updated in the GitHub UI or via the GitHub API/CLI. If you want me to update that for you, provide a GitHub personal access token (PAT) with `repo` scope and the desired description text and I will call the GitHub API to update the repository metadata.

Proceed?

Reply `yes` and I'll add the remote and push to `pieby2/Face-recognition` now. If you want me to also update the GitHub repo description, provide a PAT and the description text.
