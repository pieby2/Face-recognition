import os
import logging
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional
from scipy.spatial.distance import cosine
from FRS_Microservice.database import get_all_embeddings, DATABASE_PATH

# Small logger configured for friendlier, human-like messages
logger = logging.getLogger("MatchingService")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def download_database_if_missing(url: Optional[str] = None, dest: Optional[str] = None, overwrite: bool = False) -> bool:
    """Download the SQLite database file if it's missing.

    Behavior:
    - If dest exists and overwrite is False -> do nothing and return True.
    - If dest missing and url provided -> try to download and save to dest.
    - If dest missing and url None -> return False.

    Returns True when the DB file exists locally (either already present or successfully downloaded).
    """
    dest = dest or DATABASE_PATH
    dest_path = Path(dest)

    if dest_path.exists() and not overwrite:
        logger.info(f"Nice — database already exists at '{dest}'. No download needed.")
        return True

    # Prefer explicit url parameter, fall back to environment variable
    url = url or os.environ.get("DB_DOWNLOAD_URL")
    if not url:
        logger.info("No database file found and no download URL provided. Create or provide a DB to proceed.")
        return False

    try:
        logger.info(f"Downloading database from {url} — this may take a moment...")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        logger.info(f"All set — downloaded database to '{dest}'.")
        return True
    except urllib.error.URLError as e:
        logger.info(f"Couldn't download the database: {e}. Please check the URL and your connection.")
        return False
    except Exception as e:
        logger.info(f"Unexpected error while downloading DB: {e}")
        return False

class MatchingService:
    def __init__(self, threshold: float = 0.6, top_k: int = 1):
        """
        Initializes the Matching Service with configurable threshold and top-K return.
        
        Args:
            threshold (float): Maximum cosine distance for a match to be considered valid.
            top_k (int): Number of top matches to return.
        """
        self.threshold = threshold
        self.top_k = top_k
        self.gallery = []
        self.embeddings_matrix = None
        self.load_gallery()

    def load_gallery(self):
        """Loads all identities and embeddings from the database into memory."""
        self.gallery = get_all_embeddings()

        if not self.gallery:
            self.embeddings_matrix = None
            logger.info("Heads-up: your face gallery looks empty right now. Add some identities and I'll be ready to match.")
            return

        # Create a matrix of all embeddings for efficient calculation
        embeddings_list = [item["embedding"] for item in self.gallery]
        try:
            self.embeddings_matrix = np.stack(embeddings_list)
        except Exception as e:
            # Defensive: if embeddings lengths differ, report clearly
            logger.info(f"Could not stack embeddings into a matrix: {e}. Check that stored embeddings have consistent dimensions.")
            self.embeddings_matrix = None
            return

        logger.info(f"Loaded {len(self.gallery)} identities into the matching gallery. Ready to match! 😊")

    def match_identity(self, query_embedding: np.ndarray) -> List[Dict]:
        """
        Performs nearest neighbor search on the gallery using cosine similarity.
        
        Args:
            query_embedding (np.ndarray): The 512-dimensional embedding of the face to recognize.
            
        Returns:
            List[Dict]: A list of top-K matches, sorted by confidence (lower distance is better).
        """
        if self.embeddings_matrix is None or len(self.gallery) == 0:
            return []

        # Calculate cosine distance between the query and all gallery embeddings
        # Note: cosine(u, v) returns 1 - cosine_similarity(u, v)
        # We want similarity, so we look for the minimum distance.
        # We can use scipy.spatial.distance.cosine for a single query, 
        # or numpy for a matrix operation if we had multiple queries.
        
        # For a single query, we can use the dot product for cosine similarity:
        # Cosine Similarity = (A . B) / (||A|| * ||B||)
        # Since FaceNet embeddings are L2 normalized, ||A|| and ||B|| are 1.
        # Cosine Similarity = A . B
        
        # Reshape query for matrix multiplication
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity (dot product)
        similarities = np.dot(self.embeddings_matrix, query_embedding.T).flatten()
        
        # Convert similarity to distance (1 - similarity)
        distances = 1 - similarities
        
        # Find indices of the top-K smallest distances
        # We use argsort to get indices of sorted array, then take the first K
        sorted_indices = np.argsort(distances)
        
        matches = []
        for i in sorted_indices:
            distance = distances[i]
            
            # Check against the configurable threshold
            if distance <= self.threshold:
                identity = self.gallery[i]
                matches.append({
                    "identity_id": identity["id"],
                    "name": identity["name"],
                    "confidence": float(1.0 - distance), # Convert distance back to similarity/confidence
                    "distance": float(distance)
                })
            
            if len(matches) >= self.top_k:
                break
                
        return matches

if __name__ == '__main__':
    # Simple, human-friendly test + DB-download helper
    from FRS_Microservice.database import add_identity
    import sqlite3

    logger.info("--- Quick demo: Matching Service (friendly mode) ---")

    # Attempt to download DB if it's missing. Set DB_DOWNLOAD_URL env var to enable automatic download.
    db_ready = download_database_if_missing()
    if not db_ready:
        logger.info("I'll continue but the database isn't available. To auto-download, set DB_DOWNLOAD_URL to a valid file URL.")

    # 1. Ensure database is initialized and clean for the demo
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute("DELETE FROM identities")
        conn.commit()
        conn.close()
        logger.info("Database cleared for the short demo — ready to add a couple of identities.")
    except Exception as e:
        logger.info(f"Could not reset the database: {e}")

    # 2. Add a couple of small dummy identities (toy 5-D embeddings for demo only)
    known_emb = np.array([0.1, 0.9, 0.1, 0.1, 0.1], dtype=np.float32)
    known_id = add_identity("Alice", known_emb, "path/to/alice.jpg")

    diff_emb = np.array([0.9, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    diff_id = add_identity("Bob", diff_emb, "path/to/bob.jpg")

    logger.info("Added two demo identities: Alice and Bob.")

    # 3. Initialize Matching Service (note: real embeddings are typically 512-D)
    matcher = MatchingService(threshold=0.6, top_k=1)

    # 4. Query that should match Alice
    query_emb_match = np.array([0.11, 0.89, 0.1, 0.1, 0.1], dtype=np.float32)
    matches = matcher.match_identity(query_emb_match)

    logger.info("\nQuery 1 — expected: Alice")
    if matches:
        m = matches[0]
        logger.info(f"Sweet — I think that's {m['name']} (confidence {m['confidence']*100:.1f}%) — distance {m['distance']:.4f}.")
    else:
        logger.info("No match found for Query 1.")

    # 5. Query that should not match
    query_emb_no_match = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    matches = matcher.match_identity(query_emb_no_match)

    logger.info("\nQuery 2 — expected: No match")
    if matches:
        m = matches[0]
        logger.info(f"Hmm — I think that's {m['name']} (confidence {m['confidence']*100:.1f}%).")
    else:
        logger.info("Yup — no one in the gallery looks like that.")

    # 6. Clean up demo identities
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute("DELETE FROM identities WHERE id IN (?, ?)", (known_id, diff_id))
        conn.commit()
        conn.close()
        logger.info("Cleaned up demo identities. All done!")
    except Exception as e:
        logger.info(f"Couldn't clean up demo identities: {e}")
