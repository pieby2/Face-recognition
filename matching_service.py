import numpy as np
from typing import List, Dict, Optional
from scipy.spatial.distance import cosine
from FRS_Microservice.database import get_all_embeddings

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
            print("Warning: Face gallery is empty.")
            return

        # Create a matrix of all embeddings for efficient calculation
        embeddings_list = [item["embedding"] for item in self.gallery]
        self.embeddings_matrix = np.stack(embeddings_list)
        print(f"Loaded {len(self.gallery)} identities into the matching gallery.")

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
                    "confidence": 1.0 - distance, # Convert distance back to similarity/confidence
                    "distance": distance
                })
            
            if len(matches) >= self.top_k:
                break
                
        return matches

if __name__ == '__main__':
    # Simple test case
    from FRS_Microservice.database import add_identity
    
    print("--- Testing Matching Service ---")
    
    # 1. Ensure database is initialized and empty
    # (database.py init_db() runs on import, but we need to ensure clean state for test)
    import sqlite3
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("DELETE FROM identities")
    conn.commit()
    conn.close()
    
    # 2. Add dummy identities
    # Identity 1: Known
    known_emb = np.array([0.1, 0.9, 0.1, 0.1, 0.1]) # Simplified 5-dim embedding
    known_id = add_identity("Alice", known_emb, "path/to/alice.jpg")
    
    # Identity 2: Different
    diff_emb = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
    diff_id = add_identity("Bob", diff_emb, "path/to/bob.jpg")
    
    # 3. Initialize Matching Service
    # Note: The actual FaceNet embeddings are 512-dim, but this test uses 5-dim.
    # The threshold is set to 0.6 (distance).
    matcher = MatchingService(threshold=0.6, top_k=1)
    
    # 4. Test a query that should match Alice (low distance/high similarity)
    query_emb_match = np.array([0.11, 0.89, 0.1, 0.1, 0.1])
    matches = matcher.match_identity(query_emb_match)
    
    print("\nQuery 1 (Match Alice):")
    if matches:
        print(f"Match found: {matches[0]['name']} with confidence {matches[0]['confidence']:.4f} (distance {matches[0]['distance']:.4f})")
    else:
        print("No match found.")
        
    # 5. Test a query that should not match (high distance/low similarity)
    query_emb_no_match = np.array([0.5, 0.5, 0.5, 0.5, 0.5]) # Should be far from both
    matches = matcher.match_identity(query_emb_no_match)
    
    print("\nQuery 2 (No Match):")
    if matches:
        print(f"Match found: {matches[0]['name']} with confidence {matches[0]['confidence']:.4f} (distance {matches[0]['distance']:.4f})")
    else:
        print("No match found.")
        
    # 6. Clean up
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("DELETE FROM identities WHERE id IN (?, ?)", (known_id, diff_id))
    conn.commit()
    conn.close()
    print("\nCleaned up test identities.")
