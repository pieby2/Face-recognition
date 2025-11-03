import sqlite3
import numpy as np
import json
from typing import List, Dict, Optional

DATABASE_PATH = "FRS_Microservice/face_gallery.db"

def init_db():
    """Initializes the SQLite database and creates the identities table."""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # The 'embedding' column stores the 512-dimensional vector as a JSON string
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS identities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding TEXT NOT NULL,
            image_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def add_identity(name: str, embedding: np.ndarray, image_path: str) -> int:
    """Adds a new identity and its face embedding to the gallery."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Convert numpy array to JSON string for storage
    embedding_json = json.dumps(embedding.tolist())
    
    cursor.execute(
        "INSERT INTO identities (name, embedding, image_path) VALUES (?, ?, ?)",
        (name, embedding_json, image_path)
    )
    conn.commit()
    identity_id = cursor.lastrowid
    conn.close()
    return identity_id

def get_all_embeddings() -> List[Dict]:
    """Retrieves all identities, names, and embeddings from the gallery."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, embedding FROM identities")
    rows = cursor.fetchall()
    conn.close()
    
    gallery = []
    for row in rows:
        identity_id, name, embedding_json = row
        # Convert JSON string back to numpy array
        embedding = np.array(json.loads(embedding_json), dtype=np.float32)
        gallery.append({
            "id": identity_id,
            "name": name,
            "embedding": embedding
        })
        
    return gallery

def list_identities() -> List[Dict]:
    """Lists all registered identities with their metadata (excluding embedding)."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, image_path, timestamp FROM identities")
    rows = cursor.fetchall()
    conn.close()
    
    identities = []
    for row in rows:
        identity_id, name, image_path, timestamp = row
        identities.append({
            "id": identity_id,
            "name": name,
            "image_path": image_path,
            "timestamp": timestamp
        })
        
    return identities

# Initialize the database when the module is imported
import os
init_db()

if __name__ == '__main__':
    # Simple test case
    print("--- Testing Database Utilities ---")
    
    # 1. Initialize (already done on import)
    print(f"Database initialized at {DATABASE_PATH}")
    
    # 2. Add a dummy identity
    dummy_embedding = np.random.rand(512).astype(np.float32)
    dummy_name = "Test User"
    dummy_path = "/path/to/test_image.jpg"
    
    identity_id = add_identity(dummy_name, dummy_embedding, dummy_path)
    print(f"Added identity '{dummy_name}' with ID: {identity_id}")
    
    # 3. Retrieve all embeddings
    gallery = get_all_embeddings()
    print(f"Gallery size: {len(gallery)}")
    
    # 4. List identities
    id_list = list_identities()
    print(f"Listed identities: {id_list}")
    
    # 5. Clean up (optional, but good practice for testing)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("DELETE FROM identities WHERE id=?", (identity_id,))
    conn.commit()
    conn.close()
    print(f"Cleaned up identity with ID: {identity_id}")
