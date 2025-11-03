"""Evaluation utilities for the FRS Microservice.

This module provides a simple, reproducible evaluation flow that:
- Generates a synthetic gallery of identities (random, normalized embeddings).
- Creates a validation/query set with a controllable match ratio.
- Evaluates identification rate (top-1, top-5), precision and recall.
- Measures average matching latency.

Usage:
    python -m FRS_Microservice.evaluation

Note: This uses synthetic embeddings (random vectors) so results are illustrative
and useful for testing/evaluating the matching code-path. For real evaluation,
replace the gallery generation with real face images and embeddings produced by
the `FaceRecognitionPipeline` in `ml_pipeline.py`.
"""

import time
import random
import numpy as np
from typing import List, Tuple, Optional

from FRS_Microservice.database import add_identity, DATABASE_PATH
from FRS_Microservice.matching_service import MatchingService


def _normalize(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def generate_synthetic_gallery(n_identities: int = 50, emb_dim: int = 512) -> List[Tuple[int, np.ndarray]]:
    """Create `n_identities` random normalized embeddings and add them to DB.

    Returns a list of tuples (identity_id, embedding).
    """
    created = []
    for i in range(n_identities):
        emb = np.random.randn(emb_dim).astype(np.float32)
        emb = _normalize(emb)
        identity_id = add_identity(f"person_{i}", emb, "synthetic")
        created.append((identity_id, emb))
    return created


def generate_validation_set(gallery: List[Tuple[int, np.ndarray]],
                            n_queries: int = 200,
                            match_ratio: float = 0.5,
                            perturb_sigma: float = 0.08) -> List[Tuple[bool, Optional[int], np.ndarray]]:
    """Create a validation set of queries.

    Each entry is a tuple: (is_positive, true_identity_id_or_None, query_embedding)
    - `is_positive=True` means the query corresponds to an identity in the gallery
      (we create it by perturbing that identity's embedding slightly).
    - `is_positive=False` means the query should not match anyone (random embedding).
    """
    queries = []
    n_pos = int(n_queries * match_ratio)
    emb_dim = gallery[0][1].shape[0]

    # Positive queries: perturb an existing identity's embedding
    for _ in range(n_pos):
        identity_id, emb = random.choice(gallery)
        q = emb + np.random.normal(0, perturb_sigma, size=emb.shape).astype(np.float32)
        q = _normalize(q)
        queries.append((True, identity_id, q))

    # Negative queries: random embeddings unlikely to match
    for _ in range(n_queries - n_pos):
        q = np.random.randn(emb_dim).astype(np.float32)
        q = _normalize(q)
        queries.append((False, None, q))

    random.shuffle(queries)
    return queries


def evaluate_matching(matcher: MatchingService,
                      queries: List[Tuple[bool, Optional[int], np.ndarray]],
                      topk_list: List[int] = [1, 5]) -> dict:
    """Evaluate identification (top-k), precision and recall using the provided matcher.

    Returns a dict with top-k rates, precision, recall, counts, and average latency.
    """
    total = len(queries)
    topk_correct = {k: 0 for k in topk_list}

    TP = 0
    FP = 0
    FN = 0

    latencies = []

    for is_pos, true_id, q in queries:
        t0 = time.perf_counter()
        matches = matcher.match_identity(q)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

        # Extract matched ids in order (may be empty)
        matched_ids = [m['identity_id'] for m in matches]

        # Top-k checks
        for k in topk_list:
            topk_ids = matched_ids[:k]
            if is_pos and (true_id in topk_ids):
                topk_correct[k] += 1

        # Precision/recall bookkeeping
        if is_pos:
            if matches and matches[0]['identity_id'] == true_id:
                TP += 1
            else:
                FN += 1
                if matches:
                    FP += 1  # returned some identity but it's wrong
        else:
            if matches:
                FP += 1

    topk_rates = {k: topk_correct[k] / total for k in topk_list}
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000 if latencies else 0.0

    return {
        "total_queries": total,
        "topk": topk_rates,
        "precision": precision,
        "recall": recall,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "avg_match_latency_ms": avg_latency_ms
    }


def _clear_database():
    """Helper to clear the identities table for a fresh run."""
    import sqlite3
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("DELETE FROM identities")
    conn.commit()
    conn.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run a synthetic evaluation of the matching service.")
    parser.add_argument("--n-identities", type=int, default=50, help="Number of synthetic gallery identities")
    parser.add_argument("--n-queries", type=int, default=200, help="Number of validation queries")
    parser.add_argument("--match-ratio", type=float, default=0.5, help="Fraction of queries that should be positive matches")
    parser.add_argument("--emb-dim", type=int, default=512, help="Embedding dimensionality")
    args = parser.parse_args()

    print("Starting synthetic evaluation run...")

    # 0. Clear DB
    _clear_database()

    # 1. Generate gallery
    print(f"Generating {args.n_identities} synthetic identities (emb_dim={args.emb_dim})...")
    gallery = generate_synthetic_gallery(n_identities=args.n_identities, emb_dim=args.emb_dim)
    print(f"Created {len(gallery)} identities in the database.")

    # 2. Initialize matcher (loads embeddings from DB)
    matcher = MatchingService(threshold=0.6, top_k=5)

    # 3. Create validation set
    queries = generate_validation_set(gallery, n_queries=args.n_queries, match_ratio=args.match_ratio)
    print(f"Prepared {len(queries)} validation queries (match ratio={args.match_ratio}).")

    # 4. Evaluate
    print("Running evaluation (this may take a moment)...")
    results = evaluate_matching(matcher, queries, topk_list=[1, 5])

    # 5. Print results
    print("\nEvaluation results:")
    print(f"Total queries: {results['total_queries']}")
    print(f"Top-1 identification rate: {results['topk'][1]*100:.2f}%")
    print(f"Top-5 identification rate: {results['topk'][5]*100:.2f}%")
    print(f"Precision: {results['precision']*100:.2f}%")
    print(f"Recall: {results['recall']*100:.2f}%")
    print(f"True positives: {results['TP']}, False positives: {results['FP']}, False negatives: {results['FN']}")
    print(f"Average matching latency: {results['avg_match_latency_ms']:.3f} ms per query")

    print("\nSynthetic evaluation complete.")
