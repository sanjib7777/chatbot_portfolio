import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from redis_client import redis_client, REDIS_TTL

SIM_THRESHOLD = 0.90


def normalize(text: str):
    return text.lower().strip()


def get_semantic_cache(session_id: str, query: str, embedder):
    query_vec = embedder.embed_query(query)

    keys = redis_client.keys(f"sem:{session_id}:*")

    for k in keys:
        data = json.loads(redis_client.get(k))
        cached_vec = data["embedding"]

        sim = cosine_similarity([query_vec], [cached_vec])[0][0]
        if sim >= SIM_THRESHOLD:
            return data["answer"]

    return None


def set_semantic_cache(session_id: str, query: str, answer: str, embedder):
    vec = embedder.embed_query(query)

    key = f"sem:{session_id}:{hash(normalize(query))}"

    redis_client.setex(
        key,
        REDIS_TTL,
        json.dumps({
            "question": query,
            "embedding": vec,
            "answer": answer
        })
    )
