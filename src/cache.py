import hashlib
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import diskcache

def init_caches(cache_dir="./cache"):
    exact_cache = diskcache.Cache(f"{cache_dir}/exact")
    semantic_cache =  diskcache.Cache(f"{cache_dir}/semantic")

    return exact_cache, semantic_cache


def normalize_question(question):
    return question.lower().strip().rstrip("?!.") # might need more work here


def exact_cache_get(question, cache):
    key =  hashlib.md5(normalize_question(question).encode()).hexdigest()
    result = cache.get(key)

    if result:
        print(f"[Cache] Exact hit for: '{question}'") # convert to log
        return json.loads(result)

    return None

def exact_cache_set(question, answer, cache, ttl_days=180):
    key = hashlib.md5(normalize_question(question).encode()).hexdigest()
    cache.set(key, json.dumps(answer), expire=ttl_days*86400)


def semantic_cache_get(question, model, cache, threshold=0.93):
    entries = cache.get("entries", [])

    if not entries:
        return None

    question_embedding = model.encode(question)

    for entry in entries:
        cache_embedding = np.array(entry["embedding"])

        similarity = np.dot(question_embedding, cache_embedding / 
                    np.linalg.norm(question_embedding)*np.linalg.norm(cache_embedding))

        if similarity >= threshold:
            print(f"[Cache] Semantic hit (similarity={similarity:.3f})")
            print(f"[Cache] Matched: '{entry['question']}'")
            return entry["answer"]

    return None


def semantic_cache_set(question, answer, model, cache):
    entries = cache.get("entries", [])

    entries.append({
        "question": question,
        "embedding": model.encode(question).tolist(),
        "answer":  answer
    })

    cache.set("entries", entries)


def cache_get(question, model, exact_cache, semantic_cache):
    result = exact_cache_get(question, exact_cache)
    if result:
        result["cache_layer"] = "exact"
        return result

    result = semantic_cache_get(question, model, semantic_cache)
    if result:
        result["cache_layer"] = "semantic"
        return result

    return None


def cache_set(question, answer, model, exact_cache, semantic_cache):
    exact_cache_set(question, answer, exact_cache)
    semantic_cache_set(question, answer, model, semantic_cache)