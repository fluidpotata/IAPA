import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM250kapi
import numpy as np


def load_resources(persist_dir):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection("uni_info")

    return model, collection


def build_bm25_index(collection):
    results = collection.get(include=['documents', 'metadatas'])

    docs = results['documents']
    metadatas = results['metadatas']

    tokenized = [doc.lower().split() for doc in docs]

    bm25 = BM250kapi(tokenized)

    return bm25, docs, metadatas


def semantic_search(question, model, collection, category, top_k):
    question_embedding = model.encode(question).tolist()

    where = {"category": category} if category else None

    results = collection.query(
        query_embeddings = [question_embedding],
        n_result = top_k,
        where = where,
        include=['documents','metadatas','distances']
    )

    docs =  results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return docs, metadatas, distances


def bm25_search(question, bm25, all_docs, all_metadatas, top_k):
    tokenized_question =  question.lower().split()
    scores = bm25.get_scores(tokenized_question)

    top_indices =  np.argsort(scores)[::-1][:top_k]

    docs = [all_docs[i] for i in top_indices]
    metadatas = [all_metadatas[i] for i in top_indices]
    scores = [scores[i] for i in top_indices]

    return docs, metadatas, scores


def reciprocal_rank_fusion(semantic_docs, semantic_meta, bm25_docs, bm25_meta, top_k):
    scores = {}
    chunks = {}

    for rank, (doc, meta) in enumerate(zip(semantic_docs, semantic_meta)):
        chunk_id = meta['rule_id'] + "_"+meta["section"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rank+60)
        chunks[chunk_id] = (doc, meta)

    for rank, (doc, meta) in enumerate(zip(bm25_docs, bm25_data)):
        chunk_id = meta["rule_id"] + "_"+meta["section"]
        scores[chunk_id] = scores.get(chunk_id, 0)+1 / (rank+60)
        chunks[chunk_id] = (doc, meta)

    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

    final_docs = [chunks[i][0] for i in sorted_ids]
    final_metas = [chunks[i][1] for i in sorted_ids]

    return final_docs, final_metas


def detect_category(question):
    return None


def retrieve(question, model, collection, bm25, all_docs, all_metadatas, top_k):
    category = detect_category(question)

    sem_docs, sem_meta, _ = semantic_search(question, model, collection, category)
    bm25_docs, bm25_meta, _ = bm25_search(question, bm25, all_docs, all_metadatas)

    final_docs, final_metas = reciprocal_rank_fusion(
        sem_docs, sem_meta, bm25_docs, bm25_meta, top_k
    )

    return final_docs, final_metas



if __name__=="__main__":
    model, collection = load_resources()
    bm25, all_docs, all_metas = build_bm25_index(collection)

    sample_ques = "When do I get probation?"

    docs, metas = retrieve(question, model, collection, bm25, all_docs, all_metas)

    for i, (doc, meta) in enumerate(zip(docs, metas)):
        print(f"\n--- Result {i+1} ---")
        print(f"Rule: {meta['title']} | Section: {meta['section']}")
        print(doc[:300])