from fastapi import FastAPI,  HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib  import asynccontextmanager
from sentence_transformers import SentenceTransformer

from retriever import load_resources, build_bm25_index, retrieve
from generator import generate
from cache import init_caches, cache_get, cache_set


class QuestionRequest(BaseModel):
    question: str
    category: str | None = None


class AnswerResponse(BaseModel):
    answer: str
    rule_ids: list[str]
    source: str | list[str] | None
    method: str
    used_llm: bool
    caveat: str | None
    cache_layer: str | None = None


app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("ChromaDB connection")
    model_chroma, collection = load_resources()

    print("Building bm25 index")
    bm25, all_docs, all_metas = build_bm25_index(collection)

    print("Cache init")
    exact_cache, semantic_cache = init_caches()

    app_state["model"] = model
    app_state["collection"] = collection
    app_state["bm25"] = bm25
    app_state["all_docs"]  = all_docs
    app_state["all_metas"] = all_metas
    app_state["exact_cache"] = exact_cache
    app_state["semantic_cache"] = semantic_cache

    print("Ready \n")
    yield

    exact_cache.close()
    semantic_cache.close()


app = FastAPI(
    title="BRACU Assistant",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    cached = cache_get(
        question,
        app_state["model"],
        app_state["exact_cache"],
        app_state["semantic_cache"]
    )
    if cached:
        return AnswerResponse(**cached)

    docs, metas, scores = retrieve(
        question,
        app_state["model"],
        app_state["collection"],
        app_state["bm25"],
        app_state["all_docs"],
        app_state["all_metas"]
    )

    result = generate(question, docs, metas, scores)

    cache_set(
        question,
        result,
        app_state["model"],
        app_state["exact_cache"],
        app_state["semantic_cache"]
    )

    return AnswerResponse(**result)


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "docs_indexed": app_state["collection"].count(),
        "cache_size":   len(app_state["exact_cache"])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)