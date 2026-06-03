import os
import glob
import frontmatter
import chromadb
from sentence_transformers import SentenceTransformer


def load_doc(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        doc = frontmatter.load(f)

    return {
        'rule_id': doc.metadata.get('rule_id', "UNKNOWN"),
        'title': doc.metadata.get('title', os.path.basename(file_path)),
        'category': doc.metadata.get('category', "general"),
        'tags': doc.metadata.get('tags', []),
        'aliases': doc.metadata.get('aliases', []),
        'severity': doc.metadata.get('severity', "normal"),
        'content': doc.content.strip(),
        'filepath': file_path
    }

def chunk_doc(doc):
    content = doc['content']
    chunks = []

    sections = content.split("\n## ")

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        
        lines = section.split("\n")
        section_title = lines[0].replace('#', '').strip() if i>0 else "Overview"
        section_body = "\n".join(lines[1:]).strip() if i>0 else section

        if not section_body:
            continue

        chunks.append({
            'chunk_id': f"{doc['rule_id']}_{i}",
            'text': f"{doc['title']} - {section_title}\n\n{section_body}",
            'rule_id': doc["rule_id"],
            'title': doc['title'],
            'category': doc["category"],
            'tags': ', '.join(doc['tags']),
            'severity': doc['severity'],
            'section': section_title,
            'filepath': doc['filepath']
        })
    return chunks


def get_vector_store(chroma_dir= "./chroma_db"):
    client = chromadb.PersistentClient(path=chroma_dir)

    collection = client.get_or_create_collection(
        name="uni_info",
        metadata={"hnsw:space":"cosine"}
    )

    return collection


def embed_and_store(chunks, collection, model):
    if not chunks:
        return

    texts = []
    ids = []
    metadatas = []
    
    for c in chunks:
        texts.append(c['text'])
        ids.append(c['chunk_id'])
        metadatas.append({
            "rule_id": c['rule_id'],
            "title":     c['title'],
            "category":  c['category'],
            "tags":      c['tags'],
            "severity":  c['severity'],
            "section":   c['section'],
            "filepath":  c['filepath'],
        })

    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.upsert(
        ids = ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} chunks")


def ingest_all(docs_dir):
    model =  SentenceTransformer("all-MiniLM-L6-v2")

    collection = get_vector_store()

    md_files = glob.glob(os.path.join(docs_dir, '**/*.md'), recursive=True)

    for filepath in md_files:
        doc = load_doc(filepath)
        chunks = chunk_doc(doc)

        embed_and_store(chunks, collection, model)

    print(f"Collection Size: {collection.count()} vectors")


if __name__=="__main__":
    ingest_all("./docs")