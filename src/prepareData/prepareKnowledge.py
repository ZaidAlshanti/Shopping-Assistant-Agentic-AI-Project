import hashlib
import uuid
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from FlagEmbedding import BGEM3FlagModel
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from langchain_ollama import OllamaEmbeddings
import ollama

load_dotenv(find_dotenv(usecwd=True))

COLLECTION = "shopassist_knowledge_v1"
VECTOR_SIZE = 1024                       # bge-m3 dense vectors
REBUILD = False                          # True once to wipe and recreate the collection

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data"

# OpenAI: decides split points only
# chunker = SemanticChunker(
#     OpenAIEmbeddings(model="text-embedding-3-small"),
#     breakpoint_threshold_type="percentile",
#     breakpoint_threshold_amount=85,      # lower = fewer, larger chunks (default 95)
#     min_chunk_size=200,                  # characters; merges tiny fragments
# )

# chunker = SemanticChunker(
#     embeddings=ollama.embed(model="qwen3.5:9b "),
#     breakpoint_threshold_type="percentile",
#     breakpoint_threshold_amount=85,      # lower = fewer, larger chunks (default 95)
#     min_chunk_size=200             # characters; merges tiny fragments
# )

semantic_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

chunker = SemanticChunker(
    semantic_embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85,
    min_chunk_size=200,
    buffer_size=2
)
# BGE-M3: produces the vectors stored in Qdrant

client = QdrantClient(url="http://localhost:6333")

_bge = None

def get_bge():
    global _bge
    if _bge is None:
        _bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    return _bge


def ensure_collection(rebuild: bool = False):
    if rebuild and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        client.create_payload_index(COLLECTION, "source", models.PayloadSchemaType.KEYWORD)


def ingest_text(text: str, source: str) -> int:
    ensure_collection()
    chunks = [c.page_content for c in chunker.create_documents([text]) if c.page_content.strip()]
    if not chunks:
        return 0

    vectors = get_bge().encode(chunks, batch_size=16, max_length=1024)["dense_vecs"]

    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        parent = " ".join(chunks[max(i - 1, 0): i + 2])
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, source + hashlib.md5(chunk.encode()).hexdigest()))
        points.append(models.PointStruct(
            id=pid, vector=vec.tolist(),
            payload={"text": chunk, "parent_text": parent, "source": source},
        ))

    # embedding succeeded, so now replace this file's older chunks and add the new ones
    client.delete(
        COLLECTION,
        points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="source", match=models.MatchValue(value=source))
        ])),
    )
    client.upsert(COLLECTION, points=points)
    print(f"{source}: {len(points)} chunks")
    return len(points)


if __name__ == "__main__":
    ensure_collection(rebuild=REBUILD)
    for f in sorted(DATA_DIR.glob("*")):
        if f.suffix.lower() in {".txt", ".md"}:
            ingest_text(f.read_text(encoding="utf-8-sig"), f.name)