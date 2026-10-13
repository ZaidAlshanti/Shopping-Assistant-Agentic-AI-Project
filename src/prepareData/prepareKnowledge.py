
import hashlib
import uuid
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from FlagEmbedding import BGEM3FlagModel
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient, models

load_dotenv(find_dotenv(usecwd=True))

COLLECTION = "shopassist_knowledge_v2"
VECTOR_SIZE = 1024

# Set True ONCE if you need to recreate the collection.
# WARNING: this deletes all existing points in this collection.
REBUILD = False

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data"

# Nomic is used ONLY to determine semantic chunk boundaries.
semantic_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)

chunker = SemanticChunker(
    embeddings=semantic_embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85,
    min_chunk_size=200,
    buffer_size=1,
)

# BGE-M3 generates the actual dense and learned sparse vectors.
_bge = None


def get_bge():
    global _bge
    if _bge is None:
        _bge = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False,
        )
    return _bge


client = QdrantClient(url="http://localhost:6333")


def ensure_collection(rebuild=False):
    if rebuild and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    # BGE-M3 learned sparse weights are not raw BM25.
                    # Do not apply Qdrant's IDF modifier to these weights.
                    modifier=models.Modifier.NONE,
                ),
            },
        )

        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="source",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def ingest_text(text: str, source: str) -> int:
    ensure_collection()

    chunks = [
        doc.page_content.strip()
        for doc in chunker.create_documents([text])
        if doc.page_content.strip()
    ]

    if not chunks:
        print(f"No chunks generated for {source}")
        return 0

    # Generate both dense and sparse representations in one pass.
    encoded = get_bge().encode(
        chunks,
        batch_size=16,
        max_length=1024,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_vectors = encoded["dense_vecs"]
    sparse_vectors = encoded["lexical_weights"]

    points = []

    for i, (chunk, dense, sparse) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors)
    ):
        # Include neighboring chunks as additional context.
        parent = " ".join(chunks[max(0, i - 1): i + 2])

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                source + hashlib.md5(chunk.encode("utf-8")).hexdigest(),
            )
        )

        sparse_indices = [int(token_id) for token_id in sparse.keys()]
        sparse_values = [float(weight) for weight in sparse.values()]

        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense.tolist(),
                    "bm25": models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                },
                payload={
                    "text": chunk,
                    "parent_text": parent,
                    "source": source,
                },
            )
        )

    # Replace this source's previous chunks only after embedding succeeds.
    client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source),
                    )
                ]
            )
        ),
        wait=True,
    )

    client.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True,
    )

    print(f"{source}: indexed {len(points)} chunks")
    return len(points)


if __name__ == "__main__":
    ensure_collection(rebuild=REBUILD)

    for file in sorted(DATA_DIR.glob("*")):
        if file.suffix.lower() in {".txt", ".md"}:
            content = file.read_text(encoding="utf-8-sig")
            ingest_text(content, file.name)
