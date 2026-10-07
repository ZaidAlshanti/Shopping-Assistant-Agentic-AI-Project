import os
import uuid
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from prepareChunks import get_all_chunks

def main():
    # 1. Fetch the data using the imported function
    print("Loading documents...")
    docs = get_all_chunks()
    print(f"Loaded {len(docs)} chunks successfully.")

    # 2. Initialize Qdrant and the Embedding Model
    print("Initializing embedding model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    client = QdrantClient(url="http://localhost:6333")
    COLLECTION_NAME = "shopassist_knowledge"

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE
        )
    )

    # 3. Embed and Upsert
    print("Generating embeddings...")
    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts, batch_size=16, max_length=512)["dense_vecs"]

    points = []
    for doc, vector in zip(docs, embeddings):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "text": doc["text"],
                    "source": doc["source"],
                    "section": doc["section"],
                    "type": doc["type"]
                }
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Successfully upserted {len(points)} chunks into Qdrant!")

if __name__ == "__main__":
    main()