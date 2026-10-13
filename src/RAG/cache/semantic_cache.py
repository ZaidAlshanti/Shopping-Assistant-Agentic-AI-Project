import json, time
import numpy as np
import ollama
from pathlib import Path

class SemanticCache:
    def __init__(self, path=".agents/query_cache.json", threshold=0.95,
                 ttl=7 * 24 * 3600, max_entries=500):
        self.path, self.threshold, self.ttl, self.max = Path(path), threshold, ttl, max_entries
        self.items = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    @staticmethod
    def _embed(q: str) -> np.ndarray:
        v = np.array(ollama.embed(model="nomic-embed-text", input=q)["embeddings"][0], dtype=np.float32)
        return v / np.linalg.norm(v)

    def get(self, q: str):
        now = time.time()
        self.items = [i for i in self.items if now - i["t"] < self.ttl]
        if not self.items:
            return None
        v = self._embed(q)
        sims = np.array([np.dot(v, np.array(i["v"], dtype=np.float32)) for i in self.items])
        best = int(sims.argmax())
        if sims[best] >= self.threshold:
            return self.items[best]["a"]
        return None

    def put(self, q: str, answer: str):
        self.items.append({"q": q, "a": answer, "t": time.time(), "v": self._embed(q).tolist()})
        self.items = self.items[-self.max:]
        self.path.write_text(json.dumps(self.items, ensure_ascii=False), encoding="utf-8")

    def clear(self):  # call this after re-ingesting the knowledge base
        self.items = []
        self.path.unlink(missing_ok=True)