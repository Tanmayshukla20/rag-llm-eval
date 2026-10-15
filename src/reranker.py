import torch

from sentence_transformers import CrossEncoder
<<<<<<< HEAD
from langsmith import traceable
=======

>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
from src.retriever import load_store


# small, fast, CPU-friendly reranker.
# Downloads once (~80MB) on first run.
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever:

    def __init__(self, fetch_k=10, top_k=5):

        self.store = load_store()

        # Use Apple's MPS GPU when available.
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        print("Reranker device:", self.device)

        self.reranker = CrossEncoder(
            CROSS_ENCODER,
            device=self.device
        )

        self.fetch_k = fetch_k
        self.top_k = top_k

<<<<<<< HEAD
    @traceable(run_type="retriever", name="RerankingRetriever")
=======
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
    def invoke(self, query):

        # 1. OVER-RETRIEVE:
        # Fast bi-encoder retrieves more candidates than we need.
        candidates = self.store.similarity_search(
            query,
            k=self.fetch_k
        )

        # 2. RERANK:
        # Score each (query, chunk) pair together
        # using the cross-encoder.
        pairs = [
            (query, doc.page_content)
            for doc in candidates
        ]

        scores = self.reranker.predict(
            pairs
        )

        # 3. SORT by cross-encoder score
        # and keep only the best top_k documents.
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc
            for doc, _ in ranked[:self.top_k]
        ]