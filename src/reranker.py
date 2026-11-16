# # import torch

# # from sentence_transformers import CrossEncoder
# # from langsmith import traceable
# # from src.retriever import load_store


# # # small, fast, CPU-friendly reranker.
# # # Downloads once (~80MB) on first run.
# # CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# # class RerankingRetriever:

# #     def __init__(self, fetch_k=10, top_k=5):

# #         self.store = load_store()

# #         # Use Apple's MPS GPU when available.
# #         self.device = (
# #             "mps"
# #             if torch.backends.mps.is_available()
# #             else "cpu"
# #         )

# #         print("Reranker device:", self.device)

# #         self.reranker = CrossEncoder(
# #             CROSS_ENCODER,
# #             device=self.device
# #         )

# #         self.fetch_k = fetch_k
# #         self.top_k = top_k

# #     @traceable(run_type="retriever", name="RerankingRetriever")
# #     def invoke(self, query):

# #         # 1. OVER-RETRIEVE:
# #         # Fast bi-encoder retrieves more candidates than we need.
# #         candidates = self.store.similarity_search(
# #             query,
# #             k=self.fetch_k
# #         )

# #         # 2. RERANK:
# #         # Score each (query, chunk) pair together
# #         # using the cross-encoder.
# #         pairs = [
# #             (query, doc.page_content)
# #             for doc in candidates
# #         ]

# #         scores = self.reranker.predict(
# #             pairs
# #         )

# #         # 3. SORT by cross-encoder score
# #         # and keep only the best top_k documents.
# #         ranked = sorted(
# #             zip(candidates, scores),
# #             key=lambda x: x[1],
# #             reverse=True
# #         )

# #         return [
# #             doc
# #             for doc, _ in ranked[:self.top_k]
# #         ]











# #Reranker            correct 

# import torch
# from sentence_transformers import CrossEncoder
# from langsmith import traceable
# from src.retriever import load_store, FETCH_K, TOP_K

# CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# class RerankingRetriever:

#     def __init__(self, fetch_k=FETCH_K, top_k=TOP_K):
#         self.store = load_store()
#         device = "mps" if torch.backends.mps.is_available() else "cpu"
#         print("Reranker device:", device)

#         self.reranker = CrossEncoder(CROSS_ENCODER, device=device)
#         self.fetch_k = fetch_k
#         self.top_k = top_k

#     @traceable(run_type="retriever", name="RerankingRetriever")
#     def invoke(self, query):
#         candidates = self.store.similarity_search(query, k=self.fetch_k)
#         pairs = [(query, doc.page_content) for doc in candidates]
#         scores = self.reranker.predict(pairs)

#         ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
#         return [doc for doc, _ in ranked[:self.top_k]]











import torch

from sentence_transformers import CrossEncoder
from langsmith import traceable
from langchain_ollama import ChatOllama

try:
    from langchain_classic.retrievers import MultiQueryRetriever
except ImportError:
    from langchain.retrievers.multi_query import MultiQueryRetriever

from src.retriever import (
    load_store,
    FETCH_K,
    TOP_K
)


# =========================================================
# CONFIGURATION
# =========================================================

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MULTI_QUERY_MODEL = "qwen2.5:3b"

MULTI_QUERY_CONTEXT_SIZE = 4096


# =========================================================
# MULTI-QUERY + CROSS-ENCODER RERANKER
# =========================================================

class RerankingRetriever:

    def __init__(self, fetch_k=FETCH_K, top_k=TOP_K):

        self.fetch_k = fetch_k
        self.top_k = top_k

        # Load Chroma vector database.
        self.store = load_store()

        # Select device for cross-encoder.
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        print("Reranker device:", device)

        self.reranker = CrossEncoder(
            CROSS_ENCODER,
            device=device
        )

        # Local Ollama model for generating query variations.
        self.query_llm = ChatOllama(
            model=MULTI_QUERY_MODEL,
            temperature=0,
            num_ctx=MULTI_QUERY_CONTEXT_SIZE
        )

        # Original vector retriever.
        base_retriever = self.store.as_retriever(
            search_kwargs={
                "k": self.fetch_k
            }
        )

        # Multi-Query Retriever.
        self.multi_query_retriever = (
            MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=self.query_llm,
                include_original=True
            )
        )

        print("Multi-Query Retriever initialized.")
        print("Fetch K per query:", self.fetch_k)
        print("Final Top K:", self.top_k)

    # =====================================================
    # RETRIEVAL + RERANKING
    # =====================================================

    @traceable(
        run_type="retriever",
        name="MultiQueryRerankingRetriever"
    )
    def invoke(self, query):

        # Step 1: Retrieve documents using multiple queries.
        candidates = self.multi_query_retriever.invoke(
            query
        )

        print(
            "\nMulti-query candidate count:",
            len(candidates)
        )

        if not candidates:
            return []

        # Step 2: Remove duplicate documents.
        unique_candidates = []
        seen = set()

        for doc in candidates:

            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("session", ""),
                doc.page_content
            )

            if key not in seen:

                seen.add(key)
                unique_candidates.append(doc)

        print(
            "Unique candidate count:",
            len(unique_candidates)
        )

        # Step 3: Compare every candidate with the original query.
        pairs = [
            (query, doc.page_content)
            for doc in unique_candidates
        ]

        # Step 4: Calculate cross-encoder relevance scores.
        scores = self.reranker.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False
        )

        # Step 5: Rank documents by relevance.
        ranked = sorted(
            zip(unique_candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        # Step 6: Return the final top K documents.
        final_results = [
            doc
            for doc, score in ranked[:self.top_k]
        ]

        print(
            "Final documents returned:",
            len(final_results)
        )

        return final_results
