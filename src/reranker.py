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











#Reranker            correct 

import torch
from sentence_transformers import CrossEncoder
from langsmith import traceable
from src.retriever import load_store, FETCH_K, TOP_K

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever:

    def __init__(self, fetch_k=FETCH_K, top_k=TOP_K):
        self.store = load_store()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print("Reranker device:", device)

        self.reranker = CrossEncoder(CROSS_ENCODER, device=device)
        self.fetch_k = fetch_k
        self.top_k = top_k

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query):
        candidates = self.store.similarity_search(query, k=self.fetch_k)
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.top_k]]









# #Reranker with MMR 
    
# import torch
# from sentence_transformers import CrossEncoder
# from langsmith import traceable

# from src.retriever import load_store, FETCH_K, TOP_K

# CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# class RerankingRetriever:

#     def __init__(
#         self,
#         fetch_k=FETCH_K,
#         top_k=TOP_K,
#         mmr_fetch_k=40,
#         mmr_lambda=0.7
#     ):
#         self.store = load_store()

#         device = "mps" if torch.backends.mps.is_available() else "cpu"
#         print("Reranker device:", device)

#         self.reranker = CrossEncoder(CROSS_ENCODER, device=device)

#         self.fetch_k = fetch_k
#         self.top_k = top_k
#         self.mmr_fetch_k = mmr_fetch_k
#         self.mmr_lambda = mmr_lambda

#     @traceable(run_type="retriever", name="MMR_RerankingRetriever")
#     def invoke(self, query):

#         # Step 1: Retrieve diverse candidate chunks using MMR
#         candidates = self.store.max_marginal_relevance_search(
#             query,
#             k=self.fetch_k,
#             fetch_k=self.mmr_fetch_k,
#             lambda_mult=self.mmr_lambda
#         )

#         # Step 2: Rerank candidates using cross-encoder
#         pairs = [
#             (query, doc.page_content)
#             for doc in candidates
#         ]

#         scores = self.reranker.predict(pairs)

#         ranked = sorted(
#             zip(candidates, scores),
#             key=lambda x: x[1],
#             reverse=True
#         )

#         # Step 3: Return the top-ranked chunks
#         return [
#             doc for doc, score in ranked[:self.top_k]
#         ]


















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
#         docs = self.store.similarity_search(query, k=self.fetch_k)

#         if not docs:
#             return []

#         scores = self.reranker.predict([
#             (query, doc.page_content) for doc in docs
#         ])

#         ranked = sorted(
#             zip(docs, scores),
#             key=lambda x: float(x[1]),
#             reverse=True
#         )

#         return [doc for doc, _ in ranked[:self.top_k]]
