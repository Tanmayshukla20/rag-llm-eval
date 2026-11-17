# # import os
# # import re
# # import glob

# # from dotenv import load_dotenv
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain_chroma import Chroma
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_core.documents import Document

# # load_dotenv()

# # DATA_DIR = "data"
# # DB_DIR = "chroma_store"


# # # 1. LOAD ---- read each transcript, throw away the VTT timestamps
# # def load_transcripts():

# #     docs = []
# #     for path in glob.glob(f"{DATA_DIR}/*.vtt"):
# #         lines = []
# #         for line in open(path):
# #             line = line.strip()
# #             if not line or line == "WEBVTT" or "-->" in line:
# #                 continue
# #             lines.append(line)
# #         text = " ".join(lines)

# #         session = re.search(r"Session[ _]*(\d+)", path).group(1)

# #         docs.append(Document(page_content=text, metadata={"session": session}))

# #     return docs


# # # 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
# # def load_store():

# #     embeddings = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2")

# #     if os.path.exists(DB_DIR):
# #         return Chroma(
# #             persist_directory=DB_DIR,
# #             embedding_function=embeddings
# #         )

# #     docs = load_transcripts()

# #     chunks = RecursiveCharacterTextSplitter(
# #         chunk_size=800,
# #         chunk_overlap=100,
# #     ).split_documents(docs)

# #     return Chroma.from_documents(
# #         chunks,
# #         embeddings,
# #         persist_directory=DB_DIR
# #     )


# # def build_retriever():
# #     return load_store().as_retriever(search_kwargs={"k": 5})


# # # 3. TRY IT ---- python src/retriever.py
# # if __name__ == "__main__":

# #     retriever = build_retriever()


# #     results = retriever.invoke("what is regression testing?") #since retriever is a runnable hence it can be invoked

    
# #     for r in results:
# #         print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")













# import os
# import re
# import glob

# from dotenv import load_dotenv
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

# load_dotenv()

# DATA_DIR = "data"
# DB_DIR = "chroma_store_minilm"

# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 100
# FETCH_K = 20
# TOP_K = 5
# THRESHOLD = 0.7


# def load_transcripts():
#     docs = []

#     for path in sorted(glob.glob(f"{DATA_DIR}/*.vtt")):
#         with open(path, encoding="utf-8") as f:
#             lines = [
#                 line.strip() for line in f
#                 if line.strip()
#                 and line.strip() != "WEBVTT"
#                 and "-->" not in line
#                 and not line.strip().isdigit()
#             ]

#         if not lines:
#             continue

#         filename = os.path.basename(path)
#         match = re.search(r"Session[ _]*(\d+)", filename, re.I)
#         session = match.group(1) if match else filename

#         docs.append(Document(
#             page_content=" ".join(lines),
#             metadata={"session": session}
#         ))

#     if not docs:
#         raise ValueError(f"No usable VTT files found in {DATA_DIR}")

#     return docs


# def load_store():
#     embeddings = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL
#     )

#     if os.path.exists(DB_DIR):
#         return Chroma(
#             persist_directory=DB_DIR,
#             embedding_function=embeddings
#         )

#     chunks = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP
#     ).split_documents(load_transcripts())

#     print("Chunks created:", len(chunks))

#     return Chroma.from_documents(
#         chunks,
#         embeddings,
#         persist_directory=DB_DIR
#     )


# def build_retriever():
#     return load_store().as_retriever(
#         search_kwargs={"k": TOP_K}
#     )


# if __name__ == "__main__":
#     for doc in build_retriever().invoke("What is regression testing?"):
#         print(f"[Session {doc.metadata['session']}] {doc.page_content[:150]}...\n")












import os
import re
import glob

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


load_dotenv()


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = "data"

# New database for the semantic chunking experiment.
# The previous chroma_store_minilm database is preserved.
DB_DIR = "chroma_store"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Maximum size for the secondary recursive splitter.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100

FETCH_K = 20
TOP_K = 5
THRESHOLD = 0.7

SEMANTIC_BREAKPOINT_TYPE = "percentile"
SEMANTIC_BREAKPOINT_AMOUNT = 95


# =========================================================
# LOAD TRANSCRIPTS
# =========================================================

def load_transcripts():

    docs = []

    paths = sorted(
        glob.glob(os.path.join(DATA_DIR, "*.vtt"))
    )

    for path in paths:

        with open(path, "r", encoding="utf-8") as file:

            lines = [
                line.strip()
                for line in file
                if line.strip()
                and line.strip() != "WEBVTT"
                and "-->" not in line
                and not line.strip().isdigit()
            ]

        if not lines:
            continue

        filename = os.path.basename(path)

        match = re.search(
            r"Session[ _]*(\d+)",
            filename,
            re.IGNORECASE
        )

        session = match.group(1) if match else filename

        docs.append(
            Document(
                page_content=" ".join(lines),
                metadata={
                    "session": session,
                    "source": filename
                }
            )
        )

    if not docs:

        raise ValueError(
            f"No usable VTT files found in '{DATA_DIR}'."
        )

    print("Transcript files loaded:", len(docs))

    return docs


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# =========================================================
# SEMANTIC CHUNKING
# =========================================================

def create_semantic_chunks(documents, embeddings):

    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT
    )

    semantic_chunks = semantic_splitter.split_documents(
        documents
    )

    print(
        "Semantic chunks created:",
        len(semantic_chunks)
    )

    # Secondary safeguard for oversized semantic chunks.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    final_chunks = []

    for doc in semantic_chunks:

        if len(doc.page_content) > CHUNK_SIZE:

            smaller_chunks = recursive_splitter.split_documents(
                [doc]
            )

            final_chunks.extend(smaller_chunks)

        else:

            final_chunks.append(doc)

    print(
        "Final chunks after length safeguard:",
        len(final_chunks)
    )

    return final_chunks


# =========================================================
# CREATE OR LOAD CHROMA DATABASE
# =========================================================

def load_store():

    embeddings = load_embeddings()

    # Reuse the database if it already contains vectors.
    if os.path.isdir(DB_DIR):

        print("Checking existing Chroma database:", DB_DIR)

        existing_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

        if existing_store._collection.count() > 0:

            print(
                "Existing vectors found:",
                existing_store._collection.count()
            )

            return existing_store

        print("Existing database is empty. Rebuilding it.")

    print("Creating semantic Chroma database...")

    documents = load_transcripts()

    chunks = create_semantic_chunks(
        documents,
        embeddings
    )

    if not chunks:

        raise ValueError(
            "Semantic chunking did not generate any chunks."
        )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print("Vectors stored in:", DB_DIR)

    print(
        "Total vectors stored:",
        vector_store._collection.count()
    )

    return vector_store


# =========================================================
# BASIC VECTOR RETRIEVER
# =========================================================

def build_retriever():

    vector_store = load_store()

    return vector_store.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )


# =========================================================
# TEST RETRIEVER
# =========================================================

if __name__ == "__main__":

    retriever = build_retriever()

    query = "What is regression testing?"

    results = retriever.invoke(query)

    print("\nQuery:", query)
    print("Retrieved documents:", len(results))

    for rank, doc in enumerate(results, start=1):

        print(f"\nRANK {rank}")

        print(
            "SESSION:",
            doc.metadata.get("session")
        )

        print(
            "SOURCE:",
            doc.metadata.get("source")
        )

        print(
            "CONTENT:",
            doc.page_content[:500]
        )
