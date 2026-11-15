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
# DB_DIR = "chroma_store"


# # 1. LOAD ---- read each transcript, throw away the VTT timestamps
# def load_transcripts():

#     docs = []
#     for path in glob.glob(f"{DATA_DIR}/*.vtt"):
#         lines = []
#         for line in open(path):
#             line = line.strip()
#             if not line or line == "WEBVTT" or "-->" in line:
#                 continue
#             lines.append(line)
#         text = " ".join(lines)

#         session = re.search(r"Session[ _]*(\d+)", path).group(1)

#         docs.append(Document(page_content=text, metadata={"session": session}))

#     return docs


# # 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
# def load_store():

#     embeddings = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2")

#     if os.path.exists(DB_DIR):
#         return Chroma(
#             persist_directory=DB_DIR,
#             embedding_function=embeddings
#         )

#     docs = load_transcripts()

#     chunks = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=100,
#     ).split_documents(docs)

#     return Chroma.from_documents(
#         chunks,
#         embeddings,
#         persist_directory=DB_DIR
#     )


# def build_retriever():
#     return load_store().as_retriever(search_kwargs={"k": 5})


# # 3. TRY IT ---- python src/retriever.py
# if __name__ == "__main__":

#     retriever = build_retriever()


#     results = retriever.invoke("what is regression testing?") #since retriever is a runnable hence it can be invoked

    
#     for r in results:
#         print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")







# import os
# import re
# import glob
# import json

# import torch
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document


# DATA_DIR = "data"
# DB_DIR = "chroma_store"
# EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
# CHUNK_SIZE = 800 #600
# CHUNK_OVERLAP = 150 #100
# THRESHOLD = 0.7

# FETCH_K = 25
# TOP_K = 5


# def load_transcripts():
#     docs = []

#     for path in sorted(glob.glob(f"{DATA_DIR}/*.vtt")):
#         with open(path, encoding="utf-8") as f:
#             content = f.read()

#         paragraphs, timestamps = [], []

#         for block in re.split(r"\n\s*\n", content):
#             if "-->" not in block:
#                 continue

#             times = re.search(
#                 r"(\d{2}:\d{2}:\d{2}\.\d+)\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d+)",
#                 block
#             )
#             if not times:
#                 continue

#             timestamps.append(times.group(1) + " --> " + times.group(2))

#             lines = [
#                 line.strip() for line in block.splitlines()
#                 if line.strip() and "-->" not in line
#                 and not line.strip().isdigit()
#             ]

#             if lines:
#                 paragraphs.append(" ".join(lines))

#         session = re.search(r"Session[ _]*(\d+)", os.path.basename(path))
#         docs.append(Document(
#             page_content="\n\n".join(paragraphs),
#             metadata={
#                 "session": session.group(1) if session else os.path.basename(path),
#                 "timestamp_start": timestamps[0] if timestamps else "",
#                 "timestamp_end": timestamps[-1] if timestamps else "",
#                 "cue_timestamps": json.dumps(timestamps)
#             }
#         ))

#     return docs


# def load_store():
#     embeddings = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-base-en-v1.5",
#         model_kwargs={
#             "device": "mps" if torch.backends.mps.is_available() else "cpu"
#         },
#         encode_kwargs={"normalize_embeddings": True}
#     )

#     if os.path.exists(DB_DIR):
#         return Chroma(
#             persist_directory=DB_DIR,
#             embedding_function=embeddings
#         )

#     chunks = RecursiveCharacterTextSplitter(
#         chunk_size=600,
#         chunk_overlap=100,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     ).split_documents(load_transcripts())

#     return Chroma.from_documents(
#         chunks,
#         embeddings,
#         persist_directory=DB_DIR
#     )


# def build_retriever():
#     return load_store().as_retriever(search_kwargs={"k": 20})


# if __name__ == "__main__":
#     retriever = build_retriever()

#     results = retriever.invoke("what is regression testing?")

#     for r in results:
#         print(
#             f"[Session {r.metadata['session']}] "
#             f"[{r.metadata['timestamp_start']} - "
#             f"{r.metadata['timestamp_end']}] "
#             f"{r.page_content[:150]}...\n"
#         )
        
















import os
import re
import glob

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_store_minilm"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
FETCH_K = 20
TOP_K = 5
THRESHOLD = 0.7


def load_transcripts():
    docs = []

    for path in sorted(glob.glob(f"{DATA_DIR}/*.vtt")):
        with open(path, encoding="utf-8") as f:
            lines = [
                line.strip() for line in f
                if line.strip()
                and line.strip() != "WEBVTT"
                and "-->" not in line
                and not line.strip().isdigit()
            ]

        if not lines:
            continue

        filename = os.path.basename(path)
        match = re.search(r"Session[ _]*(\d+)", filename, re.I)
        session = match.group(1) if match else filename

        docs.append(Document(
            page_content=" ".join(lines),
            metadata={"session": session}
        ))

    if not docs:
        raise ValueError(f"No usable VTT files found in {DATA_DIR}")

    return docs


def load_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    if os.path.exists(DB_DIR):
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    ).split_documents(load_transcripts())

    print("Chunks created:", len(chunks))

    return Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIR
    )


def build_retriever():
    return load_store().as_retriever(
        search_kwargs={"k": TOP_K}
    )


if __name__ == "__main__":
    for doc in build_retriever().invoke("What is regression testing?"):
        print(f"[Session {doc.metadata['session']}] {doc.page_content[:150]}...\n")
