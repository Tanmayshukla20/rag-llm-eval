



# import json
# from json_repair import loads as repair_json
# from deepeval import evaluate
# from deepeval.evaluate import AsyncConfig
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
# from deepeval.models import DeepEvalBaseLLM
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# from src.reranker import RerankingRetriever
# from src.retriever import FETCH_K, TOP_K
# from evals.harness import summarize_by_metric


# GOLDEN_PATH = "goldens/retriever_goldens.json"
# JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# CHUNK_SIZE = 600
# CHUNK_OVERLAP = 100
# FETCH_K = 20
# TOP_K = 5
# THRESHOLD = 0.7


# class LocalQwenLLM(DeepEvalBaseLLM):

#     def __init__(self):
#         self.model_name = JUDGE_MODEL
#         self.device = "mps" if torch.backends.mps.is_available() else "cpu"
#         self.dtype = torch.float16 if self.device == "mps" else torch.float32

#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name, torch_dtype=self.dtype
#         ).to(self.device)

#         self.model.eval()
#         print("Using device:", self.device)
#         print("Model device:", next(self.model.parameters()).device)

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str, schema=None):
#         if schema is not None:
#             prompt = f"""
# You are an LLM evaluator.
# Follow the evaluation instructions exactly.

# Return ONLY a valid JSON object matching this schema:
# {json.dumps(schema.model_json_schema(), indent=2)}

# Do not include markdown, explanations, or text outside the JSON.
# Use double quotes and follow the schema exactly.

# Evaluation prompt:
# {prompt}
# """

#         messages = [
#             {
#                 "role": "system",
#                 "content": "You are a precise LLM evaluator. Return only valid JSON when requested."
#             },
#             {"role": "user", "content": prompt}
#         ]

#         text = self.tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )

#         inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

#         with torch.inference_mode():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=512,
#                 do_sample=False,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#         response = self.tokenizer.decode(
#             outputs[0][inputs["input_ids"].shape[1]:],
#             skip_special_tokens=True
#         ).strip()

#         if schema is None:
#             return response

#         start = min(
#             (i for i in [response.find("{"), response.find("[")] if i != -1),
#             default=-1
#         )

#         if start == -1:
#             raise ValueError(f"Qwen did not return JSON:\n{response}")

#         parsed = repair_json(response[start:])

#         if "verdicts" in schema.model_fields:
#             if isinstance(parsed, list):
#                 parsed = {"verdicts": parsed}
#             elif isinstance(parsed, dict) and "verdict" in parsed:
#                 parsed = {"verdicts": [parsed]}

#             if isinstance(parsed, dict) and isinstance(parsed.get("verdicts"), list):
#                 parsed["verdicts"] = [
#                     {
#                         "verdict": str(
#                             item.get("verdict", {}).get("verdict", "")
#                             if isinstance(item.get("verdict"), dict)
#                             else item.get("verdict", "")
#                         ),
#                         "reason": str(
#                             item.get("verdict", {}).get(
#                                 "reason", item.get("reason", "")
#                             )
#                             if isinstance(item.get("verdict"), dict)
#                             else item.get("reason", "")
#                         )
#                     }
#                     for item in parsed["verdicts"]
#                     if isinstance(item, dict)
#                 ]

#         try:
#             return schema.model_validate(parsed)
#         except Exception as e:
#             raise ValueError(
#                 f"Qwen response did not match DeepEval schema.\n"
#                 f"Response: {json.dumps(parsed, indent=2)}\n"
#                 f"Error: {e}"
#             ) from e

#     async def a_generate(self, prompt: str, schema=None):
#         return self.generate(prompt, schema)

#     def get_model_name(self):
#         return self.model_name


# def run(retriever):

#     with open(GOLDEN_PATH, encoding="utf-8") as f:
#         goldens = json.load(f)

#     test_cases = [
#         LLMTestCase(
#             input=g["query"],
#             expected_output=g["ideal_answer"],
#             retrieval_context=[
#                 doc.page_content for doc in retriever.invoke(g["query"])
#             ],
#             actual_output="(generator not evaluated in this run)"
#         )
#         for g in goldens
#     ]

#     judge_model = LocalQwenLLM()

#     metrics = [
#         ContextualRecallMetric(
#             threshold=THRESHOLD, model=judge_model, include_reason=True
#         ),
#         ContextualPrecisionMetric(
#             threshold=THRESHOLD, model=judge_model, include_reason=True
#         )
#     ]

#     result = evaluate(
#         test_cases=test_cases,
#         metrics=metrics,
#         async_config=AsyncConfig(run_async=False),
#         hyperparameters={
#             "retriever": "reranker",
#             "embedding_model": EMBEDDING_MODEL,
#             "chunk_size": CHUNK_SIZE,
#             "chunk_overlap": CHUNK_OVERLAP,
#             "fetch_k": FETCH_K,
#             "top_k": TOP_K,
#             "judge_model": JUDGE_MODEL,
#             "golden_set": GOLDEN_PATH
#         }
#     )

#     return summarize_by_metric(result)


# def run_local():
#     return run(RerankingRetriever())


# if __name__ == "__main__":
#     run_local()























# import json
# import torch

# from json_repair import loads as repair_json
# from transformers import AutoTokenizer, AutoModelForCausalLM

# from deepeval import evaluate
# from deepeval.evaluate import AsyncConfig
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
# from deepeval.models import DeepEvalBaseLLM

# from src.reranker import RerankingRetriever
# from src.retriever import (
#     EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
#     FETCH_K, TOP_K, THRESHOLD
# )
# from evals.harness import summarize_by_metric


# GOLDEN_PATH = "goldens/retriever_goldens.json"
# JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# MAX_NEW_TOKENS = 1024


# class LocalQwenLLM(DeepEvalBaseLLM):

#     def __init__(self):
#         self.model_name = JUDGE_MODEL
#         self.device = "mps" if torch.backends.mps.is_available() else "cpu"
#         dtype = torch.float16 if self.device == "mps" else torch.float32

#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name, torch_dtype=dtype
#         ).to(self.device)

#         self.model.eval()
#         print("Using device:", self.device)
#         print("Model device:", next(self.model.parameters()).device)

#     def load_model(self):
#         return self.model

#     def generate(self, prompt, schema=None):
#         if schema:
#             prompt = f"""
# Return only valid JSON matching this schema:
# {json.dumps(schema.model_json_schema(), indent=2)}

# Every verdict must be "yes" or "no".
# Provide a short reason for each verdict.

# Evaluation prompt:
# {prompt}
# """

#         messages = [
#             {"role": "system", "content": "You are a precise LLM evaluator. Return only JSON when requested."},
#             {"role": "user", "content": prompt}
#         ]

#         text = self.tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )

#         inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

#         with torch.inference_mode():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=MAX_NEW_TOKENS,
#                 do_sample=False,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#         response = self.tokenizer.decode(
#             outputs[0][inputs["input_ids"].shape[1]:],
#             skip_special_tokens=True
#         ).strip()

#         if schema is None:
#             return response

#         start = min(
#             (i for i in (response.find("{"), response.find("[")) if i >= 0),
#             default=-1
#         )

#         if start == -1:
#             raise ValueError(f"Qwen did not return JSON:\n{response}")

#         parsed = repair_json(response[start:])

#         if "verdicts" in schema.model_fields:
#             if isinstance(parsed, list):
#                 parsed = {"verdicts": parsed}
#             elif isinstance(parsed, dict) and "verdict" in parsed:
#                 parsed = {"verdicts": [parsed]}

#             if isinstance(parsed, dict) and isinstance(parsed.get("verdicts"), list):
#                 verdicts = []

#                 for item in parsed["verdicts"]:
#                     if not isinstance(item, dict):
#                         continue

#                     verdict = item.get("verdict", "")
#                     reason = item.get("reason", "")

#                     if isinstance(verdict, dict):
#                         reason = verdict.get("reason", reason)
#                         verdict = verdict.get("verdict", "")

#                     verdict = str(verdict).strip().lower()

#                     if verdict not in ("yes", "no"):
#                         raise ValueError(f"Invalid Qwen verdict: {item}")

#                     verdicts.append({
#                         "verdict": verdict,
#                         "reason": str(reason).strip()
#                     })

#                 parsed["verdicts"] = verdicts

#         try:
#             return schema.model_validate(parsed)
#         except Exception as e:
#             raise ValueError(
#                 f"Invalid Qwen response:\n{json.dumps(parsed, indent=2)}\n{e}"
#             ) from e

#     async def a_generate(self, prompt, schema=None):
#         return self.generate(prompt, schema)

#     def get_model_name(self):
#         return self.model_name


# def run(retriever):
#     with open(GOLDEN_PATH, encoding="utf-8") as f:
#         goldens = json.load(f)

#     if not isinstance(goldens, list) or not goldens:
#         raise ValueError("Golden file must contain a non-empty JSON list.")

#     test_cases = []

#     for golden in goldens:
#         query = golden["query"]
#         docs = retriever.invoke(query)

#         test_cases.append(LLMTestCase(
#             input=query,
#             actual_output="(generator not evaluated in this run)",
#             expected_output=golden["ideal_answer"],
#             retrieval_context=[doc.page_content for doc in docs]
#         ))

#     judge = LocalQwenLLM()

#     metrics = [
#         ContextualRecallMetric(
#             threshold=THRESHOLD, model=judge, include_reason=True
#         ),
#         ContextualPrecisionMetric(
#             threshold=THRESHOLD, model=judge, include_reason=True
#         )
#     ]

#     result = evaluate(
#         test_cases=test_cases,
#         metrics=metrics,
#         async_config=AsyncConfig(run_async=False),
#         hyperparameters={
#             "retriever": "reranker",
#             "embedding_model": EMBEDDING_MODEL,
#             "chunk_size": CHUNK_SIZE,
#             "chunk_overlap": CHUNK_OVERLAP,
#             "fetch_k": FETCH_K,
#             "top_k": TOP_K,
#             "judge_model": JUDGE_MODEL,
#             "golden_set": GOLDEN_PATH
#         }
#     )

#     return summarize_by_metric(result)


# def run_local():
#     return run(RerankingRetriever(fetch_k=FETCH_K, top_k=TOP_K))


# if __name__ == "__main__":
#     run_local()


























import json
import os
from pathlib import Path

from dotenv import load_dotenv
from json_repair import loads as repair_json
from langchain_ollama import ChatOllama

# Set this before importing DeepEval so its synchronous metric calls
# allow enough time for a local model to complete long verdict lists.
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.models import DeepEvalBaseLLM

from src.reranker import RerankingRetriever
from src.retriever import (
    EMBEDDING_MODEL,
    DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FETCH_K,
    TOP_K,
    THRESHOLD,
    SEMANTIC_BREAKPOINT_TYPE,
    SEMANTIC_BREAKPOINT_AMOUNT,
)
from evals.harness import summarize_by_metric


load_dotenv()


# =========================================================
# CONFIGURATION
# =========================================================

GOLDEN_PATH = Path("goldens/retriever_goldens.json")

# Gemma 3 is used locally through Ollama as the DeepEval judge.
JUDGE_MODEL = "gemma3:4b"

# The previous Qwen run produced a very large verdict list and was
# truncated. Allow enough output tokens for the complete structured result.
JUDGE_CONTEXT_SIZE = 8192
MAX_NEW_TOKENS = 8192
MAX_SCHEMA_ATTEMPTS = 2

EXPERIMENT_NAME = "semantic-chunking-multiquery-crossencoder"


# =========================================================
# LOCAL OLLAMA JUDGE
# =========================================================

class LocalOllamaLLM(DeepEvalBaseLLM):
    """DeepEval-compatible local LLM using Ollama JSON-schema output."""

    def __init__(self):
        self.model_name = JUDGE_MODEL
        self.model = ChatOllama(
            model=self.model_name,
            temperature=0,
            num_ctx=JUDGE_CONTEXT_SIZE,
            num_predict=MAX_NEW_TOKENS,
        )
        self.max_schema_attempts = MAX_SCHEMA_ATTEMPTS

        print("Evaluation judge:", self.model_name)
        print("Judge provider: Ollama")
        print("Judge mode: Local")
        print("Judge output mode: Ollama JSON Schema constrained output")

    def load_model(self):
        return self.model

    @staticmethod
    def _extract_content(response):
        content = getattr(response, "content", response)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(str(part["text"]))
                else:
                    text_parts.append(str(part))
            return "\n".join(text_parts)

        return str(content)

    @staticmethod
    def _parse_json(raw_text):
        """Parse normal JSON first, then try json_repair as a fallback."""
        try:
            return json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            parsed = repair_json(raw_text)
            if isinstance(parsed, str):
                parsed = repair_json(parsed)
            return parsed

    @staticmethod
    def _normalize_verdicts(parsed, schema):
        """Normalize DeepEval verdict objects without changing yes/no labels."""
        if not isinstance(parsed, dict):
            if isinstance(parsed, list):
                parsed = {"verdicts": parsed}
            else:
                raise ValueError("Judge response must be a JSON object.")

        schema_fields = getattr(schema, "model_fields", {})
        if "verdicts" not in schema_fields:
            return parsed

        verdicts = parsed.get("verdicts")
        if not isinstance(verdicts, list) or not verdicts:
            raise ValueError("Judge response has no non-empty 'verdicts' list.")

        normalized = []
        for index, item in enumerate(verdicts):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Verdict at position {index} is not a JSON object."
                )

            item = dict(item)
            verdict = item.get("verdict", "")
            reason = item.get("reason", "")

            if isinstance(verdict, dict):
                reason = verdict.get("reason", reason)
                verdict = verdict.get("verdict", "")

            verdict = str(verdict).strip().lower()
            if verdict not in {"yes", "no"}:
                raise ValueError(
                    f"Invalid verdict at position {index}: {verdict!r}"
                )

            # DeepEval requires a reason. Add a neutral placeholder only
            # when the yes/no verdict itself is valid but its reason is blank.
            if not str(reason).strip():
                reason = "No explanation was provided."

            item["verdict"] = verdict
            item["reason"] = str(reason).strip()
            normalized.append(item)

        parsed["verdicts"] = normalized
        return parsed

    def generate(self, prompt, schema=None):
        # DeepEval may request ordinary text generation for some operations.
        if schema is None:
            response = self.model.invoke(prompt)
            return self._extract_content(response)

        # Ollama accepts a JSON Schema object in its `format` parameter.
        # This constrains the output structure instead of relying only on
        # instructions asking the model to produce valid JSON.
        json_schema = schema.model_json_schema()
        structured_model = self.model.bind(format=json_schema)

        schema_prompt = f"""{prompt}

Return only the JSON object required by the supplied response schema.
For every verdict, use exactly \"yes\" or \"no\". Include a concise reason.
Do not omit any verdict. Complete the entire response."""

        last_error = None
        last_raw = ""

        for attempt in range(1, self.max_schema_attempts + 1):
            try:
                attempt_prompt = schema_prompt
                if attempt > 1:
                    attempt_prompt += (
                        "\n\nThe previous response failed validation: "
                        f"{last_error}. Regenerate the complete JSON response."
                    )

                response = structured_model.invoke(attempt_prompt)
                last_raw = self._extract_content(response)
                parsed = self._parse_json(last_raw)
                parsed = self._normalize_verdicts(parsed, schema)

                # Validate against the exact Pydantic schema expected by DeepEval.
                return schema.model_validate(parsed)

            except Exception as error:
                last_error = str(error)
                print(
                    f"Judge structured-output attempt {attempt}/"
                    f"{self.max_schema_attempts} failed: {last_error[:500]}"
                )

        raise ValueError(
            "Ollama could not produce a valid response matching "
            f"{schema.__name__} after {self.max_schema_attempts} attempts. "
            f"Last validation error: {last_error}. "
            f"Last response (first 2000 characters): {last_raw[:2000]}"
        )

    async def a_generate(self, prompt, schema=None):
        # Evaluation is configured with async_mode=False, but this method
        # remains available to satisfy DeepEval's model interface.
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


# =========================================================
# LOAD GOLDEN DATASET
# =========================================================

def load_goldens():
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found: {GOLDEN_PATH}")

    with GOLDEN_PATH.open("r", encoding="utf-8") as file:
        goldens = json.load(file)

    if not isinstance(goldens, list) or not goldens:
        raise ValueError("Golden file must contain a non-empty JSON list.")

    for index, golden in enumerate(goldens):
        if not isinstance(golden, dict):
            raise ValueError(f"Golden entry {index} must be a JSON object.")

        if not golden.get("query") or not golden.get("ideal_answer"):
            raise ValueError(
                f"Golden entry {index} must contain non-empty "
                "'query' and 'ideal_answer' fields."
            )

    print("Golden test cases loaded:", len(goldens))
    return goldens


# =========================================================
# RUN EVALUATION
# =========================================================

def run(retriever):
    goldens = load_goldens()
    test_cases = []

    # -----------------------------------------------------
    # RETRIEVAL
    # -----------------------------------------------------
    for index, golden in enumerate(goldens, start=1):
        query = golden["query"]

        print("\n" + "=" * 70)
        print(f"TEST CASE {index}/{len(goldens)}")
        print("QUERY:", query)

        docs = retriever.invoke(query)
        retrieval_context = [doc.page_content for doc in docs]

        print("Retrieved documents:", len(docs))

        for rank, doc in enumerate(docs, start=1):
            print(f"\nRANK {rank}")
            print("SESSION:", doc.metadata.get("session"))
            print("SOURCE:", doc.metadata.get("source"))
            print("CONTENT:", doc.page_content[:700])

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output="(generator not evaluated in this run)",
                expected_output=golden["ideal_answer"],
                retrieval_context=retrieval_context,
            )
        )

    if not test_cases:
        raise ValueError("No DeepEval test cases were generated.")

    # -----------------------------------------------------
    # INITIALIZE LOCAL JUDGE
    # -----------------------------------------------------
    judge = LocalOllamaLLM()

    # -----------------------------------------------------
    # DEEPEVAL METRICS
    # -----------------------------------------------------
    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
    ]

    # -----------------------------------------------------
    # RUN DEEPEVAL
    # -----------------------------------------------------
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        hyperparameters={
            "experiment": EXPERIMENT_NAME,
            "retriever": "MultiQueryRetriever",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "query_generation_model": JUDGE_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "vector_database": "Chroma",
            "database_directory": str(DB_DIR),
            "chunking_strategy": "SemanticChunker",
            "semantic_breakpoint_type": SEMANTIC_BREAKPOINT_TYPE,
            "semantic_breakpoint_amount": SEMANTIC_BREAKPOINT_AMOUNT,
            "max_chunk_length": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "fetch_k": FETCH_K,
            "top_k": TOP_K,
            "judge_model": JUDGE_MODEL,
            "judge_context_size": JUDGE_CONTEXT_SIZE,
            "judge_max_new_tokens": MAX_NEW_TOKENS,
            "threshold": THRESHOLD,
            "golden_set": str(GOLDEN_PATH),
        },
    )

    # -----------------------------------------------------
    # SUMMARIZE RESULTS
    # -----------------------------------------------------
    summary = summarize_by_metric(result)
    print("\nEvaluation summary:")
    print(summary)
    return summary


# =========================================================
# RUN COMPLETE PIPELINE
# =========================================================

def run_local():
    retriever = RerankingRetriever(
        fetch_k=FETCH_K,
        top_k=TOP_K,
    )
    return run(retriever)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_local()












