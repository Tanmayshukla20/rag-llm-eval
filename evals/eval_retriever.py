

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
# from src.retriever import FETCH_K, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP, THRESHOLD, FETCH_K, TOP_K
# from evals.harness import summarize_by_metric


# GOLDEN_PATH = "goldens/retriever_goldens.json"
# JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


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
# from json_repair import loads as repair_json
# from dotenv import load_dotenv
# from deepeval.evaluate import AsyncConfig
# from deepeval import evaluate
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
# from deepeval.models import DeepEvalBaseLLM
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from src.reranker import RerankingRetriever

# from evals.harness import summarize_by_metric


# load_dotenv()

# GOLDEN_PATH = "goldens/retriever_goldens.json"
# JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# THRESHOLD = 0.7


# class LocalQwenLLM(DeepEvalBaseLLM):

#     def __init__(self):
#         self.model_name = JUDGE_MODEL
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.device = "mps" if torch.backends.mps.is_available() else "cpu"
#         print("Using device:", self.device)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name, dtype=torch.float16
#         ).to(self.device)
#         print("Model device:", next(self.model.parameters()).device)

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str, schema=None):
#         model = self.load_model()

#         if schema is not None:
#             schema_json = json.dumps(schema.model_json_schema(), indent=2)
#             prompt = f"""
# You are an LLM evaluator.

# Follow the instructions in the evaluation prompt exactly.

# You MUST return a valid JSON object matching the following JSON schema:

# {schema_json}

# IMPORTANT RULES:
# - Return ONLY the JSON object.
# - Do NOT return a JSON array.
# - Use double quotes for all JSON keys.
# - Use double quotes for all string values.
# - Do NOT use single quotes.
# - Do NOT use markdown.
# - Do NOT include ```json.
# - Do NOT include ``` .
# - Do NOT include explanations outside the JSON object.
# - Do NOT add trailing commas.
# - Follow the schema exactly.

# Evaluation prompt:

# {prompt}
# """

#         messages = [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a precise LLM evaluator. "
#                     "Follow the requested format exactly. "
#                     "When JSON output is requested, return only valid JSON."
#                 )
#             },
#             {"role": "user", "content": prompt}
#         ]

#         text = self.tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
#         print("Input device:", inputs["input_ids"].device)

#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=512,
#                 do_sample=False,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#         generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
#         response = self.tokenizer.decode(
#             generated_tokens, skip_special_tokens=True
#         ).strip()

#         if schema is None:
#             return response

#         response = response.strip()
#         if response.startswith("```json"):
#             response = response[len("```json"):].strip()
#         elif response.startswith("```"):
#             response = response[len("```"):].strip()
#         if response.endswith("```"):
#             response = response[:-len("```")].strip()

#         object_start = response.find("{")
#         array_start = response.find("[")

#         if object_start == -1 and array_start == -1:
#             raise ValueError(
#                 "Qwen did not return JSON.\n\n"
#                 f"Raw response:\n{response}"
#             )

#         if object_start == -1:
#             start = array_start
#         elif array_start == -1:
#             start = object_start
#         else:
#             start = min(object_start, array_start)

#         try:
#             parsed_json = repair_json(response[start:])
#         except Exception as e:
#             raise ValueError(
#                 "Qwen returned JSON that could not be repaired.\n\n"
#                 f"Raw response:\n{response}\n\n"
#                 f"Repair error:\n{e}"
#             ) from e

#         if "verdicts" in getattr(schema, "model_fields", {}):
#             if isinstance(parsed_json, list):
#                 parsed_json = {"verdicts": parsed_json}
#             elif (
#                 isinstance(parsed_json, dict)
#                 and "verdict" in parsed_json
#                 and "verdicts" not in parsed_json
#             ):
#                 parsed_json = {"verdicts": [parsed_json]}

#             if (
#                 isinstance(parsed_json, dict)
#                 and isinstance(parsed_json.get("verdicts"), list)
#             ):
#                 normalized = []

#                 for item in parsed_json["verdicts"]:
#                     if not isinstance(item, dict):
#                         continue

#                     verdict = item.get("verdict", "")
#                     reason = item.get("reason", "")

#                     if isinstance(verdict, dict):
#                         nested = verdict
#                         verdict = nested.get("verdict", "")
#                         reason = nested.get("reason", reason)

#                     normalized.append({
#                         "verdict": (
#                             verdict if isinstance(verdict, str)
#                             else str(verdict)
#                         ),
#                         "reason": (
#                             reason if isinstance(reason, str)
#                             else str(reason)
#                         )
#                     })

#                 parsed_json["verdicts"] = normalized

#         try:
#             return schema.model_validate(parsed_json)
#         except Exception as e:
#             raise ValueError(
#                 "Qwen returned JSON, but it did not "
#                 "match the schema expected by DeepEval.\n\n"
#                 f"Parsed response:\n{json.dumps(parsed_json, indent=2)}\n\n"
#                 f"Validation error:\n{e}"
#             ) from e

#     async def a_generate(self, prompt: str, schema=None):
#         return self.generate(prompt, schema)

#     def get_model_name(self):
#         return self.model_name



# def run(retriever):

#     with open(GOLDEN_PATH) as f:
#         goldens = json.load(f)

#     test_cases = []

#     for g in goldens:
#         retrieved = retriever.invoke(g["query"])
#         retrieval_context = [doc.page_content for doc in retrieved]

#         test_cases.append(
#             LLMTestCase(
#                 input=g["query"],
#                 expected_output=g["ideal_answer"],
#                 retrieval_context=retrieval_context,
#                 actual_output="(generator not evaluated in this run)",
#             )
#         )

#     judge_model = LocalQwenLLM()

#     metrics = [
#         ContextualRecallMetric(
#             threshold=THRESHOLD,
#             model=judge_model,
#             include_reason=True
#         ),
#         ContextualPrecisionMetric(
#             threshold=THRESHOLD,
#             model=judge_model,
#             include_reason=True
#         ),
#     ]

#     result = evaluate(
#         test_cases=test_cases,
#         metrics=metrics,
#         async_config=AsyncConfig(run_async=False),
#         hyperparameters={
#             "retriever": "reranker",
#             "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
#             "chunk_size": 800,
#             "chunk_overlap": 100,
#             "top_K": 3,
#             "judge_model": JUDGE_MODEL,
#             "golden_set": GOLDEN_PATH,
#         }
#     )

#     return summarize_by_metric(result)


# def run_local():
#     return run(RerankingRetriever(fetch_k=10, top_k=3))


# if __name__ == "__main__":
#     run_local()

# with open(GOLDEN_PATH) as f:
#     goldens = json.load(f)

# retriever = RerankingRetriever(fetch_k=10, top_k=3)
# test_cases = []

# for g in goldens:
#     retrieved = retriever.invoke(g["query"])
#     retrieval_context = [doc.page_content for doc in retrieved]

#     test_cases.append(
#         LLMTestCase(
#             input=g["query"],
#             expected_output=g["ideal_answer"],
#             retrieval_context=retrieval_context,
#             actual_output="(generator not evaluated in this run)",
#         )
#     )

# judge_model = LocalQwenLLM()

# metrics = [
#     ContextualRecallMetric(
#         threshold=THRESHOLD,
#         model=judge_model,
#         include_reason=True
#     ),
#     ContextualPrecisionMetric(
#         threshold=THRESHOLD,
#         model=judge_model,
#         include_reason=True
#     ),
# ]

# evaluate(
#     test_cases=test_cases,
#     metrics=metrics,
#     async_config=AsyncConfig(run_async=False),
#     hyperparameters={
#         "retriever": "reranker",
#         "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
#         "chunk_size": 800,
#         "chunk_overlap": 100,
#         "top_K": 3,
#         "judge_model": JUDGE_MODEL,
#         "golden_set": GOLDEN_PATH,
#     }
# )






























import json
import torch

from json_repair import loads as repair_json
from transformers import AutoTokenizer, AutoModelForCausalLM

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import DeepEvalBaseLLM

from src.reranker import RerankingRetriever
from src.retriever import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    FETCH_K, TOP_K, THRESHOLD
)
from evals.harness import summarize_by_metric


GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 1024


class LocalQwenLLM(DeepEvalBaseLLM):

    def __init__(self):
        self.model_name = JUDGE_MODEL
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device == "mps" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype
        ).to(self.device)

        self.model.eval()
        print("Using device:", self.device)
        print("Model device:", next(self.model.parameters()).device)

    def load_model(self):
        return self.model

    def generate(self, prompt, schema=None):
        if schema:
            prompt = f"""
Return only valid JSON matching this schema:
{json.dumps(schema.model_json_schema(), indent=2)}

Every verdict must be "yes" or "no".
Provide a short reason for each verdict.

Evaluation prompt:
{prompt}
"""

        messages = [
            {"role": "system", "content": "You are a precise LLM evaluator. Return only JSON when requested."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        if schema is None:
            return response

        start = min(
            (i for i in (response.find("{"), response.find("[")) if i >= 0),
            default=-1
        )

        if start == -1:
            raise ValueError(f"Qwen did not return JSON:\n{response}")

        parsed = repair_json(response[start:])

        if "verdicts" in schema.model_fields:
            if isinstance(parsed, list):
                parsed = {"verdicts": parsed}
            elif isinstance(parsed, dict) and "verdict" in parsed:
                parsed = {"verdicts": [parsed]}

            if isinstance(parsed, dict) and isinstance(parsed.get("verdicts"), list):
                verdicts = []

                for item in parsed["verdicts"]:
                    if not isinstance(item, dict):
                        continue

                    verdict = item.get("verdict", "")
                    reason = item.get("reason", "")

                    if isinstance(verdict, dict):
                        reason = verdict.get("reason", reason)
                        verdict = verdict.get("verdict", "")

                    verdict = str(verdict).strip().lower()

                    if verdict not in ("yes", "no"):
                        raise ValueError(f"Invalid Qwen verdict: {item}")

                    verdicts.append({
                        "verdict": verdict,
                        "reason": str(reason).strip()
                    })

                parsed["verdicts"] = verdicts

        try:
            return schema.model_validate(parsed)
        except Exception as e:
            raise ValueError(
                f"Invalid Qwen response:\n{json.dumps(parsed, indent=2)}\n{e}"
            ) from e

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


def run(retriever):
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        goldens = json.load(f)

    if not isinstance(goldens, list) or not goldens:
        raise ValueError("Golden file must contain a non-empty JSON list.")

    test_cases = []

    for golden in goldens:
        query = golden["query"]
        docs = retriever.invoke(query)

        test_cases.append(LLMTestCase(
            input=query,
            actual_output="(generator not evaluated in this run)",
            expected_output=golden["ideal_answer"],
            retrieval_context=[doc.page_content for doc in docs]
        ))

    judge = LocalQwenLLM()

    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD, model=judge, include_reason=True
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD, model=judge, include_reason=True
        )
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        hyperparameters={
            "retriever": "reranker",
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "fetch_k": FETCH_K,
            "top_k": TOP_K,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH
        }
    )

    return summarize_by_metric(result)


def run_local():
    return run(RerankingRetriever(fetch_k=FETCH_K, top_k=TOP_K))


if __name__ == "__main__":
    run_local()
