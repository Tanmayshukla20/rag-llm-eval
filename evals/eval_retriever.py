
# ## __________________________________________________________________________________
# ##                           BEFORE APPLYING RERANKING:
# ##___________________________________________________________________________________

# # import json

# # from json_repair import loads as repair_json
# # from dotenv import load_dotenv

# # from deepeval.evaluate import AsyncConfig
# # from deepeval import evaluate
# # from deepeval.test_case import LLMTestCase
# # from deepeval.metrics import (
# #     ContextualRecallMetric,
# #     ContextualPrecisionMetric
# # )
# # from deepeval.models import DeepEvalBaseLLM

# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # import torch

# # from src.retriever import build_retriever


# # load_dotenv()


# # GOLDEN_PATH = "goldens/retriever_goldens.json"
# # JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# # THRESHOLD = 0.7


# # # --- local replacement for OpenAI gpt-4.1-mini ---

# # class LocalQwenLLM(DeepEvalBaseLLM):

# #     def __init__(self):

# #         self.model_name = JUDGE_MODEL

# #         self.tokenizer = AutoTokenizer.from_pretrained(
# #             self.model_name
# #         )

# #         self.device = (
# #             "mps"
# #             if torch.backends.mps.is_available()
# #             else "cpu"
# #         )

# #         print("Using device:", self.device)

# #         self.model = AutoModelForCausalLM.from_pretrained(
# #             self.model_name,
# #             dtype=torch.float16,
# #         ).to(self.device)

# #         print(
# #             "Model device:",
# #             next(self.model.parameters()).device
# #         )

# #     def load_model(self):
# #         return self.model

# #     def generate(self, prompt: str, schema=None):

# #         model = self.load_model()

# #         # ---------------------------------------------------------
# #         # 1. ADD THE JSON SCHEMA TO THE PROMPT
# #         # ---------------------------------------------------------

# #         if schema is not None:

# #             schema_json = json.dumps(
# #                 schema.model_json_schema(),
# #                 indent=2
# #             )

# #             prompt = f"""
# # You are an LLM evaluator.

# # Follow the instructions in the evaluation prompt exactly.

# # You MUST return a valid JSON object matching the following JSON schema:

# # {schema_json}

# # IMPORTANT RULES:
# # - Return ONLY the JSON object.
# # - Do NOT return a JSON array.
# # - Use double quotes for all JSON keys.
# # - Use double quotes for all string values.
# # - Do NOT use single quotes.
# # - Do NOT use markdown.
# # - Do NOT include ```json.
# # - Do NOT include ``` .
# # - Do NOT include explanations outside the JSON object.
# # - Do NOT add trailing commas.
# # - Follow the schema exactly.

# # Evaluation prompt:

# # {prompt}
# # """

# #         # ---------------------------------------------------------
# #         # 2. BUILD QWEN CHAT PROMPT
# #         # ---------------------------------------------------------

# #         messages = [
# #             {
# #                 "role": "system",
# #                 "content": (
# #                     "You are a precise LLM evaluator. "
# #                     "Follow the requested format exactly. "
# #                     "When JSON output is requested, "
# #                     "return only valid JSON."
# #                 )
# #             },
# #             {
# #                 "role": "user",
# #                 "content": prompt
# #             }
# #         ]

# #         text = self.tokenizer.apply_chat_template(
# #             messages,
# #             tokenize=False,
# #             add_generation_prompt=True
# #         )

# #         inputs = self.tokenizer(
# #             text,
# #             return_tensors="pt"
# #         ).to(self.device)

# #         print(
# #             "Input device:",
# #             inputs["input_ids"].device
# #         )

# #         # ---------------------------------------------------------
# #         # 3. GENERATE RESPONSE
# #         # ---------------------------------------------------------

# #         with torch.no_grad():

# #             outputs = model.generate(
# #                 **inputs,
# #                 max_new_tokens=512,
# #                 do_sample=False,
# #                 pad_token_id=self.tokenizer.eos_token_id
# #             )

# #         generated_tokens = outputs[
# #             0
# #         ][
# #             inputs["input_ids"].shape[1]:
# #         ]

# #         response = self.tokenizer.decode(
# #             generated_tokens,
# #             skip_special_tokens=True
# #         ).strip()

# #         # ---------------------------------------------------------
# #         # 4. NORMAL TEXT RESPONSE
# #         # ---------------------------------------------------------

# #         if schema is None:
# #             return response

# #         # ---------------------------------------------------------
# #         # 5. CLEAN MARKDOWN FENCES
# #         # ---------------------------------------------------------

# #         response = response.strip()

# #         if response.startswith("```json"):

# #             response = response[
# #                 len("```json"):
# #             ].strip()

# #         elif response.startswith("```"):

# #             response = response[
# #                 len("```"):
# #             ].strip()

# #         if response.endswith("```"):

# #             response = response[
# #                 :-len("```")
# #             ].strip()

# #         # ---------------------------------------------------------
# #         # 6. FIND JSON
# #         # ---------------------------------------------------------

# #         object_start = response.find("{")
# #         array_start = response.find("[")

# #         # Choose whichever JSON structure appears first.
# #         if (
# #             object_start == -1
# #             and array_start == -1
# #         ):

# #             raise ValueError(
# #                 "Qwen did not return JSON.\n\n"
# #                 f"Raw response:\n{response}"
# #             )

# #         if object_start == -1:

# #             start = array_start

# #         elif array_start == -1:

# #             start = object_start

# #         else:

# #             start = min(
# #                 object_start,
# #                 array_start
# #             )

# #         # ---------------------------------------------------------
# #         # 7. REPAIR MALFORMED JSON
# #         # ---------------------------------------------------------

# #         try:

# #             parsed_json = repair_json(
# #                 response[start:]
# #             )

# #         except Exception as e:

# #             raise ValueError(
# #                 "Qwen returned JSON that could not be repaired.\n\n"
# #                 f"Raw response:\n{response}\n\n"
# #                 f"Repair error:\n{e}"
# #             ) from e

# #         # ---------------------------------------------------------
# #         # 8. NORMALIZE DEEPEVAL VERDICTS
# #         # ---------------------------------------------------------

# #         schema_fields = getattr(
# #             schema,
# #             "model_fields",
# #             {}
# #         )

# #         expects_verdicts = (
# #             "verdicts" in schema_fields
# #         )

# #         if expects_verdicts:

# #             # Qwen sometimes returns:
# #             #
# #             # [
# #             #     {
# #             #         "verdict": "yes",
# #             #         "reason": "..."
# #             #     }
# #             # ]
# #             #
# #             # DeepEval expects:
# #             #
# #             # {
# #             #     "verdicts": [...]
# #             # }

# #             if isinstance(
# #                 parsed_json,
# #                 list
# #             ):

# #                 parsed_json = {
# #                     "verdicts": parsed_json
# #                 }

# #             # Qwen sometimes returns:
# #             #
# #             # {
# #             #     "verdict": "yes",
# #             #     "reason": "..."
# #             # }
# #             #
# #             # Convert it to:
# #             #
# #             # {
# #             #     "verdicts": [
# #             #         {
# #             #             "verdict": "yes",
# #             #             "reason": "..."
# #             #         }
# #             #     ]
# #             # }

# #             elif (
# #                 isinstance(
# #                     parsed_json,
# #                     dict
# #                 )
# #                 and "verdict" in parsed_json
# #                 and "verdicts" not in parsed_json
# #             ):

# #                 parsed_json = {
# #                     "verdicts": [
# #                         parsed_json
# #                     ]
# #                 }

# #         # ---------------------------------------------------------
# #         # 9. VALIDATE AGAINST DEEPEVAL SCHEMA
# #         # ---------------------------------------------------------

# #         try:

# #             return schema.model_validate(
# #                 parsed_json
# #             )

# #         except Exception as e:

# #             raise ValueError(
# #                 "Qwen returned JSON, but it did not "
# #                 "match the schema expected by DeepEval.\n\n"
# #                 f"Parsed response:\n"
# #                 f"{json.dumps(parsed_json, indent=2)}\n\n"
# #                 f"Validation error:\n{e}"
# #             ) from e

# #     async def a_generate(
# #         self,
# #         prompt: str,
# #         schema=None
# #     ):
# #         return self.generate(
# #             prompt,
# #             schema
# #         )

# #     def get_model_name(self):
# #         return self.model_name


# # # 1. LOAD the golden set - the fixed, human-authored truth

# # with open(GOLDEN_PATH) as f:

# #     goldens = json.load(f)


# # # 2. RUN THE RETRIEVER on each question to fill retrieval_context,
# # #    then build one test case per golden.

# # retriever = build_retriever()

# # test_cases = []

# # for g in goldens:

# #     retrieved = retriever.invoke(
# #         g["query"]
# #     )

# #     retrieval_context = [
# #         doc.page_content
# #         for doc in retrieved
# #     ]

# #     test_cases.append(
# #         LLMTestCase(
# #             input=g["query"],
# #             expected_output=g["ideal_answer"],
# #             retrieval_context=retrieval_context,
# #             actual_output=(
# #                 "(generator not evaluated in this run)"
# #             ),
# #         )
# #     )


# # # 3. THE METRICS -- recall (did we miss?) and precision (ranked well?)

# # judge_model = LocalQwenLLM()

# # metrics = [

# #     ContextualRecallMetric(
# #         threshold=THRESHOLD,
# #         model=judge_model,
# #         include_reason=True
# #     ),

# #     ContextualPrecisionMetric(
# #         threshold=THRESHOLD,
# #         model=judge_model,
# #         include_reason=True
# #     ),

# # ]


# # # 4. EVALUATE -- every metric on every case, batched + parallel, with a printed report

# # evaluate(
# #     test_cases=test_cases,
# #     metrics=metrics,
# #     async_config=AsyncConfig(
# #         run_async=False
# #     ),
# #     hyperparameters={

# #         "retriever": "base_K5",
# #         # vs "reranked" when you swap it in

# #         "embedding_model":
# #             "sentence-transformers/all-MiniLM-L6-v2",

# #         "chunk_size": 1500,

# #         "chunk_overlap": 200,

# #         "top_K": 5,

# #         "judge_model": JUDGE_MODEL,

# #         "golden_set": GOLDEN_PATH,
# #     }
# # )




# ## __________________________________________________________________________________
# ##                           AFTER APPLYING RERANKING:
# ##___________________________________________________________________________________


# import json

# from json_repair import loads as repair_json
# from dotenv import load_dotenv

# from deepeval.evaluate import AsyncConfig
# from deepeval import evaluate
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import (
#     ContextualRecallMetric,
#     ContextualPrecisionMetric
# )
# from deepeval.models import DeepEvalBaseLLM

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# from src.reranker import RerankingRetriever


# load_dotenv()


# GOLDEN_PATH = "goldens/retriever_goldens.json"
# JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# THRESHOLD = 0.7


# # --- local replacement for OpenAI gpt-4.1-mini ---

# class LocalQwenLLM(DeepEvalBaseLLM):

#     def __init__(self):

#         self.model_name = JUDGE_MODEL

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name
#         )

#         self.device = (
#             "mps"
#             if torch.backends.mps.is_available()
#             else "cpu"
#         )

#         print("Using device:", self.device)

#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             dtype=torch.float16,
#         ).to(self.device)

#         print(
#             "Model device:",
#             next(self.model.parameters()).device
#         )

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str, schema=None):

#         model = self.load_model()

#         # ---------------------------------------------------------
#         # 1. ADD THE JSON SCHEMA TO THE PROMPT
#         # ---------------------------------------------------------

#         if schema is not None:

#             schema_json = json.dumps(
#                 schema.model_json_schema(),
#                 indent=2
#             )

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

#         # ---------------------------------------------------------
#         # 2. BUILD QWEN CHAT PROMPT
#         # ---------------------------------------------------------

#         messages = [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a precise LLM evaluator. "
#                     "Follow the requested format exactly. "
#                     "When JSON output is requested, "
#                     "return only valid JSON."
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]

#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         inputs = self.tokenizer(
#             text,
#             return_tensors="pt"
#         ).to(self.device)

#         print(
#             "Input device:",
#             inputs["input_ids"].device
#         )

#         # ---------------------------------------------------------
#         # 3. GENERATE RESPONSE
#         # ---------------------------------------------------------

#         with torch.no_grad():

#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=512,
#                 do_sample=False,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#         generated_tokens = outputs[
#             0
#         ][
#             inputs["input_ids"].shape[1]:
#         ]

#         response = self.tokenizer.decode(
#             generated_tokens,
#             skip_special_tokens=True
#         ).strip()

#         # ---------------------------------------------------------
#         # 4. NORMAL TEXT RESPONSE
#         # ---------------------------------------------------------

#         if schema is None:
#             return response

#         # ---------------------------------------------------------
#         # 5. CLEAN MARKDOWN FENCES
#         # ---------------------------------------------------------

#         response = response.strip()

#         if response.startswith("```json"):

#             response = response[
#                 len("```json"):
#             ].strip()

#         elif response.startswith("```"):

#             response = response[
#                 len("```"):
#             ].strip()

#         if response.endswith("```"):

#             response = response[
#                 :-len("```")
#             ].strip()

#         # ---------------------------------------------------------
#         # 6. FIND JSON
#         # ---------------------------------------------------------

#         object_start = response.find("{")
#         array_start = response.find("[")

#         # Choose whichever JSON structure appears first.
#         if (
#             object_start == -1
#             and array_start == -1
#         ):

#             raise ValueError(
#                 "Qwen did not return JSON.\n\n"
#                 f"Raw response:\n{response}"
#             )

#         if object_start == -1:

#             start = array_start

#         elif array_start == -1:

#             start = object_start

#         else:

#             start = min(
#                 object_start,
#                 array_start
#             )

#         # ---------------------------------------------------------
#         # 7. REPAIR MALFORMED JSON
#         # ---------------------------------------------------------

#         try:

#             parsed_json = repair_json(
#                 response[start:]
#             )

#         except Exception as e:

#             raise ValueError(
#                 "Qwen returned JSON that could not be repaired.\n\n"
#                 f"Raw response:\n{response}\n\n"
#                 f"Repair error:\n{e}"
#             ) from e

#         # ---------------------------------------------------------
#         # 8. NORMALIZE DEEPEVAL VERDICTS
#         # ---------------------------------------------------------

#         schema_fields = getattr(
#             schema,
#             "model_fields",
#             {}
#         )

#         expects_verdicts = (
#             "verdicts" in schema_fields
#         )

#         if expects_verdicts:

#             # ---------------------------------------------------------
#             # CASE 1:
#             # Qwen returns a list directly
#             #
#             # [
#             #     {
#             #         "verdict": "yes",
#             #         "reason": "..."
#             #     }
#             # ]
#             #
#             # Convert to:
#             #
#             # {
#             #     "verdicts": [...]
#             # }
#             # ---------------------------------------------------------

#             if isinstance(
#                 parsed_json,
#                 list
#             ):

#                 parsed_json = {
#                     "verdicts": parsed_json
#                 }

#             # ---------------------------------------------------------
#             # CASE 2:
#             # Qwen returns one verdict directly
#             #
#             # {
#             #     "verdict": "yes",
#             #     "reason": "..."
#             # }
#             #
#             # Convert to:
#             #
#             # {
#             #     "verdicts": [
#             #         {
#             #             "verdict": "yes",
#             #             "reason": "..."
#             #         }
#             #     ]
#             # }
#             # ---------------------------------------------------------

#             elif (
#                 isinstance(
#                     parsed_json,
#                     dict
#                 )
#                 and "verdict" in parsed_json
#                 and "verdicts" not in parsed_json
#             ):

#                 parsed_json = {
#                     "verdicts": [
#                         parsed_json
#                     ]
#                 }

#             # ---------------------------------------------------------
#             # CASE 3:
#             # Normalize the verdicts list
#             # ---------------------------------------------------------

#             if (
#                 isinstance(
#                     parsed_json,
#                     dict
#                 )
#                 and isinstance(
#                     parsed_json.get("verdicts"),
#                     list
#                 )
#             ):

#                 normalized_verdicts = []

#                 for item in parsed_json["verdicts"]:

#                     # Skip invalid entries
#                     if not isinstance(
#                         item,
#                         dict
#                     ):
#                         continue

#                     verdict = item.get(
#                         "verdict",
#                         ""
#                     )

#                     reason = item.get(
#                         "reason",
#                         ""
#                     )

#                     # -------------------------------------------------
#                     # CASE 3A:
#                     # Correct format
#                     #
#                     # "verdict": "yes"
#                     # -------------------------------------------------

#                     if isinstance(
#                         verdict,
#                         str
#                     ):

#                         normalized_verdicts.append(
#                             {
#                                 "verdict": verdict,
#                                 "reason": (
#                                     reason
#                                     if isinstance(
#                                         reason,
#                                         str
#                                     )
#                                     else str(reason)
#                                 )
#                             }
#                         )

#                     # -------------------------------------------------
#                     # CASE 3B:
#                     # Incorrect nested format
#                     #
#                     # "verdict": {
#                     #     "verdict": "no",
#                     #     "reason": "..."
#                     # }
#                     # -------------------------------------------------

#                     elif isinstance(
#                         verdict,
#                         dict
#                     ):

#                         nested_verdict = verdict.get(
#                             "verdict",
#                             ""
#                         )

#                         nested_reason = verdict.get(
#                             "reason",
#                             reason
#                         )

#                         normalized_verdicts.append(
#                             {
#                                 "verdict": (
#                                     nested_verdict
#                                     if isinstance(
#                                         nested_verdict,
#                                         str
#                                     )
#                                     else str(
#                                         nested_verdict
#                                     )
#                                 ),
#                                 "reason": (
#                                     nested_reason
#                                     if isinstance(
#                                         nested_reason,
#                                         str
#                                     )
#                                     else str(
#                                         nested_reason
#                                     )
#                                 )
#                             }
#                         )

#                     # -------------------------------------------------
#                     # CASE 3C:
#                     # Unexpected verdict type
#                     # -------------------------------------------------

#                     else:

#                         normalized_verdicts.append(
#                             {
#                                 "verdict": str(
#                                     verdict
#                                 ),
#                                 "reason": (
#                                     reason
#                                     if isinstance(
#                                         reason,
#                                         str
#                                     )
#                                     else str(reason)
#                                 )
#                             }
#                         )

#                 parsed_json["verdicts"] = (
#                     normalized_verdicts
#                 )

#         # ---------------------------------------------------------
#         # 9. VALIDATE AGAINST DEEPEVAL SCHEMA
#         # ---------------------------------------------------------

#         try:
#             return schema.model_validate(
#                 parsed_json
#             )

#         except Exception as e:

#             raise ValueError(
#                 "Qwen returned JSON, but it did not "
#                 "match the schema expected by DeepEval.\n\n"
#                 f"Parsed response:\n"
#                 f"{json.dumps(parsed_json, indent=2)}\n\n"
#                 f"Validation error:\n{e}"
#             ) from e

#     async def a_generate(
#         self,
#         prompt: str,
#         schema=None
#     ):
#         return self.generate(
#             prompt,
#             schema
#         )

#     def get_model_name(self):
#         return self.model_name


# # 1. LOAD the golden set - the fixed, human-authored truth

# with open(GOLDEN_PATH) as f:

#     goldens = json.load(f)


# # 2. RUN THE RETRIEVER on each question to fill retrieval_context,
# #    then build one test case per golden.

# retriever = RerankingRetriever(
#     fetch_k=10,
#     top_k=5
# )

# test_cases = []

# for g in goldens:

#     retrieved = retriever.invoke(
#         g["query"]
#     )

#     retrieval_context = [
#         doc.page_content
#         for doc in retrieved
#     ]

#     test_cases.append(
#         LLMTestCase(
#             input=g["query"],
#             expected_output=g["ideal_answer"],
#             retrieval_context=retrieval_context,
#             actual_output=(
#                 "(generator not evaluated in this run)"
#             ),
#         )
#     )


# # 3. THE METRICS -- recall (did we miss?) and precision (ranked well?)

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


# # 4. EVALUATE -- every metric on every case, batched + parallel, with a printed report

# evaluate(
#     test_cases=test_cases,
#     metrics=metrics,
#     async_config=AsyncConfig(
#         run_async=False
#     ),
#     hyperparameters={

#         "retriever": "reranker",
#         # base_K5 before reranking

#         "embedding_model":
#             "sentence-transformers/all-MiniLM-L6-v2",

#         "chunk_size": 1500,

#         "chunk_overlap": 200,

#         "top_K": 5,

#         "judge_model": JUDGE_MODEL,

#         "golden_set": GOLDEN_PATH,
#     }
# )



# # ## __________________________________________________________________________________
# # ##                           AFTER APPLYING RERANKING:
# # ##___________________________________________________________________________________

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
from json_repair import loads as repair_json
from dotenv import load_dotenv
from deepeval.evaluate import AsyncConfig
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import DeepEvalBaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.reranker import RerankingRetriever
from evals.harness import summarize_by_metric

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
THRESHOLD = 0.7


class LocalQwenLLM(DeepEvalBaseLLM):

    def __init__(self):
        self.model_name = JUDGE_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print("Using device:", self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float16
        ).to(self.device)
        print("Model device:", next(self.model.parameters()).device)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        model = self.load_model()

        if schema is not None:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            prompt = f"""
You are an LLM evaluator.

Follow the instructions in the evaluation prompt exactly.

You MUST return a valid JSON object matching the following JSON schema:

{schema_json}

IMPORTANT RULES:
- Return ONLY the JSON object.
- Do NOT return a JSON array.
- Use double quotes for all JSON keys.
- Use double quotes for all string values.
- Do NOT use single quotes.
- Do NOT use markdown.
- Do NOT include ```json.
- Do NOT include ``` .
- Do NOT include explanations outside the JSON object.
- Do NOT add trailing commas.
- Follow the schema exactly.

Evaluation prompt:

{prompt}
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise LLM evaluator. "
                    "Follow the requested format exactly. "
                    "When JSON output is requested, return only valid JSON."
                )
            },
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        print("Input device:", inputs["input_ids"].device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        if schema is None:
            return response

        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json"):].strip()
        elif response.startswith("```"):
            response = response[len("```"):].strip()
        if response.endswith("```"):
            response = response[:-len("```")].strip()

        object_start = response.find("{")
        array_start = response.find("[")

        if object_start == -1 and array_start == -1:
            raise ValueError(
                "Qwen did not return JSON.\n\n"
                f"Raw response:\n{response}"
            )

        if object_start == -1:
            start = array_start
        elif array_start == -1:
            start = object_start
        else:
            start = min(object_start, array_start)

        try:
            parsed_json = repair_json(response[start:])
        except Exception as e:
            raise ValueError(
                "Qwen returned JSON that could not be repaired.\n\n"
                f"Raw response:\n{response}\n\n"
                f"Repair error:\n{e}"
            ) from e

        if "verdicts" in getattr(schema, "model_fields", {}):
            if isinstance(parsed_json, list):
                parsed_json = {"verdicts": parsed_json}
            elif (
                isinstance(parsed_json, dict)
                and "verdict" in parsed_json
                and "verdicts" not in parsed_json
            ):
                parsed_json = {"verdicts": [parsed_json]}

            if (
                isinstance(parsed_json, dict)
                and isinstance(parsed_json.get("verdicts"), list)
            ):
                normalized = []

                for item in parsed_json["verdicts"]:
                    if not isinstance(item, dict):
                        continue

                    verdict = item.get("verdict", "")
                    reason = item.get("reason", "")

                    if isinstance(verdict, dict):
                        nested = verdict
                        verdict = nested.get("verdict", "")
                        reason = nested.get("reason", reason)

                    normalized.append({
                        "verdict": (
                            verdict if isinstance(verdict, str)
                            else str(verdict)
                        ),
                        "reason": (
                            reason if isinstance(reason, str)
                            else str(reason)
                        )
                    })

                parsed_json["verdicts"] = normalized

        try:
            return schema.model_validate(parsed_json)
        except Exception as e:
            raise ValueError(
                "Qwen returned JSON, but it did not "
                "match the schema expected by DeepEval.\n\n"
                f"Parsed response:\n{json.dumps(parsed_json, indent=2)}\n\n"
                f"Validation error:\n{e}"
            ) from e

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


def run(retriever):

    with open(GOLDEN_PATH) as f:
        goldens = json.load(f)

    test_cases = []

    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    judge_model = LocalQwenLLM()

    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=judge_model,
            include_reason=True
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=judge_model,
            include_reason=True
        ),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        hyperparameters={
            "retriever": "reranker",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 800,
            "chunk_overlap": 100,
            "top_K": 3,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        }
    )

    return summarize_by_metric(result)


def run_local():
    return run(RerankingRetriever(fetch_k=10, top_k=3))


if __name__ == "__main__":
    run_local()