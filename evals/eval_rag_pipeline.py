# # evals/eval_rag_pipeline.py

# import json
# import torch
# from json_repair import loads as repair_json
# from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModelForCausalLM

# from deepeval import evaluate
# from deepeval.evaluate import AsyncConfig
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import (
#     FaithfulnessMetric,
#     AnswerRelevancyMetric,
#     ContextualRelevancyMetric,
# )
# from deepeval.models import DeepEvalBaseLLM

# from src.rag_pipeline import RagPipeline

# load_dotenv()

# GOLDEN_PATH = "goldens/faithfulness_dataset.json"
# JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# THRESHOLD = 0.7


# class LocalQwenLLM(DeepEvalBaseLLM):

#     def __init__(self):
#         self.model_name = JUDGE_MODEL
#         self.device = "mps" if torch.backends.mps.is_available() else "cpu"

#         print("Using device:", self.device)

#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             dtype=torch.float16
#         ).to(self.device)

#         self.model.eval()

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str, schema=None):

#         if schema is not None:
#             schema_json = json.dumps(
#                 schema.model_json_schema(),
#                 indent=2
#             )

#             prompt = f"""
# You are an LLM evaluator.

# Return ONLY valid JSON matching this schema:

# {schema_json}

# STRICT RULES:
# - Return only the JSON object.
# - Use double quotes.
# - Do not use markdown.
# - Do not use ```json.
# - Follow the schema exactly.
# - Do not omit any required field.
# - Do not rename any field.
# - Do not add fields that are not required.

# For every item inside "verdicts":
# - "verdict" must contain the required verdict.
# - "reason" must contain the reason for the verdict.
# - If "statement" exists in the schema, you MUST include "statement".
# - When "statement" is required, copy the statement being evaluated exactly
#   from the evaluation prompt. Do not invent or summarize it.

# Evaluation prompt:

# {prompt}
# """

#         messages = [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a precise LLM evaluator. "
#                     "When JSON is requested, return only valid JSON "
#                     "that exactly follows the provided schema."
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

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=256,
#                 do_sample=False,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#         response = self.tokenizer.decode(
#             outputs[0][inputs["input_ids"].shape[1]:],
#             skip_special_tokens=True
#         ).strip()

#         if schema is None:
#             return response

#         response = (
#             response
#             .removeprefix("```json")
#             .removeprefix("```")
#             .removesuffix("```")
#             .strip()
#         )

#         starts = [
#             i for i in (
#                 response.find("{"),
#                 response.find("[")
#             )
#             if i >= 0
#         ]

#         if not starts:
#             raise ValueError(
#                 f"Qwen did not return JSON:\n{response}"
#             )

#         parsed = repair_json(response[min(starts):])

#         # --------------------------------------------------------
#         # NORMALIZE DEEPEVAL VERDICTS
#         # --------------------------------------------------------

#         if "verdicts" in getattr(schema, "model_fields", {}):

#             if isinstance(parsed, list):
#                 parsed = {"verdicts": parsed}

#             elif (
#                 isinstance(parsed, dict)
#                 and "verdict" in parsed
#             ):
#                 parsed = {"verdicts": [parsed]}

#             if (
#                 isinstance(parsed, dict)
#                 and isinstance(parsed.get("verdicts"), list)
#             ):
#                 normalized = []

#                 for item in parsed["verdicts"]:

#                     if not isinstance(item, dict):
#                         continue

#                     statement = item.get("statement", "")
#                     verdict = item.get("verdict", "")
#                     reason = item.get("reason", "")

#                     # Handle Qwen occasionally nesting verdict data.
#                     if isinstance(verdict, dict):
#                         nested = verdict

#                         statement = nested.get(
#                             "statement",
#                             statement
#                         )

#                         reason = nested.get(
#                             "reason",
#                             reason
#                         )

#                         verdict = nested.get(
#                             "verdict",
#                             ""
#                         )

#                     result = {
#                         "verdict": str(verdict),
#                         "reason": str(reason)
#                     }

#                     # ContextualRelevancyVerdicts requires
#                     # "statement". Other DeepEval verdict schemas
#                     # do not necessarily require it.
#                     if "statement" in item or statement:
#                         result["statement"] = str(statement)

#                     normalized.append(result)

#                 parsed["verdicts"] = normalized

#         # --------------------------------------------------------
#         # VALIDATE AGAINST THE EXACT DEEPEVAL SCHEMA
#         # --------------------------------------------------------

#         try:
#             return schema.model_validate(parsed)

#         except Exception as first_error:

#             # One repair attempt using Qwen itself.
#             repair_prompt = f"""
# Your previous JSON response did not match the required schema.

# REQUIRED SCHEMA:

# {json.dumps(schema.model_json_schema(), indent=2)}

# ORIGINAL EVALUATION:

# {prompt}

# PREVIOUS JSON:

# {json.dumps(parsed, indent=2)}

# Fix the JSON.

# STRICT RULES:
# - Return ONLY valid JSON.
# - Match the schema exactly.
# - Do not omit required fields.
# - Do not rename fields.
# - If "statement" is required, include it.
# - Copy each required statement from the original evaluation.
# - Keep the original verdict and reason whenever possible.

# Return the corrected JSON only.
# """

#             repair_messages = [
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are a JSON schema repair system. "
#                         "Return only valid JSON."
#                     )
#                 },
#                 {
#                     "role": "user",
#                     "content": repair_prompt
#                 }
#             ]

#             repair_text = self.tokenizer.apply_chat_template(
#                 repair_messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )

#             repair_inputs = self.tokenizer(
#                 repair_text,
#                 return_tensors="pt"
#             ).to(self.device)

#             with torch.no_grad():
#                 repair_outputs = self.model.generate(
#                     **repair_inputs,
#                     max_new_tokens=256,
#                     do_sample=False,
#                     pad_token_id=self.tokenizer.eos_token_id
#                 )

#             repair_response = self.tokenizer.decode(
#                 repair_outputs[0][
#                     repair_inputs["input_ids"].shape[1]:
#                 ],
#                 skip_special_tokens=True
#             ).strip()

#             repair_response = (
#                 repair_response
#                 .removeprefix("```json")
#                 .removeprefix("```")
#                 .removesuffix("```")
#                 .strip()
#             )

#             repair_starts = [
#                 i for i in (
#                     repair_response.find("{"),
#                     repair_response.find("[")
#                 )
#                 if i >= 0
#             ]

#             if not repair_starts:
#                 raise ValueError(
#                     f"Qwen JSON repair failed:\n{repair_response}"
#                 ) from first_error

#             repaired = repair_json(
#                 repair_response[min(repair_starts):]
#             )

#             return schema.model_validate(repaired)

#     async def a_generate(self, prompt: str, schema=None):
#         return self.generate(prompt, schema)

#     def get_model_name(self):
#         return self.model_name


# # ------------------------------------------------------------
# # RUN RAG PIPELINE
# # ------------------------------------------------------------

# with open(GOLDEN_PATH) as f:
#     goldens = json.load(f)

# rag = RagPipeline()
# test_cases = []

# for g in goldens:
#     result = rag.invoke(g["query"])

#     test_cases.append(
#         LLMTestCase(
#             input=g["query"],
#             actual_output=result["answer"],
#             retrieval_context=result["context"]
#         )
#     )

# print(f"Evaluating {len(test_cases)} test cases...")


# # ------------------------------------------------------------
# # EVALUATE
# # ------------------------------------------------------------

# judge_model = LocalQwenLLM()

# metrics = [
#     ContextualRelevancyMetric(
#         threshold=THRESHOLD,
#         model=judge_model,
#         include_reason=True
#     ),
#     FaithfulnessMetric(
#         threshold=THRESHOLD,
#         model=judge_model,
#         include_reason=True
#     ),
#     AnswerRelevancyMetric(
#         threshold=THRESHOLD,
#         model=judge_model,
#         include_reason=True
#     )
# ]

# evaluate(
#     test_cases=test_cases,
#     metrics=metrics,
#     async_config=AsyncConfig(run_async=False),
#     hyperparameters={
#         "rag_pipeline": "RagPipeline",
#         "judge_model": JUDGE_MODEL,
#         "judge_device": "mps",
#         "threshold": THRESHOLD,
#         "golden_set": GOLDEN_PATH,
#     }
# )






"""
evals/eval_rag_pipeline.py
"""


import json


import torch


from json_repair import loads as repair_json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM


from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.models import DeepEvalBaseLLM


from src.rag_pipeline import RagPipeline
from evals.harness import summarize_by_metric


load_dotenv()


GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
THRESHOLD = 0.7


class LocalQwenLLM(DeepEvalBaseLLM):


    def __init__(self):
        self.model_name = JUDGE_MODEL
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"


        print("Using device:", self.device)


        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16
        ).to(self.device)


        self.model.eval()


    def load_model(self):
        return self.model


    def generate(self, prompt: str, schema=None):


        if schema is not None:


            schema_json = json.dumps(
                schema.model_json_schema(),
                indent=2
            )


            prompt = f"""
You are an LLM evaluator.

Return ONLY valid JSON matching this schema:


{schema_json}


STRICT RULES:


- Return only the JSON object.
- Use double quotes.
- Do not use markdown.
- Do not use ```json.
- Follow the schema exactly.
- Do not omit any required field.
- Do not rename any field.
- Do not add fields that are not required.


For every item inside "verdicts":


- "verdict" must contain the required verdict.
- "reason" must contain the reason for the verdict.
- If "statement" exists in the schema, you MUST include "statement".
- When "statement" is required, copy the statement being evaluated exactly
  from the evaluation prompt. Do not invent or summarize it.


Evaluation prompt:


{prompt}

"""


        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise LLM evaluator. "
                    "When JSON is requested, return only valid JSON "
                    "that exactly follows the provided schema."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]


        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)


        with torch.no_grad():


            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )


        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()


        if schema is None:
            return response


        response = (
            response
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )


        starts = [
            i for i in (
                response.find("{"),
                response.find("[")
            )
            if i >= 0
        ]


        if not starts:
            raise ValueError(
                f"Qwen did not return JSON:\n{response}"
            )


        parsed = repair_json(
            response[min(starts):]
        )


        # --------------------------------------------------------
        # NORMALIZE DEEPEVAL VERDICTS
        # --------------------------------------------------------


        if "verdicts" in getattr(schema, "model_fields", {}):


            if isinstance(parsed, list):
                parsed = {"verdicts": parsed}


            elif (
                isinstance(parsed, dict)
                and "verdict" in parsed
            ):
                parsed = {"verdicts": [parsed]}


            if (
                isinstance(parsed, dict)
                and isinstance(parsed.get("verdicts"), list)
            ):


                normalized = []


                for item in parsed["verdicts"]:


                    if not isinstance(item, dict):
                        continue


                    statement = item.get("statement", "")
                    verdict = item.get("verdict", "")
                    reason = item.get("reason", "")


                    # Handle Qwen occasionally nesting verdict data.
                    if isinstance(verdict, dict):


                        nested = verdict


                        statement = nested.get(
                            "statement",
                            statement
                        )


                        reason = nested.get(
                            "reason",
                            reason
                        )


                        verdict = nested.get(
                            "verdict",
                            ""
                        )


                    result = {
                        "verdict": str(verdict),
                        "reason": str(reason)
                    }


                    # ContextualRelevancyVerdicts requires
                    # "statement". Other DeepEval verdict schemas
                    # do not necessarily require it.
                    if "statement" in item or statement:
                        result["statement"] = str(statement)


                    normalized.append(result)


                parsed["verdicts"] = normalized


        # --------------------------------------------------------
        # VALIDATE AGAINST THE EXACT DEEPEVAL SCHEMA
        # --------------------------------------------------------


        try:


            return schema.model_validate(parsed)


        except Exception as first_error:


            # One repair attempt using Qwen itself.
            repair_prompt = f"""
Your previous JSON response did not match the required schema.


REQUIRED SCHEMA:


{json.dumps(schema.model_json_schema(), indent=2)}


ORIGINAL EVALUATION:


{prompt}


PREVIOUS JSON:


{json.dumps(parsed, indent=2)}


Fix the JSON.


STRICT RULES:


- Return ONLY valid JSON.
- Match the schema exactly.
- Do not omit required fields.
- Do not rename fields.
- If "statement" is required, include it.
- Copy each required statement from the original evaluation.
- Keep the original verdict and reason whenever possible.


Return the corrected JSON only.

"""


            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a JSON schema repair system. "
                        "Return only valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": repair_prompt
                }
            ]


            repair_text = self.tokenizer.apply_chat_template(
                repair_messages,
                tokenize=False,
                add_generation_prompt=True
            )


            repair_inputs = self.tokenizer(
                repair_text,
                return_tensors="pt"
            ).to(self.device)


            with torch.no_grad():


                repair_outputs = self.model.generate(
                    **repair_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )


            repair_response = self.tokenizer.decode(
                repair_outputs[0][
                    repair_inputs["input_ids"].shape[1]:
                ],
                skip_special_tokens=True
            ).strip()


            repair_response = (
                repair_response
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )


            repair_starts = [
                i for i in (
                    repair_response.find("{"),
                    repair_response.find("[")
                )
                if i >= 0
            ]


            if not repair_starts:


                raise ValueError(
                    f"Qwen JSON repair failed:\n{repair_response}"
                ) from first_error


            repaired = repair_json(
                repair_response[min(repair_starts):]
            )


            return schema.model_validate(repaired)


    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)


    def get_model_name(self):
        return self.model_name




# ------------------------------------------------------------
# RUN RAG PIPELINE
# ------------------------------------------------------------


def run(rag):


    with open(GOLDEN_PATH) as f:
        goldens = json.load(f)


    test_cases = []


    for g in goldens:


        result = rag.invoke(g["query"])


        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],
                retrieval_context=result["context"]
            )
        )


    print(f"Evaluating {len(test_cases)} test cases...")


    # ------------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------------


    judge_model = LocalQwenLLM()


    metrics = [
        ContextualRelevancyMetric(
            threshold=THRESHOLD,
            model=judge_model,
            include_reason=True
        ),
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=judge_model,
            include_reason=True
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=judge_model,
            include_reason=True
        )
    ]


    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        hyperparameters={
            "rag_pipeline": "RagPipeline",
            "judge_model": JUDGE_MODEL,
            "judge_device": "mps",
            "threshold": THRESHOLD,
            "golden_set": GOLDEN_PATH,
        }
    )


    return summarize_by_metric(result)




def run_local():
    return run(RagPipeline())




if __name__ == "__main__":
    run_local()