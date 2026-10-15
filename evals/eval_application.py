<<<<<<< HEAD
# # eval_application.py

# import json
# import torch

# from json_repair import loads as repair_json
# from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModelForCausalLM

# from deepeval import evaluate
# from deepeval.evaluate import AsyncConfig
# from deepeval.test_case import LLMTestCase, LLMTestCaseParams
# from deepeval.metrics import GEval
# from deepeval.metrics.g_eval import Rubric
# from deepeval.models import DeepEvalBaseLLM

# from src.rag_pipeline import RagPipeline

# load_dotenv()

# GOLDEN_PATH = "goldens/correctness_goldens.json"
# JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# THRESHOLD = 0.7


# class LocalQwenLLM(DeepEvalBaseLLM):

#     def __init__(self):
#         self.model_name = JUDGE_MODEL
#         self.device = "mps" if torch.backends.mps.is_available() else "cpu"

#         print("Judge device:", self.device)

#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             dtype=torch.float16
#         ).to(self.device)

#         self.model.eval()

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str, schema=None):

#         if schema:
#             prompt = f"""
# You are an LLM evaluator.

# Return ONLY valid JSON matching this schema:

# {json.dumps(schema.model_json_schema(), indent=2)}

# Rules:
# - Return only JSON.
# - Use double quotes.
# - Follow the schema exactly.
# - Do not use markdown.

# Evaluation prompt:

# {prompt}
# """

#         messages = [
#             {
#                 "role": "system",
#                 "content": "You are a precise LLM evaluator. Return valid JSON when requested."
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
#                 max_new_tokens=128,
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

#         start = response.find("{")

#         if start == -1:
#             raise ValueError(f"Qwen did not return JSON:\n{response}")

#         parsed = repair_json(response[start:])

#         return schema.model_validate(parsed)

#     async def a_generate(self, prompt: str, schema=None):
#         return self.generate(prompt, schema)

#     def get_model_name(self):
#         return self.model_name


# def run(rag):

#     with open(GOLDEN_PATH) as f:
#         goldens = json.load(f)

#     test_cases = []

#     for g in goldens:
#         result = rag.invoke(g["question"])

#         test_cases.append(
#             LLMTestCase(
#                 input=g["question"],
#                 actual_output=result["answer"],
#                 expected_output=g["ideal_answer"]
#             )
#         )

#     judge = LocalQwenLLM()

#     correctness = GEval(
#         name="Correctness",
#         evaluation_steps=[
#             "Compare factual claims in the actual output with the expected output.",
#             "Do not penalize brevity or missing details.",
#             "Do not penalize additional correct information.",
#             "Penalize factual errors and contradictions."
#         ],
#         rubric=[
#             Rubric(
#                 score_range=(9, 10),
#                 expected_outcome="All stated claims are factually correct."
#             ),
#             Rubric(
#                 score_range=(5, 8),
#                 expected_outcome="Mostly correct with a minor inaccuracy."
#             ),
#             Rubric(
#                 score_range=(0, 4),
#                 expected_outcome="Contains a clear factual error or contradiction."
#             )
#         ],
#         evaluation_params=[
#             LLMTestCaseParams.INPUT,
#             LLMTestCaseParams.ACTUAL_OUTPUT,
#             LLMTestCaseParams.EXPECTED_OUTPUT
#         ],
#         threshold=THRESHOLD,
#         model=judge
#     )

#     completeness = GEval(
#         name="Completeness",
#         evaluation_steps=[
#             "Identify the key points in the expected output.",
#             "Check how many are covered by the actual output.",
#             "Penalize omitted or partially covered key points.",
#             "Do not penalize additional information."
#         ],
#         rubric=[
#             Rubric(
#                 score_range=(9, 10),
#                 expected_outcome="Covers essentially all key points."
#             ),
#             Rubric(
#                 score_range=(5, 8),
#                 expected_outcome="Covers the main points but misses some."
#             ),
#             Rubric(
#                 score_range=(0, 4),
#                 expected_outcome="Misses several important points."
#             )
#         ],
#         evaluation_params=[
#             LLMTestCaseParams.INPUT,
#             LLMTestCaseParams.ACTUAL_OUTPUT,
#             LLMTestCaseParams.EXPECTED_OUTPUT
#         ],
#         threshold=THRESHOLD,
#         model=judge
#     )

#     style = GEval(
#         name="Style",
#         evaluation_steps=[
#             "Judge only the teaching style and tone.",
#             "Reward clear, intuitive, conversational explanations.",
#             "Reward plain language and briefly explained technical terms.",
#             "Penalize stiff, robotic, jargon-heavy, or bare-list answers."
#         ],
#         rubric=[
#             Rubric(
#                 score_range=(9, 10),
#                 expected_outcome="Intuitive, conversational teaching style."
#             ),
#             Rubric(
#                 score_range=(7, 8),
#                 expected_outcome="Clear, conversational, well-explained prose."
#             ),
#             Rubric(
#                 score_range=(4, 6),
#                 expected_outcome="Understandable but somewhat formal or flat."
#             ),
#             Rubric(
#                 score_range=(0, 3),
#                 expected_outcome="Dry, robotic, jargon-heavy, or poorly explained."
#             )
#         ],
#         evaluation_params=[
#             LLMTestCaseParams.INPUT,
#             LLMTestCaseParams.ACTUAL_OUTPUT
#         ],
#         threshold=THRESHOLD,
#         model=judge
#     )

#     return evaluate(
#         test_cases=test_cases,
#         metrics=[correctness, completeness, style],
#         async_config=AsyncConfig(run_async=False),
#         hyperparameters={
#             "judge_model": JUDGE_MODEL,
#             "judge_device": "mps",
#             "threshold": THRESHOLD,
#             "golden_set": GOLDEN_PATH
#         }
#     )


# if __name__ == "__main__":
#     run(RagPipeline())








=======
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
# eval_application.py

import json
import torch

from json_repair import loads as repair_json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import DeepEvalBaseLLM

from src.rag_pipeline import RagPipeline
<<<<<<< HEAD
from evals.harness import summarize_by_metric
=======
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e

load_dotenv()

GOLDEN_PATH = "goldens/correctness_goldens.json"
JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
THRESHOLD = 0.7


class LocalQwenLLM(DeepEvalBaseLLM):

    def __init__(self):
        self.model_name = JUDGE_MODEL
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        print("Judge device:", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
<<<<<<< HEAD

=======
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16
        ).to(self.device)

        self.model.eval()

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):

        if schema:
            prompt = f"""
You are an LLM evaluator.

Return ONLY valid JSON matching this schema:

{json.dumps(schema.model_json_schema(), indent=2)}

Rules:
- Return only JSON.
- Use double quotes.
- Follow the schema exactly.
- Do not use markdown.

Evaluation prompt:

{prompt}
"""

        messages = [
            {
                "role": "system",
                "content": "You are a precise LLM evaluator. Return valid JSON when requested."
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
<<<<<<< HEAD

=======
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
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

        start = response.find("{")

        if start == -1:
            raise ValueError(f"Qwen did not return JSON:\n{response}")

        parsed = repair_json(response[start:])

        return schema.model_validate(parsed)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


def run(rag):

    with open(GOLDEN_PATH) as f:
        goldens = json.load(f)

    test_cases = []

    for g in goldens:
<<<<<<< HEAD

=======
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
        result = rag.invoke(g["question"])

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=result["answer"],
                expected_output=g["ideal_answer"]
            )
        )

    judge = LocalQwenLLM()

<<<<<<< HEAD

    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare the actual answer with the expected answer.",
            "Treat the expected answer as the reference answer.",
            "Check whether the actual answer correctly addresses the question.",
            "Do not require the wording to exactly match the expected answer.",
            "Do not give credit for unsupported claims.",
            "Judge factual correctness only."
=======
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare factual claims in the actual output with the expected output.",
            "Do not penalize brevity or missing details.",
            "Do not penalize additional correct information.",
            "Penalize factual errors and contradictions."
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
<<<<<<< HEAD
                expected_outcome=(
                    "The answer is fully correct and directly addresses "
                    "the question without substantive factual errors."
                )
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome=(
                    "The answer is mostly correct but contains a minor "
                    "error, omission, or imprecision."
                )
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "The answer is substantially incorrect, contradicts "
                    "the expected answer, or fails to answer the question."
                )
=======
                expected_outcome="All stated claims are factually correct."
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome="Mostly correct with a minor inaccuracy."
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome="Contains a clear factual error or contradiction."
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
            )
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        threshold=THRESHOLD,
<<<<<<< HEAD
        model=judge,
        strict_mode=False
    )


    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Compare the actual answer with the expected answer.",
            "Identify the distinct information required to answer the question.",
            "Check whether the actual answer covers the important points present in the expected answer.",
            "Do not penalize different wording when the same information is conveyed.",
            "Judge completeness only."
=======
        model=judge
    )

    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points in the expected output.",
            "Check how many are covered by the actual output.",
            "Penalize omitted or partially covered key points.",
            "Do not penalize additional information."
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
<<<<<<< HEAD
                expected_outcome=(
                    "The answer covers all important points needed to "
                    "answer the question."
                )
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome=(
                    "The answer covers most important points but misses "
                    "one or more relevant details."
                )
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "The answer misses major parts of the expected answer "
                    "or provides very little of the required information."
                )
=======
                expected_outcome="Covers essentially all key points."
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome="Covers the main points but misses some."
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome="Misses several important points."
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
            )
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        threshold=THRESHOLD,
<<<<<<< HEAD
        model=judge,
        strict_mode=False
    )


    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge whether the answer is clear and understandable.",
            "Check whether the answer directly addresses the student's question.",
            "Check whether the answer uses a professional and appropriate teaching tone.",
            "Do not judge factual correctness or completeness.",
            "Judge writing style only."
=======
        model=judge
    )

    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the teaching style and tone.",
            "Reward clear, intuitive, conversational explanations.",
            "Reward plain language and briefly explained technical terms.",
            "Penalize stiff, robotic, jargon-heavy, or bare-list answers."
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
<<<<<<< HEAD
                expected_outcome=(
                    "The answer is clear, concise, professional, and "
                    "easy to understand."
                )
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome=(
                    "The answer is generally clear and professional but "
                    "has minor style or clarity issues."
                )
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "The answer is unclear, poorly written, unprofessional, "
                    "or difficult to understand."
                )
=======
                expected_outcome="Intuitive, conversational teaching style."
            ),
            Rubric(
                score_range=(7, 8),
                expected_outcome="Clear, conversational, well-explained prose."
            ),
            Rubric(
                score_range=(4, 6),
                expected_outcome="Understandable but somewhat formal or flat."
            ),
            Rubric(
                score_range=(0, 3),
                expected_outcome="Dry, robotic, jargon-heavy, or poorly explained."
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
            )
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        threshold=THRESHOLD,
<<<<<<< HEAD
        model=judge,
        strict_mode=False
    )


    result = evaluate(
=======
        model=judge
    )

    return evaluate(
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
        test_cases=test_cases,
        metrics=[correctness, completeness, style],
        async_config=AsyncConfig(run_async=False),
        hyperparameters={
            "judge_model": JUDGE_MODEL,
            "judge_device": "mps",
            "threshold": THRESHOLD,
            "golden_set": GOLDEN_PATH
        }
    )

<<<<<<< HEAD
    return summarize_by_metric(result)


def run_local():
    return run(RagPipeline())


if __name__ == "__main__":
    run_local()
=======

if __name__ == "__main__":
    run(RagPipeline())
>>>>>>> c21718680dba0790819097c8206099b1721f0e0e
