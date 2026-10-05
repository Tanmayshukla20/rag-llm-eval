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
        result = rag.invoke(g["question"])

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=result["answer"],
                expected_output=g["ideal_answer"]
            )
        )

    judge = LocalQwenLLM()

    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare factual claims in the actual output with the expected output.",
            "Do not penalize brevity or missing details.",
            "Do not penalize additional correct information.",
            "Penalize factual errors and contradictions."
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome="All stated claims are factually correct."
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome="Mostly correct with a minor inaccuracy."
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome="Contains a clear factual error or contradiction."
            )
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        threshold=THRESHOLD,
        model=judge
    )

    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points in the expected output.",
            "Check how many are covered by the actual output.",
            "Penalize omitted or partially covered key points.",
            "Do not penalize additional information."
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome="Covers essentially all key points."
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome="Covers the main points but misses some."
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome="Misses several important points."
            )
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        threshold=THRESHOLD,
        model=judge
    )

    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the teaching style and tone.",
            "Reward clear, intuitive, conversational explanations.",
            "Reward plain language and briefly explained technical terms.",
            "Penalize stiff, robotic, jargon-heavy, or bare-list answers."
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
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
            )
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        threshold=THRESHOLD,
        model=judge
    )

    return evaluate(
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


if __name__ == "__main__":
    run(RagPipeline())