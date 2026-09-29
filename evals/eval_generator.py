"""
evals/eval_generator.py

Evaluates the GENERATOR in isolation using golden context.

Generator: Qwen/Qwen2.5-0.5B-Instruct
Judge:     Qwen/Qwen2.5-1.5B-Instruct

Metrics:
- Faithfulness
- Answer Relevancy

Run:
    python3 -m evals.eval_generator
"""

import json
import torch

from dotenv import load_dotenv
from json_repair import loads as repair_json
from transformers import AutoTokenizer, AutoModelForCausalLM

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import DeepEvalBaseLLM

from src.generator import generate

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
THRESHOLD = 0.7


# ------------------------------------------------------------
# LOCAL QWEN JUDGE
# ------------------------------------------------------------

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

        if schema is not None:
            prompt = f"""
You are an LLM evaluator.

Return ONLY valid JSON matching this schema:

{json.dumps(schema.model_json_schema(), indent=2)}

Rules:
- Return only the JSON object.
- Use double quotes.
- Do not use markdown or ```json.
- Do not add explanations outside the JSON.
- Follow the schema exactly.

Evaluation prompt:

{prompt}
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise LLM evaluator. "
                    "When JSON is requested, return only valid JSON."
                )
            },
            {"role": "user", "content": prompt}
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
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        if schema is None:
            return response

        # Remove markdown fences.
        response = response.removeprefix("```json").removeprefix("```").strip()
        response = response.removesuffix("```").strip()

        # Find JSON.
        starts = [i for i in (response.find("{"), response.find("[")) if i >= 0]

        if not starts:
            raise ValueError(f"Qwen did not return JSON:\n{response}")

        parsed = repair_json(response[min(starts):])

        # Normalize DeepEval verdict output.
        if "verdicts" in getattr(schema, "model_fields", {}):

            if isinstance(parsed, list):
                parsed = {"verdicts": parsed}

            elif isinstance(parsed, dict) and "verdict" in parsed:
                parsed = {"verdicts": [parsed]}

            if isinstance(parsed, dict) and isinstance(parsed.get("verdicts"), list):
                normalized = []

                for item in parsed["verdicts"]:
                    if not isinstance(item, dict):
                        continue

                    verdict = item.get("verdict", "")
                    reason = item.get("reason", "")

                    # Handle nested verdict produced by small local models.
                    if isinstance(verdict, dict):
                        reason = verdict.get("reason", reason)
                        verdict = verdict.get("verdict", "")

                    normalized.append({
                        "verdict": str(verdict),
                        "reason": str(reason)
                    })

                parsed["verdicts"] = normalized

        try:
            return schema.model_validate(parsed)

        except Exception as e:
            raise ValueError(
                "Qwen JSON did not match DeepEval schema.\n\n"
                f"Parsed:\n{json.dumps(parsed, indent=2)}\n\n"
                f"Error:\n{e}"
            ) from e

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


# ------------------------------------------------------------
# LOAD GOLDENS + RUN GENERATOR
# ------------------------------------------------------------

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)

test_cases = []

for g in goldens:
    context = g["ideal_context"]
    answer = generate(g["query"], context)

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=answer,
            retrieval_context=context
        )
    )

print(f"Evaluating {len(test_cases)} test cases...")


# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------

judge = LocalQwenLLM()

metrics = [
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True
    ),
    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True
    )
]

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=True),
    hyperparameters={
        "generator": "Qwen/Qwen2.5-0.5B-Instruct",
        "generator_device": "mps",
        "generator_dtype": "float16",
        "judge_model": JUDGE_MODEL,
        "judge_device": "mps",
        "judge_dtype": "float16",
        "max_new_tokens": 128,
        "threshold": THRESHOLD,
        "golden_set": GOLDEN_PATH
    }
)