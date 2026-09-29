# evals/eval_rag_pipeline.py

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
            prompt = f"""
You are an LLM evaluator.

Return ONLY valid JSON matching this schema:

{json.dumps(schema.model_json_schema(), indent=2)}

Rules:
- Return only the JSON object.
- Use double quotes.
- Do not use markdown.
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

        response = response.removeprefix("```json").removeprefix("```").strip()
        response = response.removesuffix("```").strip()

        starts = [
            i for i in (response.find("{"), response.find("["))
            if i >= 0
        ]

        if not starts:
            raise ValueError(f"Qwen did not return JSON:\n{response}")

        parsed = repair_json(response[min(starts):])

        if "verdicts" in getattr(schema, "model_fields", {}):

            if isinstance(parsed, list):
                parsed = {"verdicts": parsed}

            elif isinstance(parsed, dict) and "verdict" in parsed:
                parsed = {"verdicts": [parsed]}

            if isinstance(parsed, dict) and isinstance(parsed.get("verdicts"), list):
                parsed["verdicts"] = [
                    {
                        "verdict": str(
                            item.get("verdict", "")
                            if isinstance(item, dict) else ""
                        ),
                        "reason": str(
                            item.get("reason", "")
                            if isinstance(item, dict) else ""
                        )
                    }
                    for item in parsed["verdicts"]
                ]

        return schema.model_validate(parsed)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


# ------------------------------------------------------------
# RUN RAG PIPELINE
# ------------------------------------------------------------

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)

rag = RagPipeline()
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

evaluate(
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