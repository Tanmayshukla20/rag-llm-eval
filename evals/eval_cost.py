"""
Operational eval: COST.

Cost is derived from token usage.

The current RAG system uses a fully local Hugging Face Qwen model, so there is
NO actual API/token charge. This evaluation therefore measures:

  1. Input tokens
  2. Output tokens
  3. Total tokens
  4. Actual local API cost ($0)
  5. Optional equivalent hosted-model cost

Why cost is useful OFFLINE:
  - input token counts are determined by the retrieved context + prompt.
  - output token counts are reasonably stable with greedy decoding.
  - token usage lets us estimate how expensive the same workload would be if
    the system were later moved to a paid hosted model.

Important:
  - Qwen runs locally, so cached provider tokens do not apply.
  - There is no provider-side cache cost to measure.
  - The hosted prices below are reference prices only.
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
import time

from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.generator import (
    MODEL_NAME,
    tokenizer,
    build_prompt,
    generate,
)

load_dotenv()

# Reuse the exact tokenizer used by the generator.
TOKENIZER = tokenizer

# ============================================================
# 2. CONFIG
# ============================================================
QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 3

# --- Current local model ---
MODEL = MODEL_NAME

# --- Actual local cost ---
# Qwen is running locally, so there is no API/token charge.
LOCAL_COST_PER_QUERY_USD = 0.0

# --- Optional hosted-model reference pricing ---
# These prices are ONLY used to estimate what the same token usage
# would cost if the application were moved to a paid API.
#
# Change these values when comparing against a specific provider/model.
REFERENCE_INPUT_PRICE_PER_1M = 0.15
REFERENCE_OUTPUT_PRICE_PER_1M = 0.60

# --- Business projection knobs ---
QUERIES_PER_DAY = 2000
USD_TO_INR = 96.0

# --- Budget for the equivalent hosted model ---
REFERENCE_COST_BUDGET_PER_QUERY_USD = 0.0015

# ============================================================
# 3. TOKEN MEASUREMENT
# ============================================================
# Retrieve the real context and tokenize the exact prompt used
# by src.generator.py.
def measure_tokens(pipeline, question):

    # Retrieve using the actual RAG retriever.
    docs = pipeline.retriever.invoke(question)

    # Generator expects a list of strings.
    context = [doc.page_content for doc in docs]

    # Build the exact same prompt used by generate().
    prompt_text = build_prompt(
        question,
        context
    )

    # Apply the model's chat template exactly as generate() does.
    formatted_prompt = TOKENIZER.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    # Count the actual input tokens sent to Qwen.
    input_encoding = TOKENIZER(
        formatted_prompt,
        return_tensors="pt",
        truncation=False
    )

    input_tokens = input_encoding["input_ids"].shape[1]

    # Generate using the actual generator.
    start = time.perf_counter()

    answer = generate(
        question,
        context
    )

    generation_time = time.perf_counter() - start

    # Count tokens in the generated answer.
    output_encoding = TOKENIZER(
        answer,
        return_tensors="pt",
        truncation=False
    )

    output_tokens = output_encoding["input_ids"].shape[1]

    total_tokens = input_tokens + output_tokens

    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": total_tokens,
        "generation_time": generation_time,
    }

# ============================================================
# 4. COST MATH
# ============================================================
def reference_cost_usd(input_tokens, output_tokens):

    input_cost = (
        input_tokens
        / 1_000_000
        * REFERENCE_INPUT_PRICE_PER_1M
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * REFERENCE_OUTPUT_PRICE_PER_1M
    )

    return {
        "input": input_cost,
        "output": output_cost,
        "total": input_cost + output_cost,
    }

# ============================================================
# 5. BENCHMARK LOOP
# ============================================================
def benchmark(pipeline):

    rows = []

    print("Measuring local token usage...")

    for question in QUESTIONS:

        for _ in range(REPEATS):

            tokens = measure_tokens(
                pipeline,
                question
            )

            reference_cost = reference_cost_usd(
                tokens["input"],
                tokens["output"]
            )

            rows.append({
                **tokens,

                "local_cost": LOCAL_COST_PER_QUERY_USD,

                "reference_cost_input":
                    reference_cost["input"],

                "reference_cost_output":
                    reference_cost["output"],

                "reference_cost_total":
                    reference_cost["total"],
            })

    return rows

# ============================================================
# 6. AGGREGATE + REPORT
# ============================================================
def avg(rows, key):

    return sum(
        row[key]
        for row in rows
    ) / len(rows)


def report(rows):

    n = len(rows)

    # --- Token statistics ---
    avg_input = avg(
        rows,
        "input"
    )

    avg_output = avg(
        rows,
        "output"
    )

    avg_total = avg(
        rows,
        "total"
    )

    min_total = min(
        row["total"]
        for row in rows
    )

    max_total = max(
        row["total"]
        for row in rows
    )

    # --- Generation time ---
    avg_generation_time = avg(
        rows,
        "generation_time"
    )

    # --- Actual local cost ---
    avg_local_cost = avg(
        rows,
        "local_cost"
    )

    # --- Reference hosted cost ---
    avg_reference_input = avg(
        rows,
        "reference_cost_input"
    )

    avg_reference_output = avg(
        rows,
        "reference_cost_output"
    )

    avg_reference_cost = avg(
        rows,
        "reference_cost_total"
    )

    min_reference_cost = min(
        row["reference_cost_total"]
        for row in rows
    )

    max_reference_cost = max(
        row["reference_cost_total"]
        for row in rows
    )

    # Percentage of reference bill caused by output tokens.
    output_share = (
        100 * avg_reference_output / avg_reference_cost
        if avg_reference_cost
        else 0
    )

    print("\n" + "=" * 70)
    print(f"COST / TOKEN USAGE  ({MODEL})")
    print("=" * 70)

    print(f"samples                : {n}")

    print(f"avg input tokens       : {avg_input:8.0f}")
    print(f"avg output tokens      : {avg_output:8.0f}")
    print(f"avg total tokens       : {avg_total:8.0f}")

    print(
        f"min / max total tokens : "
        f"{min_total:.0f} / {max_total:.0f}"
    )

    print("-" * 70)

    print(
        f"avg generation time    : "
        f"{avg_generation_time:8.2f}s"
    )

    print("-" * 70)

    # --- Actual local cost ---
    print("LOCAL MODEL")
    print(
        f"avg cost / query       : "
        f"${avg_local_cost:.6f}   "
        f"(Rs {avg_local_cost * USD_TO_INR:.4f})"
    )

    print(
        "note: local Qwen has no API/token charge."
    )

    print("-" * 70)

    # --- Hosted reference cost ---
    print("HOSTED MODEL REFERENCE")

    print(
        f"input price            : "
        f"${REFERENCE_INPUT_PRICE_PER_1M:.2f} / 1M tokens"
    )

    print(
        f"output price           : "
        f"${REFERENCE_OUTPUT_PRICE_PER_1M:.2f} / 1M tokens"
    )

    print(
        f"avg input cost/query   : "
        f"${avg_reference_input:.6f}"
    )

    print(
        f"avg output cost/query  : "
        f"${avg_reference_output:.6f}"
    )

    print(
        f"avg reference cost     : "
        f"${avg_reference_cost:.6f}   "
        f"(Rs {avg_reference_cost * USD_TO_INR:.4f})"
    )

    print(
        f"min / max reference    : "
        f"${min_reference_cost:.6f} / "
        f"${max_reference_cost:.6f}"
    )

    print(
        f"input vs output cost   : "
        f"{100 - output_share:.0f}% input / "
        f"{output_share:.0f}% output"
    )

    print("-" * 70)

    # --- Business projection ---
    daily = (
        avg_reference_cost
        * QUERIES_PER_DAY
    )

    monthly = daily * 30

    print(
        f"projection @ "
        f"{QUERIES_PER_DAY}/day :"
    )

    print(
        f"   per day             : "
        f"${daily:8.2f}   "
        f"(Rs {daily * USD_TO_INR:8.2f})"
    )

    print(
        f"   per month           : "
        f"${monthly:8.2f}   "
        f"(Rs {monthly * USD_TO_INR:8.2f})"
    )

    print("=" * 70)

    # --- Budget verdict ---
    verdict = (
        "PASS"
        if avg_reference_cost
        <= REFERENCE_COST_BUDGET_PER_QUERY_USD
        else "FAIL"
    )

    print(
        f"BUDGET: reference cost/query <= "
        f"${REFERENCE_COST_BUDGET_PER_QUERY_USD:.6f}"
        f"  ->  "
        f"${avg_reference_cost:.6f}"
        f"   [{verdict}]"
    )

    print("=" * 70)

    print(
        "note: reference cost is an estimate only."
    )

    print(
        "      Actual hosted pricing depends on the provider,"
        " model, caching, and billing rules."
    )

# ============================================================
# 7. ENTRYPOINT
# ============================================================
def main():

    pipeline = RagPipeline()

    rows = benchmark(
        pipeline
    )

    report(
        rows
    )


if __name__ == "__main__":
    main()