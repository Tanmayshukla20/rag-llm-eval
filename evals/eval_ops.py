# ============================================================================
# ==============================  COST  ======================================
# ============================================================================


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




# ============================================================================
# ==============================  LATENCY  ===================================
# ============================================================================



"""
Operational eval: LATENCY (with time-to-first-token).

Unlike correctness, faithfulness, toxicity, leakage, and scope evals,
latency needs no golden dataset and no LLM judge.

It is a deterministic measurement:
run the RAG pipeline multiple times, collect latency samples,
and report percentiles against an SLO.

Two latency numbers matter:
  - END-TO-END total : how long until the full answer is ready
  - TTFT             : how long until the first streamed token appears

For a streaming chat UI, TTFT is what "feels fast".

Key ideas:
  - perf_counter, not time()
  - many samples -> percentiles
  - discard warmup
  - decompose retrieval vs generation
  - measure TTFT
  - log answer length
  - single-user benchmark
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================

import math
import time

from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.generator import generate, generate_stream

load_dotenv()


# ============================================================
# 2. CONFIG
# ============================================================

QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 5
WARMUP_RUNS = 2

MEASURE_TTFT = True
STAGE_LEVEL = True

# SLOs / latency budgets
SLO_P95_MS = 3000
SLO_TTFT_P95_MS = 1200


# ============================================================
# 3. PIPELINE ADAPTERS
# ============================================================

def run_end_to_end(pipeline, question):
    result = pipeline.invoke(question)
    return result["answer"]


def run_stages(pipeline, question):
    t0 = time.perf_counter()

    docs = pipeline.retriever.invoke(question)
    context = [doc.page_content for doc in docs]

    t1 = time.perf_counter()

    answer = generate(question, context)

    t2 = time.perf_counter()

    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
    }


def run_stages_streaming(pipeline, question):
    t0 = time.perf_counter()

    docs = pipeline.retriever.invoke(question)
    context = [doc.page_content for doc in docs]

    t1 = time.perf_counter()

    first_token_t = None
    pieces = []

    for piece in generate_stream(question, context):

        if first_token_t is None:
            first_token_t = time.perf_counter()

        pieces.append(piece)

    t2 = time.perf_counter()

    answer = "".join(pieces)

    ttft_ms = (
        (first_token_t - t0) * 1000
        if first_token_t
        else float("nan")
    )

    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
        "ttft": ttft_ms,
    }


# ============================================================
# 4. PERCENTILE HELPER
# ============================================================

def percentile(values, p):

    values = [
        v for v in values
        if not math.isnan(v)
    ]

    if not values:
        return float("nan")

    s = sorted(values)

    k = (len(s) - 1) * (p / 100.0)

    lo = math.floor(k)
    hi = math.ceil(k)

    if lo == hi:
        return s[int(k)]

    return (
        s[lo] * (hi - k)
        + s[hi] * (k - lo)
    )


# ============================================================
# 5. BENCHMARK LOOP
# ============================================================

def benchmark(pipeline):

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(
        f"Warming up ({WARMUP_RUNS} runs, discarded)..."
    )

    for i in range(WARMUP_RUNS):
        run_end_to_end(
            pipeline,
            QUESTIONS[i % len(QUESTIONS)]
        )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    total_ms = []
    retrieval_ms = []
    generation_ms = []
    ttft_ms = []
    answer_lengths = []

    # --------------------------------------------------------
    # Measured runs
    # --------------------------------------------------------

    print("Measuring...")

    for question in QUESTIONS:

        for _ in range(REPEATS):

            start = time.perf_counter()

            if MEASURE_TTFT:

                answer, stage = run_stages_streaming(
                    pipeline,
                    question
                )

                retrieval_ms.append(
                    stage["retrieval"]
                )

                generation_ms.append(
                    stage["generation"]
                )

                ttft_ms.append(
                    stage["ttft"]
                )

            elif STAGE_LEVEL:

                answer, stage = run_stages(
                    pipeline,
                    question
                )

                retrieval_ms.append(
                    stage["retrieval"]
                )

                generation_ms.append(
                    stage["generation"]
                )

            else:

                answer = run_end_to_end(
                    pipeline,
                    question
                )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            total_ms.append(elapsed_ms)

            answer_lengths.append(
                len(answer or "")
            )

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths,
    }


# ============================================================
# 6. AGGREGATE + REPORT
# ============================================================

def summarize(samples):

    clean = [
        s for s in samples
        if not math.isnan(s)
    ]

    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }


def print_row(label, s):

    print(
        f"{label:<12} | "
        f"n={s['n']:<3} "
        f"mean={s['mean']:7.1f}  "
        f"p50={s['p50']:7.1f}  "
        f"p95={s['p95']:7.1f}  "
        f"p99={s['p99']:7.1f}  "
        f"min={s['min']:7.1f}  "
        f"max={s['max']:7.1f}"
    )


def slo_line(label, p95, budget):

    verdict = (
        "PASS"
        if p95 <= budget
        else "FAIL"
    )

    print(
        f"SLO: {label:<22} "
        f"p95 <= {budget:>5} ms  ->  "
        f"p95 = {p95:7.0f} ms   "
        f"[{verdict}]"
    )


def report(results):

    print("\n" + "=" * 78)
    print("LATENCY (milliseconds)")
    print("=" * 78)

    print(
        f"{'stage':<12} | "
        f"{'samples':<5} "
        f"{'mean':>11} "
        f"{'p50':>11} "
        f"{'p95':>11} "
        f"{'p99':>11} "
        f"{'min':>11} "
        f"{'max':>11}"
    )

    print("-" * 78)

    total = summarize(
        results["total"]
    )

    print_row(
        "end-to-end",
        total
    )

    if results["ttft"]:

        print_row(
            "ttft",
            summarize(results["ttft"])
        )

    if results["retrieval"]:

        print_row(
            "retrieval",
            summarize(results["retrieval"])
        )

        print_row(
            "generation",
            summarize(results["generation"])
        )

    avg_len = (
        sum(results["answer_len"])
        / len(results["answer_len"])
    )

    print("-" * 78)

    print(
        f"avg answer length: {avg_len:.0f} chars "
        f"(latency scales with output length)"
    )

    print("=" * 78)

    slo_line(
        "full answer",
        total["p95"],
        SLO_P95_MS
    )

    if results["ttft"]:

        slo_line(
            "first token (perceived)",
            summarize(results["ttft"])["p95"],
            SLO_TTFT_P95_MS
        )

    print("=" * 78)


# ============================================================
# 7. ENTRYPOINT
# ============================================================

def main():

    pipeline = RagPipeline()

    results = benchmark(
        pipeline
    )

    report(results)


if __name__ == "__main__":
    main()




   

# ============================================================================
# ==============================  RELIABILITY ================================
# ============================================================================



"""
Operational eval: RELIABILITY

Reliability measures whether the local RAG application can successfully
serve requests without failing.

We measure:
    - success rate
    - error rate
    - retry rate
    - retry attempts
    - average attempts per request

Retries are important because a system may eventually succeed while
still being flaky on the first attempt.

This evaluation uses the actual local RAG pipeline:
    Retriever -> Reranker -> Local Qwen Generator

No API calls or external services are required.
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
import time
from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline

load_dotenv()


# ============================================================
# 2. CONFIG
# ============================================================
QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 5

MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5


# ============================================================
# 3. RELIABILITY TRACKER
# ============================================================
class Reliability:

    def __init__(self):
        self.calls = 0
        self.successes = 0
        self.failures = 0

        # Total number of retry attempts across all requests.
        self.retry_attempts = 0

        # Number of requests that needed at least one retry.
        self.retried_requests = 0

        # Total attempts made, including the first attempt.
        self.total_attempts = 0


# ============================================================
# 4. RETRY WRAPPER
# ============================================================
def call_with_retries(fn, reliability):

    reliability.calls += 1

    request_retried = False

    for attempt in range(MAX_RETRIES + 1):

        reliability.total_attempts += 1

        try:
            result = fn()

            reliability.successes += 1

            if request_retried:
                reliability.retried_requests += 1

            return result

        except Exception as e:

            if attempt < MAX_RETRIES:

                reliability.retry_attempts += 1
                request_retried = True

                print(
                    f"Attempt {attempt + 1} failed. "
                    f"Retrying... ({e})"
                )

                # Exponential backoff.
                time.sleep(
                    BACKOFF_BASE_S * (2 ** attempt)
                )

            else:

                reliability.failures += 1

                if request_retried:
                    reliability.retried_requests += 1

                print(
                    f"FAILED after {MAX_RETRIES} retries: {e}"
                )

                return None


# ============================================================
# 5. BENCHMARK
# ============================================================
def benchmark(pipeline):

    reliability = Reliability()

    print("Measuring reliability...")

    for question in QUESTIONS:

        for _ in range(REPEATS):

            call_with_retries(
                lambda: pipeline.invoke(question),
                reliability
            )

    return reliability


# ============================================================
# 6. REPORT
# ============================================================
def report(rel):

    success_rate = (
        100 * rel.successes / rel.calls
        if rel.calls else 0
    )

    error_rate = (
        100 * rel.failures / rel.calls
        if rel.calls else 0
    )

    retry_rate = (
        100 * rel.retried_requests / rel.calls
        if rel.calls else 0
    )

    avg_attempts = (
        rel.total_attempts / rel.calls
        if rel.calls else 0
    )

    print("\n" + "=" * 60)
    print("RELIABILITY")
    print("=" * 60)

    print(f"total requests       : {rel.calls}")
    print(f"successful            : {rel.successes}")
    print(f"failed                : {rel.failures}")

    print("-" * 60)

    print(f"success rate          : {success_rate:.2f}%")
    print(f"error rate            : {error_rate:.2f}%")

    print(
        f"retried requests      : "
        f"{rel.retried_requests}"
    )

    print(
        f"retry rate            : "
        f"{retry_rate:.2f}%"
    )

    print(
        f"retry attempts        : "
        f"{rel.retry_attempts}"
    )

    print(
        f"total attempts        : "
        f"{rel.total_attempts}"
    )

    print(
        f"avg attempts/request  : "
        f"{avg_attempts:.2f}"
    )

    print("=" * 60)


# ============================================================
# 7. ENTRYPOINT
# ============================================================
def main():

    pipeline = RagPipeline()

    reliability = benchmark(
        pipeline
    )

    report(
        reliability
    )


if __name__ == "__main__":
    main()





