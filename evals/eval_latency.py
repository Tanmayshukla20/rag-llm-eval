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