import os, re, glob, json, random
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.models import DeepEvalBaseLLM
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()


# ---------------------------------------------------------
# LOCAL LLM
# Replaces OpenAI GPT-4.1-mini
# ---------------------------------------------------------

class LocalQwenLLM(DeepEvalBaseLLM):

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        )

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):

        model = self.load_model()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant generating high-quality "
                    "evaluation questions and expected answers from provided "
                    "contexts. Follow the user's instructions exactly."
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
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return response

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Qwen2.5-1.5B-Instruct"


# ---------------------------------------------------------
# reuse your own VTT cleaning + chunking
# (same as the retriever)
# ---------------------------------------------------------

def load_chunks():

    texts = []

    for path in glob.glob("data/*.vtt"):

        with open(path) as f:

            lines = [
                ln.strip()
                for ln in f
                if ln.strip()
                and ln.strip() != "WEBVTT"
                and "-->" not in ln
            ]

        texts.append(" ".join(lines))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_text("\n\n".join(texts))


# ---------------------------------------------------------
# generate
# ---------------------------------------------------------

chunks = load_chunks()

sample = random.sample(
    chunks,
    min(15, len(chunks))
)

contexts = [
    [c]
    for c in sample
]


# ---------------------------------------------------------
# DeepEval Synthesizer
# ---------------------------------------------------------

local_llm = LocalQwenLLM()

synthesizer = Synthesizer(
    model=local_llm
)

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1
)


# ---------------------------------------------------------
# convert to YOUR schema
# id / query / ideal_answer / source
# ---------------------------------------------------------

rows = []

for i, g in enumerate(goldens, 1):

    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify"
    })


# ---------------------------------------------------------
# save
# ---------------------------------------------------------

with open(
    "goldens/retriever_deepeval_goldens.json",
    "w"
) as f:

    json.dump(
        rows,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"wrote {len(rows)} DRAFT goldens -> "
    f"goldens/retriever_deepeval_goldens.json"
)

print(
    "!! REVIEW EVERY ONE before using: "
    "check grounding, trim padding, fix leading questions."
)