import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel
import torch

# Load env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_MODEL = "google/flan-t5-base"
ADAPTER = "adapters/law_adapter"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running evaluation on: {DEVICE}")

# Load tokenizer + base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    device_map="auto" if DEVICE=="cuda" else {"": "cpu"},
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

# Pipeline for seq2seq generation
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE=="cuda" else -1
)

# Interactive dynamic evaluation
print("🔹 Enter 'exit' to quit.")
while True:
    prompt = input("\nEnter your law question: ")
    if prompt.lower() in ["exit", "quit"]:
        break
    output = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    print("Answer:", output[0]['generated_text'])
