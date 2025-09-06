# src/evaluate.py
import argparse
import yaml  # NEW: Import yaml to read config
import os    # NEW: Import os for path handling
from tqdm import tqdm
from src.models import MultiAdapterModel
# MODIFIED: Import the correct data loader from utils.py
from src.utils import load_jsonl_as_dataset
from src.metrics import exact_match, bleu_score

# -------------------
# Setup project root
# -------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# -------------------
# CLI Argument
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--adapter", choices=["law", "med", "code"], required=True)
parser.add_argument("--eval", action="store_true", help="Run evaluation on dataset instead of interactive mode")
parser.add_argument("--sample_size", type=int, default=50, help="Number of samples to evaluate on")
args = parser.parse_args()

# -------------------
# Load model + adapter
# -------------------
print("Loading model and adapter...")
mm = MultiAdapterModel()
model = mm.set_adapter(args.adapter)
tokenizer = mm.tokenizer
print("Model loaded successfully.")

# -------------------
# Evaluation Mode
# -------------------
if args.eval:
    # NEW: Load adapters config to find the evaluation file path
    with open(os.path.join(BASE_DIR, "configs", "adapters.yaml"), "r") as f:
        adp_cfg = yaml.safe_load(f)

    # Get the path for the evaluation dataset
    eval_path = adp_cfg["adapters"][args.adapter]["eval"]
    full_eval_path = os.path.join(BASE_DIR, eval_path)

    # MODIFIED: Call the correct data loading function
    # The load_jsonl_as_dataset function expects a 'train_path', so we pass our eval path to it.
    # It returns a DatasetDict, so we access the 'train' key to get our dataset.
    dataset_dict = load_jsonl_as_dataset(train_path=full_eval_path)
    dataset = dataset_dict['train'].select(range(min(len(dataset_dict['train']), args.sample_size)))

    # Use the correct keys for prompts and references
    prompts = [
        f"{ex['instruction']}\n{ex['input']}" if ex['input'] else ex['instruction']
        for ex in dataset
    ]
    refs = [ex["output"] for ex in dataset]

    # Use batch processing for faster inference
    print(f"Generating predictions for {len(prompts)} samples...")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=200)
    preds = tokenizer.batch_decode(out, skip_special_tokens=True)

    em = exact_match(preds, refs)
    bleu = bleu_score(preds, refs)

    print(f"\n📊 Evaluation for {args.adapter.upper()} adapter")
    print(f"Sample Size: {len(prompts)}")
    print(f"Exact Match: {em:.2f}")
    print(f"BLEU Score : {bleu:.2f}")

else:
    # -------------------
    # Interactive Mode
    # -------------------
    print(f"\n✅ Loaded {args.adapter.upper()} adapter for interactive mode. Enter 'exit' to quit.")
    while True:
        q = input("Question > ")
        if q.lower() in ("exit", "quit"):
            break
        inputs = tokenizer(q, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=200)
        print("Answer:", tokenizer.decode(out[0], skip_special_tokens=True))