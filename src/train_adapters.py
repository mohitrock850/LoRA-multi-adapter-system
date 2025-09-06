import sys, os
import yaml
import torch
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, BitsAndBytesConfig,
    DataCollatorForSeq2Seq, default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType
from utils import load_jsonl_as_dataset, tokenize_for_causal, tokenize_for_seq2seq

# -------------------
# Setup project root
# -------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Load .env if present
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# -------------------
# Load configs safely
# -------------------
with open(os.path.join(BASE_DIR, "configs", "base.yaml"), "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

with open(os.path.join(BASE_DIR, "configs", "adapters.yaml"), "r", encoding="utf-8") as f:
    adp_cfg = yaml.safe_load(f)


def train_adapter(adapter_name: str, sample_size: int = None):
    adapter_info = adp_cfg["adapters"][adapter_name]

    # Fix dataset + eval paths (relative to BASE_DIR)
    dataset_path = os.path.join(BASE_DIR, adapter_info["dataset"])
    eval_path = os.path.join(BASE_DIR, adapter_info["eval"]) if adapter_info.get("eval") else None
    output_path = os.path.join(BASE_DIR, adapter_info["path"])

    base_model = base_cfg.get("base_model", "google/flan-t5-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, base_model: {base_model}")

    # choose seq2seq vs causal depending on model type (simple heuristic)
    is_seq2seq = "t5" in base_model or "flan" in base_model.lower()
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=HF_TOKEN)

    # -------------------
    # Load dataset
    # -------------------
    ds = load_jsonl_as_dataset(dataset_path, eval_path)
    
    # Use sample size from base_cfg if not provided as an argument
    if not sample_size:
        sample_size = base_cfg.get("dataset", {}).get("sample_size", 2000)
    
    ds["train"] = ds["train"].select(range(min(len(ds["train"]), sample_size)))
    if "eval" in ds:
        eval_size = base_cfg.get("dataset", {}).get("eval_size", 500)
        ds["eval"] = ds["eval"].select(range(min(len(ds["eval"]), eval_size)))

    # -------------------
    # Model + Tokenization
    # -------------------
    max_len = base_cfg.get("dataset", {}).get("max_length", 512)
    if is_seq2seq:
        tok_ds = tokenize_for_seq2seq(ds, tokenizer, max_length=max_len)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            token=HF_TOKEN,
            device_map="auto" if device == "cuda" else {"": "cpu"}
        )
    else: # is Causal
        tok_ds = tokenize_for_causal(ds, tokenizer, max_length=max_len)
        quant_cfg = None
        if device == "cuda" and base_cfg.get("use_4bit", False): # Check for 4-bit config
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=HF_TOKEN,
            quantization_config=quant_cfg,
            device_map="auto" if device == "cuda" else {"": "cpu"},
            trust_remote_code=True
        )

    # -------------------
    # LoRA Config
    # -------------------
    # MODIFIED: Load PEFT config from base_cfg instead of adp_cfg
    peft_cfg = base_cfg["peft"]
    lora_config = LoraConfig(
        r=peft_cfg["r"],
        lora_alpha=peft_cfg["lora_alpha"],
        lora_dropout=peft_cfg["lora_dropout"],
        target_modules=peft_cfg["target_modules"],
        bias=peft_cfg.get("bias", "none"),
        task_type=TaskType.SEQ_2_SEQ_LM if is_seq2seq else TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------
    # Training Args
    # -------------------
    train_cfg = base_cfg["training"]

    # DEBUG: Check the learning rate value and type right before it's used
    lr_value = train_cfg.get("learning_rate", 2e-4)
    print("--- DEBUG INFO ---")
    print(f"DEBUG: Loaded learning_rate value is: {lr_value}")
    print(f"DEBUG: The type of this value is: {type(lr_value)}")
    print("--------------------")

    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(lr_value),  # Failsafe: ensure it's a float
        num_train_epochs=train_cfg["num_train_epochs"],
        save_strategy=train_cfg["save_strategy"],
        logging_dir=train_cfg["logging_dir"],
        logging_steps=train_cfg["logging_steps"],
        report_to=train_cfg.get("report_to", "none")
    )

    # -------------------
    # Trainer Setup
    # -------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model) if is_seq2seq else default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds.get("eval"),
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"✅ Saved adapter at {output_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, choices=list(adp_cfg["adapters"].keys()))
    p.add_argument("--sample_size", type=int, default=None) # Default to None to use config value
    args = p.parse_args()
    train_adapter(args.adapter, sample_size=args.sample_size)