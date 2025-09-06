import json
from datasets import Dataset, DatasetDict, load_dataset


# ------------------------
# Load JSONL dataset
# ------------------------
def load_jsonl_as_dataset(train_path, eval_path=None):
    """
    Load a dataset from JSONL files into a HuggingFace DatasetDict.
    MODIFIED: Expects each line to have {"instruction": ..., "input": ..., "output": ...}
    """
    def load_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f]

    ds = {"train": Dataset.from_list(load_file(train_path))}
    if eval_path:
        ds["eval"] = Dataset.from_list(load_file(eval_path))
    return DatasetDict(ds)


# ------------------------
# Tokenization helpers
# ------------------------
def tokenize_for_causal(ds, tokenizer, max_length=512):
    """
    Tokenize dataset for causal LM.
    MODIFIED: Handles the Alpaca-style format {"instruction", "input", "output"}.
    """
    def tok(batch):
        # Combine instruction, input, and output for Causal LM training
        source_text = [
            f"{instr}\n{inp}\n{out}" if inp else f"{instr}\n{out}"
            for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
        ]
        
        tokens = tokenizer(
            source_text,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        # labels = input_ids for causal LM
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return ds.map(tok, batched=True, remove_columns=ds["train"].column_names)


def tokenize_for_seq2seq(ds, tokenizer, max_length=512):
    """
    Tokenize dataset for seq2seq (e.g. T5, FLAN).
    MODIFIED: Handles the Alpaca-style format {"instruction", "input", "output"}.
    """
    def tok(batch):
        # Combine the 'instruction' and 'input' fields to create the source text
        source_text = [
            f"{instruction}\n{inp}" if inp else instruction
            for instruction, inp in zip(batch["instruction"], batch["input"])
        ]
        
        model_inputs = tokenizer(
            source_text,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        
        labels = tokenizer(
            text_target=batch["output"], # Use 'output' as the target text
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return ds.map(tok, batched=True, remove_columns=ds["train"].column_names)


# ------------------------
# Domain dataset loader (from your request)
# ------------------------
def load_dataset_for_domain(domain, tokenizer, split="train", sample_size=2000, max_length=512):
    """
    Load and tokenize dataset for a specific domain (law, med, code).
    """
    if domain == "law":
        dataset = load_dataset("pile-of-law/pile-of-law", split=split)

    elif domain == "med":
        dataset = load_dataset("pubmed_qa", "pqa_labeled", split=split)
        dataset = dataset.map(
            lambda x: {"text": x["question"] + " " + " ".join(x["long_answer"])
                       if "long_answer" in x else x["question"]}
        )

    elif domain == "code":
        dataset = load_dataset("codeparrot/codeparrot-clean", split=split)

    else:
        raise ValueError(f"Unknown domain: {domain}. Choose from 'law', 'med', 'code'.")

    # Take subset
    dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), sample_size)))

    # Tokenization
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset