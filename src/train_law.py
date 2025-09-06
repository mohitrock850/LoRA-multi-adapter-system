import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from utils import load_law_dataset
import torch

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_MODEL = "google/flan-t5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {DEVICE}")

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    device_map="auto" if DEVICE == "cuda" else {"": "cpu"},
    trust_remote_code=True
)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],  # Flan-T5 uses q/v projections
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

# Load dataset (subset for faster testing)
train_dataset = load_law_dataset(tokenizer, split="train", sample_size=2000)
eval_dataset = load_law_dataset(tokenizer, split="train", sample_size=500)

# Training arguments
training_args = TrainingArguments(
    output_dir="adapters/law_adapter",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Save adapter
model.save_pretrained("adapters/law_adapter")
