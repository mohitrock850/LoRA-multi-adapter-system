# src/models.py
import os
import yaml
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define the project's base directory to correctly locate config files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class MultiAdapterModel:
    def __init__(self):
        with open(os.path.join(BASE_DIR, "configs", "base.yaml"), "r") as f:
            base_cfg = yaml.safe_load(f)

        self.base_model_name = base_cfg["base_model"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        # IMPORTANT: Keep a reference to the original, clean base model
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model_name,
            device_map={"":"cpu"} if self.device == "cpu" else "auto"
        )
        
        # This will hold the model with the adapter attached
        self.model = self.base_model
        print(f"Base model '{self.base_model_name}' loaded on device: {self.device}")

    def set_adapter(self, adapter_name: str):
        with open(os.path.join(BASE_DIR, "configs", "adapters.yaml"), "r") as f:
            adp_cfg = yaml.safe_load(f)

        adapter_path = adp_cfg["adapters"][adapter_name]["path"]
        full_adapter_path = os.path.join(BASE_DIR, adapter_path)
        
        print(f"Loading adapter '{adapter_name}' from: {full_adapter_path}")

        # ALWAYS load the adapter onto the clean base_model to prevent stacking
        self.model = PeftModel.from_pretrained(self.base_model, full_adapter_path)
        print(f"Adapter '{adapter_name}' loaded and activated.")
        
        return self.model
        
    def unload_adapter(self):
        if self.model is not self.base_model:
            self.model = self.base_model
            print("--> Adapter unloaded. Reverted to base model.")
        return self.model