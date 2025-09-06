# LoRA Multi-Adapter System 🧠

This project is a complete, end-to-end system for fine-tuning, evaluating, and serving a single base language model with multiple specialized LoRA adapters. The system can dynamically route user prompts to the best-suited adapter for tasks in different domains: **Law**, **Medicine**, and **Code**.

This repository showcases the full lifecycle of a modern NLP project, from data preparation and parameter-efficient fine-tuning (PEFT) to building an interactive web application with a semantic router.



---
## ## Features ✨

* **Multi-Domain Expertise:** A single base model (`google/flan-t5-base`) handles multiple specialized tasks, saving significant resources.
* **Efficient LoRA Fine-Tuning:** Uses Low-Rank Adaptation (LoRA) to train lightweight adapters, making custom model training fast and accessible.
* **Semantic Routing:** An intelligent router uses a `sentence-transformer` model to understand the *meaning* of a prompt and select the most appropriate adapter based on semantic similarity.
* **Interactive Web UI:** Built with Gradio to provide a user-friendly interface for interacting with the complete system.
* **Full ML Pipeline:** Includes scripts for training, evaluation (calculating BLEU and Exact Match scores), and live inference.
* **Configurable:** Key settings like model names, training parameters, and adapter paths are managed via YAML configuration files.

---
## ## Current Status & Evaluation Results

This project serves as a proof-of-concept demonstrating the full architecture. The adapters have been intentionally trained on a **very small dataset** to ensure the pipeline is functional without requiring extensive computational resources.

* **Training Data:** 10 samples per domain.
* **Evaluation Data:** 5 samples per domain.

Due to the limited training data, the model's performance is currently low, which is expected. This is reflected in the evaluation scores for the 'law' adapter:

* **Exact Match:** `0.00`
* **BLEU Score:** `0.30`

These baseline scores confirm that the evaluation pipeline is working correctly. To achieve high-quality, factual answers, the next step is to re-train the adapters on a much larger dataset. Additional data (500+ rows per domain) has been generated and is available in the repository.

---
## ## Setup and Installation ⚙️

**1. Clone the Repository:**
```bash
git clone <your-repository-url>
cd <repository-name>
```

**2. Create a Virtual Environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables:**
Create a file named `.env` in the project's root directory and add your Hugging Face access token:
```
HF_TOKEN="hf_YourHuggingFaceTokenHere"
```

---
## ## Usage Instructions 🚀

**1. Train the Adapters:**
Before running the main application, you must train the adapters. The command below uses the small default dataset.

```bash
# Train the law adapter
python src/train_adapters.py --adapter law
```
*(Repeat for `med` and `code` adapters.)*

**2. Evaluate an Adapter (Optional):**
To measure the performance of a trained adapter on its test set, run the evaluation script.

```bash
python -m src.evaluate --adapter law --eval
```

**3. Run the Gradio Web Application:**
Once your adapters are trained, start the main application with the semantic router.

```bash
python -m src.app
```
Open the local URL (e.g., `http://127.0.0.1:7860`) provided in your terminal to access the web interface.

---
## ## Future Improvements

* **Full-Scale Training:** Re-train the adapters using the provided larger datasets (`law_data.jsonl`, etc.) and for more epochs (e.g., 3-5) to dramatically improve model accuracy.
* **Experiment with Base Models:** Swap `flan-t5-base` for a larger model like `flan-t5-large` or a model from the Llama/Mistral families for a more capable foundation.
* **Hyperparameter Tuning:** Systematically experiment with learning rates and LoRA parameters (`r`, `lora_alpha`) to find the optimal settings.
* **Containerize:** Package the application with Docker for easier deployment and reproducibility.