import sys
import os
import gradio as gr
from sentence_transformers import SentenceTransformer, util
import torch

# Ensure the project root is in the Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.models import MultiAdapterModel

class App:
    def __init__(self):
        print("Initializing the Multi-Adapter System...")
        self.mm = MultiAdapterModel()
        self.tokenizer = self.mm.tokenizer
        
        # --- NEW: Semantic Router Setup ---
        print("Loading Sentence Transformer for routing...")
        self.router_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.mm.device)
        
        self.domains = ["law", "med", "code"]
        domain_descriptions = [
            "Legal questions about laws, courts, contracts, and civil rights",
            "Medical questions about diseases, symptoms, treatments, and health",
            "Computer programming questions about Python, JavaScript, algorithms, and data structures"
        ]
        
        # Pre-compute the embeddings for the domain descriptions
        self.domain_embeddings = self.router_model.encode(domain_descriptions, convert_to_tensor=True)
        print("Semantic router initialized successfully.")

    def route(self, prompt: str) -> str:
        """
        Selects the best adapter by calculating the semantic similarity
        between the prompt and domain descriptions.
        """
        prompt_embedding = self.router_model.encode(prompt, convert_to_tensor=True)
        cosine_scores = util.cos_sim(prompt_embedding, self.domain_embeddings)
        best_domain_index = torch.argmax(cosine_scores)
        selected_domain = self.domains[best_domain_index]
        print(f"--> Router selected '{selected_domain}' adapter with similarity score: {cosine_scores[0][best_domain_index]:.2f}")
        return selected_domain

# --- Create a single, global instance of the App ---
print("Starting the application... (This may take a moment)")
app = App()
print("Application started. Building Gradio interface...")

def generate_response(prompt: str) -> (str, str):
    adapter_name = app.route(prompt)
    router_decision = f"Adapter selected: {adapter_name.upper()}"

    if adapter_name != "default":
        model = app.mm.set_adapter(adapter_name)
    else:
        model = app.mm.unload_adapter()

    inputs = app.tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=256)
    answer = app.tokenizer.decode(out[0], skip_special_tokens=True)
    
    return answer, router_decision

# --- Build and launch the Gradio Web Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
        """
        # LoRA Multi-Adapter System (with Semantic Routing)
        Ask a question related to **Law**, **Medicine**, or **Code**.
        The system will automatically route your question to the best-specialized adapter.
        """
    )
    
    with gr.Row():
        prompt_input = gr.Textbox(lines=4, placeholder="e.g., 'Write a Python function to sort a list' or 'What is a tort?'", label="Your Question")
    
    with gr.Row():
        submit_button = gr.Button("Generate Answer", variant="primary")

    with gr.Row():
        answer_output = gr.Textbox(lines=4, label="Model's Answer")
        router_output = gr.Textbox(label="Router Decision")

    submit_button.click(
        fn=generate_response,
        inputs=prompt_input,
        outputs=[answer_output, router_output]
    )

iface.launch(share=True)