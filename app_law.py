import os
import gradio as gr
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER = "adapters/law_adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, token=HF_TOKEN, load_in_4bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

def chat_fn(history, message):
    prompt = f"User: {message}\nAI:"
    out = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    reply = out[0]["generated_text"]
    history.append((message, reply))
    return history, ""

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask about Law")
    clear = gr.Button("Clear")

    msg.submit(chat_fn, [chatbot, msg], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg])

demo.launch()
