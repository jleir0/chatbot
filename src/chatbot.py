# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)

# Función para la generación de texto
def generate_text(prompt):
    print("Received prompt:", prompt)
    with torch.no_grad():
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3
        )
    output = tokenizer.decode(output_ids[0][token_ids.size(1):])
    return output

# Crear una interfaz de Gradio
iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(placeholder="Type your prompt here and press Enter", lines=5, type="text", live_update=True),
    outputs="text",
    live=True,
    title="Text Generation",
    description="Give me a prompt, and I'll generate text for you!"
)

# Iniciar la interfaz de Gradio
iface.launch()
