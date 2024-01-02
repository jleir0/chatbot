# -*- coding: utf-8 -*-

# Agrega esta línea al principio de tu script
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
import torch
import gradio as gr

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)

# Inicializa la memoria
contexto = {'pregunta_anterior': '', 'respuesta_anterior': ''}

# Cargar el tokenizador y el modelo del clasificador
cl_tokenizer = AutoTokenizer.from_pretrained("src/classifier/model/")
cl_model = AutoModelForSequenceClassification.from_pretrained("src/classifier/model/", config="src/classifier/model/config.json")

# Cargar el clasificador
classifier = pipeline("text-classification", model=cl_model, tokenizer=cl_tokenizer)

print(contexto['pregunta_anterior'])
print(contexto['respuesta_anterior'])

# Función para determinar la clase y generar texto en consecuencia
def generate_text(prompt):
    print("Received prompt:", prompt)
    
    # Clasificar la entrada en una de las tres clases: "atencion_cliente", "comercial", "bromas"
    classification = classifier(prompt)[0]
    class_prediction = classification['label']
    score_prediction = classification['score']
    print(class_prediction)
    print(score_prediction)
    
    full_prompt = ""
    # Elegir la plantilla de prompt según la clase predicha
    if class_prediction == "LABEL_1" and score_prediction >= 0.8:
        full_prompt = "customer service: " + contexto['pregunta_anterior'] + contexto['respuesta_anterior'] + prompt
    elif class_prediction == "LABEL_0" and score_prediction >= 0.8:
        full_prompt = "business question: " + contexto['pregunta_anterior'] + contexto['respuesta_anterior'] + prompt
    else:
        full_prompt = "joke"
    
    # Concatenar la plantilla de prompt con la entrada del usuario y generar texto
    print(full_prompt)
    with torch.no_grad():
        token_ids = tokenizer.encode(full_prompt, add_special_tokens=False, return_tensors="pt")
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3
        )
        
    output = tokenizer.decode(output_ids[0][token_ids.size(1):])
    contexto['pregunta_anterior'] = prompt
    contexto['respuesta_anterior'] = output
    
    return output

# Crear una interfaz de Gradio
iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(placeholder="Type your prompt here and press Enter", lines=5, type="text", live_update=True),
    outputs="text",
    live=True,
)

# Iniciar la interfaz de Gradio
iface.launch()