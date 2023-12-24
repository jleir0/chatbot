# Chatbot with Gradio and microsoft/phi-2

This project is an initial functional and simple approach to building a chatbot using Gradio and the microsoft/phi-2 language model.

## Virtual Environment

This project utilizes a virtual environment to manage dependencies. Here is a quick guide to setting up the environment and running the project:

1. **Anaconda Installation:**
   - Download and install [Anaconda](https://www.anaconda.com/products/individual).

2. **Creating the Environment:**
   ```bash
   # Activate your Anaconda base environment (if not already activated)
   conda activate base
   
   # Create a new virtual environment named "myenv" with Python 3.10
   conda create -n myenv python=3.10
   
   # Activate the new virtual environment
   conda activate myenv

## Install dependencies using Conda
   - conda install -c huggingface -c conda-forge tokenizers gradio pytorch transformers sentencepiece accelerate einops

## Run the chatbot script
   - python src/chatbot3.py
