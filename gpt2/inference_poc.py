import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_device():
    """
    Determines the device to use for computation: CUDA, MPS (Apple Silicon), or CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Determine the device
device = get_device()

# Paths
MODEL_PATH = "openai-community/gpt2"

# Load the tokenizer and model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, use_safetensors=True).to(device)

# Input text and generation parameters
init_input_text = "Hello, I'm a smallish language model,"
MAX_LENGTH = 64

# Tokenize the input text
tokenized_input = tokenizer.encode(init_input_text, return_tensors='pt').to(device)

# Generate text
print("Generating text...")
generated_text_encoded = model.generate(tokenized_input, max_length=MAX_LENGTH)
generated_text = tokenizer.decode(generated_text_encoded[0], skip_special_tokens=True)

# Output the generated text
print("Generated Text:")
print(generated_text)