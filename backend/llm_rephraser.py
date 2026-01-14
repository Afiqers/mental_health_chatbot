from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Configuration
MODEL_NAME = "google/flan-t5-base"
MAX_LENGTH = 128
USE_LLM = True  # Toggle to easily disable if needed

print(f"Loading LLM Rephraser ({MODEL_NAME})...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("LLM Rephraser loaded successfully.")
except Exception as e:
    print(f"Error loading LLM Rephraser: {e}")
    model = None
    tokenizer = None

def rephrase_response(text: str, emotion: str) -> str:
    """
    Rephrases the input text to be natural and empathetic using Flan-T5.
    """
    if not model or not tokenizer or not USE_LLM:
        return text

    # Construct prompt
    prompt = f"Rephrase this to be kind and empathetic for someone feeling {emotion}: {text}"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        # Human-like generation parameters
        outputs = model.generate(
            **inputs, 
            max_length=MAX_LENGTH, 
            num_beams=5, 
            temperature=0.7, 
            do_sample=True,
            early_stopping=True
        )
        
        rephrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return rephrased_text
    except Exception as e:
        print(f"LLM Rephrase Error: {e}")
        return text  # Fallback to original
