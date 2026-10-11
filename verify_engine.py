import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Corrected imports matching your model layout
from minillm.model import LlamaModel
from minillm.cache import KVCache

def verify_token_generation():
    model_path = "models/SmolLM2-360M"
    prompt = "The quick brown fox jumps over"
    max_new_tokens = 10
    
    print("🚀 Loading Hugging Face Golden Reference (PyTorch)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # 1. Generate via Hugging Face Transformers (Greedy)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        hf_outputs = hf_model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,  # Forces greedy decoding
            temperature=None,
            top_p=None
        )
    
    # Extract only the newly generated token IDs
    hf_new_tokens = hf_outputs[0, input_ids.shape[-1]:].tolist()
    
    # 2. Generate via your custom NumPy LlamaModel Engine
    print("🚀 Loading Custom NumPy LlamaModel Engine...")
    custom_model = LlamaModel.from_pretrained(model_path, quant="none")
    
    # Initialize a cache instance for the sequence
    cache = KVCache(custom_model.cfg)
    
    # Ingest the prompt tokens (Prefill Phase)
    current_tokens = tokenizer(prompt)["input_ids"]
    custom_new_tokens = []
    
    # Autoregressive generation loop
    for _ in range(max_new_tokens):
        # forward returns logits shape (1, vocab) when last_only=True
        logits = custom_model.forward(current_tokens, cache, last_only=True)
        
        # Greedy selection: get the highest probability token ID from NumPy array
        next_token = int(np.argmax(logits[0]))
        custom_new_tokens.append(next_token)
        
        # For the next decode step, feed only the single new token
        current_tokens = [next_token]
    
    # 3. Token-for-Token Comparison
    print("\n--- 🔍 Evaluation Results ---")
    print(f"Prompt: '{prompt}'\n")
    print(f"Hugging Face Tokens: {hf_new_tokens} -> Decode: '{tokenizer.decode(hf_new_tokens)}'")
    print(f"Your Engine Tokens:  {custom_new_tokens} -> Decode: '{tokenizer.decode(custom_new_tokens)}'")
    
    if hf_new_tokens == custom_new_tokens:
        print("\nSUCCESS: Your engine reproduces the reference implementation token-for-token!")
    else:
        print("\nMISMATCH: Token drift detected. Check your KV cache indices or attention scaling coefficients.")
        for i, (hf_t, cust_t) in enumerate(zip(hf_new_tokens, custom_new_tokens)):
            if hf_t != cust_t:
                print(f"   Mismatch at token index {i}: HF={hf_t} ('{tokenizer.decode([hf_t])}'), Custom={cust_t} ('{tokenizer.decode([cust_t])}')")
                break

if __name__ == "__main__":
    verify_token_generation()
