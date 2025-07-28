import os

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache"
os.environ["HF_HOME"] = "/app/.cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("📥 Pre-downloading TinyLlama model for Docker image...")
print("This will cache the model in the Docker image to avoid runtime downloads")

try:
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/app/.cache"
    )
    
    # Download model
    print("Downloading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        cache_dir="/app/.cache"
    )
    
    print("✅ Model successfully downloaded and cached!")
    
    # Test the model
    print("Testing model with a simple prompt...")
    test_prompt = "Hello, I am a"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test successful! Output: {result}")
    
    # Show cache size
    cache_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk("/app/.cache")
        for filename in filenames
    ) / (1024 * 1024)  # Convert to MB
    
    print(f"📊 Total cache size: {cache_size:.1f} MB")

except Exception as e:
    print(f"❌ Error downloading model: {e}")
    raise e
