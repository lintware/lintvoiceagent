from mlx_lm import load, generate
import time

print("Loading Qwen3-0.6B-MLX-4bit model...")
model, tokenizer = load("Qwen/Qwen3-0.6B-MLX-4bit")

prompt = "Hello, this is a test. Please respond with a short greeting."

# Apply chat template with thinking disabled for faster response
if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Disable thinking for this test
    )

print("\nGenerating response...")
print("=" * 60)

start_time = time.time()

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,  # This will show token/s information
    max_tokens=100,
)

end_time = time.time()
elapsed_time = end_time - start_time

print("=" * 60)
print(f"\nResponse: {response}")
print(f"\nTotal time: {elapsed_time:.2f} seconds")
