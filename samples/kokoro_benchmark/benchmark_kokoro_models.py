#!/usr/bin/env python3
"""Benchmark script to compare generation speed of 4 Kokoro TTS models"""

import time
import os
from mlx_audio.tts.generate import generate_audio
import json

# Models to benchmark
MODELS = [
    "mlx-community/Kokoro-82M-bf16",
    "mlx-community/Kokoro-82M-8bit",
    "mlx-community/Kokoro-82M-6bit",
    "mlx-community/Kokoro-82M-4bit"
]

# Test sentence
TEST_TEXT = "Hello! This is a test of the Kokoro text to speech model running on Apple Silicon."

def benchmark_model(model_path, test_text, output_dir="benchmark_outputs"):
    """Benchmark a single model and return metrics"""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract model name for file naming
    model_name = model_path.split("/")[-1]

    print(f"\n{'='*80}")
    print(f"Benchmarking: {model_path}")
    print(f"{'='*80}")

    # Start timing
    start_time = time.time()

    try:
        # Generate audio
        generate_audio(
            text=test_text,
            model_path=model_path,
            voice="af_heart",
            speed=1.0,
            lang_code="a",  # American English
            file_prefix=f"{output_dir}/{model_name}",
            audio_format="wav",
            sample_rate=24000,
            join_audio=True,
            verbose=True
        )

        # End timing
        end_time = time.time()
        generation_time = end_time - start_time

        # Get file size
        output_file = f"{output_dir}/{model_name}.wav"
        file_size = os.path.getsize(output_file) / 1024  # KB

        metrics = {
            "model": model_path,
            "model_name": model_name,
            "success": True,
            "generation_time_seconds": round(generation_time, 3),
            "file_size_kb": round(file_size, 2),
            "output_file": output_file
        }

        print(f"\n✅ Success!")
        print(f"   Generation time: {generation_time:.3f} seconds")
        print(f"   File size: {file_size:.2f} KB")

    except Exception as e:
        print(f"\n❌ Failed: {str(e)}")
        metrics = {
            "model": model_path,
            "model_name": model_name,
            "success": False,
            "error": str(e)
        }

    return metrics

def main():
    print("="*80)
    print("KOKORO TTS MODEL BENCHMARK")
    print("="*80)
    print(f"\nTest sentence: \"{TEST_TEXT}\"")
    print(f"Number of models: {len(MODELS)}")
    print(f"\nModels to benchmark:")
    for i, model in enumerate(MODELS, 1):
        print(f"  {i}. {model}")

    # Run benchmarks
    results = []
    for model in MODELS:
        result = benchmark_model(model, TEST_TEXT)
        results.append(result)
        time.sleep(1)  # Small delay between models

    # Generate summary report
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)

    successful_results = [r for r in results if r.get("success", False)]

    if successful_results:
        print("\n┌─────────────────────────────┬──────────────────┬────────────────┐")
        print("│ Model                       │ Generation Time  │ File Size      │")
        print("├─────────────────────────────┼──────────────────┼────────────────┤")

        for result in successful_results:
            model_name = result["model_name"]
            gen_time = result["generation_time_seconds"]
            file_size = result["file_size_kb"]
            print(f"│ {model_name:27} │ {gen_time:14.3f}s │ {file_size:12.2f} KB │")

        print("└─────────────────────────────┴──────────────────┴────────────────┘")

        # Find fastest and slowest
        fastest = min(successful_results, key=lambda x: x["generation_time_seconds"])
        slowest = max(successful_results, key=lambda x: x["generation_time_seconds"])

        print(f"\n🏆 Fastest: {fastest['model_name']} ({fastest['generation_time_seconds']:.3f}s)")
        print(f"🐌 Slowest: {slowest['model_name']} ({slowest['generation_time_seconds']:.3f}s)")

        if slowest["generation_time_seconds"] > 0:
            speedup = slowest["generation_time_seconds"] / fastest["generation_time_seconds"]
            print(f"⚡ Speed difference: {speedup:.2f}x")

        # Find smallest and largest files
        smallest = min(successful_results, key=lambda x: x["file_size_kb"])
        largest = max(successful_results, key=lambda x: x["file_size_kb"])

        print(f"\n💾 Smallest file: {smallest['model_name']} ({smallest['file_size_kb']:.2f} KB)")
        print(f"💾 Largest file: {largest['model_name']} ({largest['file_size_kb']:.2f} KB)")

    # Show failed models if any
    failed_results = [r for r in results if not r.get("success", False)]
    if failed_results:
        print("\n❌ Failed models:")
        for result in failed_results:
            print(f"   - {result['model_name']}: {result.get('error', 'Unknown error')}")

    # Save results to JSON
    results_file = "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Detailed results saved to: {results_file}")
    print(f"🎵 Audio files saved in: benchmark_outputs/")

if __name__ == "__main__":
    main()
