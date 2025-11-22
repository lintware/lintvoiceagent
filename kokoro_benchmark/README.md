# Kokoro TTS Model Benchmark

Comparison of 4 quantized versions of the Kokoro-82M model on Apple Silicon.

## Test Setup
- **Hardware**: Apple Silicon (M-series)
- **Framework**: MLX Audio
- **Test sentence**: "Hello! This is a test of the Kokoro text to speech model running on Apple Silicon."
- **Voice**: af_heart (American female)
- **Speed**: 1.0x

## Results

| Model | Generation Time | Speed Rank |
|-------|----------------|------------|
| **Kokoro-82M-bf16** | **2.373s** | 🥇 Fastest |
| Kokoro-82M-4bit | 6.794s | 🥈 2nd |
| Kokoro-82M-6bit | 9.599s | 🥉 3rd |
| Kokoro-82M-8bit | 11.082s | 4th |

## Key Findings

- **bf16 is 4.67x faster** than 8bit (slowest)
- Counter-intuitive: Lower quantization did NOT improve speed
- bf16 achieves **~7.7x real-time factor** (generates audio much faster than playback)
- All models produce similar audio file sizes (~273 KB)

## Conclusion

**Use `mlx-community/Kokoro-82M-bf16` for production.** It delivers the best performance on Apple Silicon, likely due to hardware optimization for bfloat16 operations in the MLX framework.

## Files

- `benchmark_kokoro_models.py` - Benchmark script
- `benchmark_results.json` - Detailed metrics
- `benchmark_outputs/` - Generated audio samples from each model
