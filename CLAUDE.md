# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Smart Turn v3 support (optional)
pip install onnxruntime huggingface-hub librosa

# Run the application
python app.py
```

The app runs on `http://localhost:3003`

## Architecture Overview

This is a real-time voice conversation agent with streaming speech-to-text, LLM processing, and configurable Voice Activity Detection (VAD).

### Core Flow
```
Audio Chunks (WebSocket) → VAD Detection → Transcription (Parakeet) → LLM (Qwen) → Streaming Response
```

### Key Components

**1. Voice Activity Detection (VAD) - Dual System**
- Two VAD implementations via abstract interface in `vad_detector.py`:
  - `SileroVAD`: Silence-based detection (default)
  - `SmartTurnV3`: Semantic turn-completion detection using ONNX model
- **Per-session VAD instances** - each WebSocket connection has its own detector
- Configuration via `vad_config.py` with presets (`very_responsive`, `balanced`, `patient`)
- Factory pattern: `create_vad_detector(vad_type, **kwargs)`

**2. Session Management (`app.py`)**
- All state is per-session using `defaultdict` keyed by `request.sid`:
  - `audio_buffers[sid]` - accumulated audio chunks
  - `vad_detectors[sid]` - VAD instance
  - `conversation_history[sid]` - full chat history for context
  - `vad_modes[sid]` - current VAD mode
  - `silence_counters[sid]`, `last_partial_text[sid]`, etc.
- Thread-safe access via `buffer_locks[sid]`

**3. Dual-Mode Transcription**
- **Streaming Mode**: Transcribes every 2 seconds while speaking (partial results)
  - Keeps 1-second audio overlap to prevent word-cutting
  - Emits `partial_transcription` events
- **Final Mode**: Triggered when VAD detects turn completion
  - Full transcription sent to LLM
  - Emits `user_message` → `assistant_start` → `assistant_token` (streaming) → `assistant_complete`

**4. Models Loaded**
- **Parakeet TDT 0.6B v3** (`mlx-community/parakeet-tdt-0.6b-v3`) - Speech-to-text
- **Qwen 4B 4-bit** (`Qwen/Qwen3-4B-MLX-4bit`) - LLM with streaming generation
- **Silero VAD** - Loaded via torch.hub
- **Smart Turn v3 ONNX** - Downloaded from HuggingFace on-demand

### Critical Implementation Details

**Audio Processing Pipeline** (`handle_audio_chunk`):
1. Receives Int16 audio at variable sample rate
2. Resamples to 16kHz using linear interpolation
3. Appends to session buffer
4. VAD checks speech presence and turn completion
5. Minimum 1.5s buffer required before transcription
6. Writes to temporary WAV file for Parakeet
7. Cleans up temp files immediately

**VAD Turn Completion Logic**:
- `SileroVAD`: Counts consecutive silent chunks (configurable threshold, default 8 = ~2 seconds)
- `SmartTurnV3`:
  - Requires exactly 8 seconds of audio (128,000 samples)
  - Pads/trims to exact length
  - Converts to 80 mel bins × 800 time steps (with `center=False`)
  - ONNX inference returns turn completion probability
  - Applies `responsiveness` multiplier to threshold

**Configuration System**:
- `vad_config.py` centralizes all tuning parameters
- Presets available: `very_responsive` (~1s silence), `balanced` (~2s), `patient` (~3s)
- Change `ACTIVE_PRESET` and restart to apply
- Parameters: `silence_threshold`, `min_speech_duration_ms`, `responsiveness`, etc.

## Important Patterns

**Adding New VAD Detector**:
1. Subclass `VADDetector` in `vad_detector.py`
2. Implement `has_speech()` and `is_turn_complete()`
3. Add to `create_vad_detector()` factory
4. Add config to `vad_config.py`

**Modifying Response Speed**:
- Edit `vad_config.py` SILERO_CONFIG `silence_threshold` (lower = faster)
- Or change `ACTIVE_PRESET` to `'very_responsive'`
- Restart app to apply

**WebSocket Events**:
- Client → Server: `start_stream`, `audio_chunk`, `stop_stream`, `set_vad_mode`
- Server → Client: `stream_started`, `partial_transcription`, `user_message`, `assistant_start`, `assistant_token`, `assistant_complete`, `vad_mode_changed`

## Structured Output Usage

When working with AI outputs (per user's CLAUDE.md preferences), use structured output schemas instead of prompt instructions. The LLM integration uses `stream_generate()` from `mlx_lm` with chat templates.

## Session Cleanup

On disconnect (`handle_disconnect`), all session state is deleted from defaultdicts and locks to prevent memory leaks.
