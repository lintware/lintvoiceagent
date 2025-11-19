# VAD Toggle Implementation

## Overview
Your voice agent now supports **two VAD (Voice Activity Detection) modes** that you can toggle between in real-time:

1. **Silero VAD** - Energy-based detection with silence counting (original)
2. **Smart Turn v3** - Semantic turn-completion detection (new)

## Quick Start

### 1. Start the Application
```bash
cd "/Users/admin/Documents/Projects /lintvoiceagent"
source venv/bin/activate
python app.py
```

### 2. Open in Browser
Navigate to: `http://localhost:3003`

### 3. Toggle VAD Modes
- Look for the toggle switch in the header
- **Left (OFF)**: Silero VAD
- **Right (ON)**: Smart Turn v3
- Click the "?" icon for more info

## How Each Mode Works

### Silero VAD (Default)
- **Detection Method**: Energy-based + ML enhancement
- **Turn Completion**: Counts 8 consecutive silent chunks
- **Pros**: Fast, lightweight, reliable
- **Cons**: May cut off mid-thought pauses or wait too long on trailing speech

### Smart Turn v3
- **Detection Method**: Semantic analysis of raw waveform
- **Turn Completion**: Analyzes speech completion semantically
- **Pros**:
  - Understands filler words ("um", "uh", "えーと", etc.)
  - Detects intonation patterns
  - More natural conversation flow
  - 23 language support
- **Cons**: Slightly more computational (12ms on CPU)

## Architecture

### Files Created/Modified

1. **`vad_detector.py`** (NEW)
   - Abstract `VADDetector` base class
   - `SileroVAD` implementation
   - `SmartTurnV3` implementation
   - Factory function: `create_vad_detector()`

2. **`app.py`** (MODIFIED)
   - Removed hardcoded Silero VAD
   - Added per-session VAD detector management
   - New socket event: `set_vad_mode`
   - All speech detection now uses VAD detector interface

3. **`templates/index.html`** (MODIFIED)
   - Toggle switch UI in header
   - Info tooltip
   - JavaScript to handle mode switching
   - Socket event handlers for `vad_mode_changed`

4. **`requirements.txt`** (MODIFIED)
   - Added `transformers`, `accelerate`, `safetensors`

## Code Flow

### VAD Mode Switching
```
User toggles switch → Frontend sends 'set_vad_mode' event
                    ↓
Backend validates mode → Creates new VAD detector
                    ↓
Backend emits 'vad_mode_changed' → Frontend updates UI
```

### Speech Processing
```
Audio chunk arrives → VAD detector checks speech
                    ↓
                VAD detector checks turn completion
                    ↓
If complete → Transcribe → Send to LLM → Stream response
```

## Testing the Modes

### Test Scenario 1: Mid-Sentence Pause
**Action**: Say "I want to... [pause 2 sec] ...order a pizza"
- **Silero VAD**: May cut you off after "I want to..."
- **Smart Turn v3**: Waits for semantic completion

### Test Scenario 2: Trailing Filler
**Action**: Say "I need help with ummmm..."
- **Silero VAD**: Waits for full 8 chunks of silence
- **Smart Turn v3**: Detects incomplete thought faster

### Test Scenario 3: Natural Completion
**Action**: Say "What's the weather today?"
- **Both**: Should work similarly and respond promptly

## Configuration

### Adjusting Detection Sensitivity

In `vad_detector.py`, you can adjust:

```python
# Silero VAD
class SileroVAD(VADDetector):
    def __init__(self, threshold=0.5):  # 0.0-1.0
        self.silence_threshold = 8      # Number of chunks

# Smart Turn v3
class SmartTurnV3(VADDetector):
    def __init__(self, threshold=0.5):  # 0.0-1.0 (turn completion probability)
```

### Changing Default Mode

In `app.py`:
```python
DEFAULT_VAD_MODE = "silero"  # or "smart_turn_v3"
```

## Troubleshooting

### Smart Turn v3 Fails to Load
- Check: Dependencies installed? `pip install -r requirements.txt`
- System: Falls back to Silero VAD automatically
- Check console for error messages

### Toggle Not Working During Recording
- Expected behavior: Cannot change modes while recording
- Stop recording first, then switch modes

### Model Loading is Slow
- Smart Turn v3 downloads model on first use (~360 MB)
- Subsequent loads are faster (cached)

## Performance

### Model Sizes
- **Silero VAD**: ~5 MB
- **Smart Turn v3**: ~360 MB (int8 quantized, ~8M params)

### Latency (per chunk)
- **Silero VAD**: <5ms
- **Smart Turn v3**: ~12ms on Mac (CPU)

## API Reference

### Socket Events

#### Client → Server
```javascript
// Set VAD mode
socket.emit('set_vad_mode', { mode: 'silero' | 'smart_turn_v3' });
```

#### Server → Client
```javascript
// VAD mode changed
socket.on('vad_mode_changed', (data) => {
    // data.mode: 'silero' | 'smart_turn_v3'
    // data.message: Status message
});

// Stream started with VAD info
socket.on('stream_started', (data) => {
    // data.vad_mode: Current VAD mode
});
```

## Future Enhancements

Potential improvements:
- [ ] Add visual indicator showing current VAD mode during recording
- [ ] Per-user VAD preferences (persistent)
- [ ] Hybrid mode: Use Silero for speech detection, Smart Turn for completion
- [ ] Real-time VAD confidence visualization
- [ ] A/B testing metrics for both modes

## Resources

- **Smart Turn v3 Blog**: https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/
- **GitHub Repo**: https://github.com/pipecat-ai/smart-turn
- **Hugging Face Model**: https://huggingface.co/pipecat-ai/smart-turn-v3
- **Silero VAD**: https://github.com/snakers4/silero-vad

---

**Status**: ✅ Fully implemented and tested
**Version**: 1.0
**Date**: 2025-11-19
