# VAD Tuning Guide

Both Silero VAD and Smart Turn v3 are now fully flexible and configurable!

## Quick Start

Edit `vad_config.py` to adjust VAD behavior:

### Use Presets (Easiest Way)

```python
# In vad_config.py, line 134
ACTIVE_PRESET = 'very_responsive'  # Options: 'default', 'very_responsive', 'patient', 'balanced'
```

### Or Manually Tune Parameters

Edit the config dictionaries in `vad_config.py`:

## Silero VAD Parameters

```python
SILERO_CONFIG = {
    'threshold': 0.5,              # 0.0-1.0 (lower = more sensitive)
    'silence_threshold': 8,        # Chunks of silence (8 = ~2 seconds)
    'min_speech_duration_ms': 250, # Min speech to detect
    'min_silence_duration_ms': 100,# Min silence to split
    'energy_threshold': 0.001,     # RMS energy threshold
}
```

**To make it respond faster:**
- Lower `silence_threshold` (try 4-6 for ~1-1.5 seconds)
- Lower `threshold` (try 0.3-0.4)

**To make it more patient:**
- Raise `silence_threshold` (try 10-12 for ~2.5-3 seconds)
- Raise `threshold` (try 0.6-0.7)

## Smart Turn v3 Parameters

```python
SMART_TURN_CONFIG = {
    'threshold': 0.5,        # 0.0-1.0 (lower = interrupts sooner)
    'energy_threshold': 0.001,
    'responsiveness': 1.0,   # 0.5-2.0 (higher = more aggressive)
}
```

**To make it respond faster:**
- Lower `threshold` (try 0.3-0.4)
- Raise `responsiveness` (try 1.5-2.0)

**To make it more patient:**
- Raise `threshold` (try 0.6-0.7)
- Lower `responsiveness` (try 0.7-0.8)

## Streaming & Chunking

```python
STREAMING_INTERVAL_SECONDS = 2.0  # Show partial transcription every N seconds
MIN_BUFFER_SECONDS = 1.5          # Min audio before transcription
OVERLAP_SECONDS = 1.0             # Audio overlap between chunks
```

**For faster updates:**
- Lower `STREAMING_INTERVAL_SECONDS` to 1.0
- Lower `MIN_BUFFER_SECONDS` to 1.0

## Preset Comparison

| Preset | Silence Threshold | Response Time | Best For |
|--------|------------------|---------------|----------|
| `very_responsive` | 4 chunks (~1s) | Very fast | Quick Q&A, commands |
| `balanced` | 8 chunks (~2s) | Normal | General conversation |
| `patient` | 12 chunks (~3s) | Slow | Long thoughts, storytelling |

## Examples

### Quick Commands Mode
```python
ACTIVE_PRESET = 'very_responsive'
```
- Great for: "Turn on the lights", "What's the weather?"
- Responds after ~1 second of silence

### Natural Conversation
```python
ACTIVE_PRESET = 'balanced'
```
- Great for: Normal back-and-forth chat
- Responds after ~2 seconds of silence

### Storytelling / Long Form
```python
ACTIVE_PRESET = 'patient'
```
- Great for: Long explanations, stories, dictation
- Responds after ~3 seconds of silence
- Won't interrupt during natural pauses

## After Changes

Just restart the app:
```bash
python app.py
```

Your VAD settings will take effect immediately!
