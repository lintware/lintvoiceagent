# Streaming TTS Integration - Complete Guide

## Overview

The voice agent now has **real-time streaming TTS** integrated with comprehensive console logging at each pipeline stage. The UI responds immediately at each step, not waiting for audio generation.

## Pipeline Flow

The complete pipeline consists of 5 distinct stages:

### Stage 1: Voice Activity Detection (VAD)
- **Trigger**: Audio chunk contains speech
- **Console Log**: `[STAGE 1 - VAD] Speech detected in chunk`
- **Frontend Event**: `pipeline_stage` with `stage: 'vad'`
- **UI Update**: Status shows speech detected

### Stage 2: Transcription
- **Trigger**: VAD detects turn completion (silence detected)
- **Console Logs**:
  - `[STAGE 2 - TRANSCRIPTION] Starting transcription (is_final=true)`
  - `[STAGE 2 - TRANSCRIPTION] Transcribed text: '...'`
- **Frontend Events**:
  - `pipeline_stage` with `stage: 'transcription_start'`
  - `pipeline_stage` with `stage: 'transcription_complete'`
  - `partial_transcription` (during speaking, shown in lighter text)
  - `user_message` (final transcription)
- **UI Update**: User message bubble appears **IMMEDIATELY** after transcription

### Stage 3: User Message Display
- **Console Log**: `[STAGE 3 - USER MESSAGE] Displaying user message in UI`
- **Frontend Event**: `user_message`
- **UI Update**: Final user message replaces partial transcription
- **Important**: This happens **BEFORE** LLM processing starts

### Stage 4: LLM Processing
- **Console Logs**:
  - `[STAGE 4 - LLM] Starting LLM processing`
  - `[STAGE 4 - LLM] First token generated: '...'`
  - `[STAGE 4 - LLM] LLM complete: '...'`
- **Frontend Events**:
  - `pipeline_stage` with `stage: 'llm_start'`
  - `assistant_start` (creates empty assistant bubble)
  - `assistant_token` (each token as it's generated)
  - `assistant_complete`
  - `pipeline_stage` with `stage: 'llm_complete'`
- **UI Update**: Assistant message bubble streams text **in real-time**, token by token

### Stage 5: TTS Generation (Parallel with Stage 4)
- **Trigger**: Text chunker detects complete phrase/sentence
- **Console Logs**:
  - `[STAGE 5 - TTS] Generating audio for chunk: '...'`
  - `[STAGE 5 - TTS] Sending audio chunk (N bytes)`
  - `[AUDIO PLAYBACK] Received audio chunk, queuing for playback`
  - `[AUDIO PLAYBACK] Playing audio chunk (N remaining in queue)`
- **Frontend Events**:
  - `pipeline_stage` with `stage: 'tts_start'`
  - `assistant_audio` (base64 encoded WAV chunks)
- **UI Update**: Audio queued and played automatically

## Key Features

### 1. Independent Streaming
- **Text Display**: LLM tokens appear immediately in the UI
- **Audio Generation**: Happens in parallel, doesn't block text display
- **Audio Playback**: Queued and played smoothly in sequence

### 2. Responsive UI
- Transcription shows **immediately** after VAD detects turn completion
- LLM text streams **before** any audio is ready
- User sees progress at every stage

### 3. Console Logging
All stages log to both:
- **Backend** (server console): Python print statements
- **Frontend** (browser console): JavaScript console.log statements

### 4. Partial Transcription
- While speaking: Shows lighter, updating transcription
- After silence: Replaced with final message before LLM starts

## File Changes

### Backend Changes

#### `app.py`
1. Added `streaming_tts` imports
2. Lazy-loaded TTS engine (loads on first use)
3. Added console logging at all stages
4. Modified `get_llm_response_streaming()` to:
   - Use `TextChunker` for phrase-based chunking
   - Generate TTS for complete chunks
   - Emit audio via WebSocket
5. Added `pipeline_stage` events throughout

#### `streaming_tts.py` (New)
- `StreamingTTS`: Main TTS class using Kokoro
- `TextChunker`: Intelligently chunks text at phrase/sentence boundaries
- `audio_to_base64()`: Converts WAV to base64 for transmission

#### `requirements.txt`
- Added `kokoro>=0.7.13`

### Frontend Changes

#### `templates/index.html`
1. Added audio playback queue system
2. Added `pipeline_stage` event handler with console logging
3. Added detailed logging for all events:
   - Partial transcription
   - User messages
   - Assistant start/tokens/complete
   - Audio reception and playback
4. Implemented audio queue with sequential playback
5. Added partial transcription display (lighter opacity)

## Testing the Pipeline

### Open Browser Console
1. Navigate to `http://localhost:3003`
2. Open Developer Tools (F12)
3. Go to Console tab

### Expected Console Output

```
[VAD] Speech detected
[TRANSCRIPTION_START] Transcribing audio...
[TRANSCRIPTION_COMPLETE] Transcribed: Hello, how are you?
[USER MESSAGE] Displaying final transcription: Hello, how are you?
[LLM_START] Processing with LLM...
[ASSISTANT START] Creating assistant message bubble
[ASSISTANT TOKEN] Received token: I
[ASSISTANT TOKEN] Received token: 'm
...
[TTS_START] Generating speech for: I'm doing great, thanks for asking!
[ASSISTANT AUDIO] Received audio chunk, queuing for playback
[AUDIO PLAYBACK] Audio context initialized
[AUDIO PLAYBACK] Playing audio chunk (0 remaining in queue)
[AUDIO PLAYBACK] Audio decoded: 1.23s
[AUDIO PLAYBACK] Chunk finished, playing next
[AUDIO PLAYBACK] Queue empty, playback complete
[ASSISTANT COMPLETE] Full response received
```

## Configuration

### TTS Voice
Change in `app.py`:
```python
tts_engine = StreamingTTS(voice='af_heart', speed=1.0, use_gpu=False)
```

Available voices:
- `af_heart` - American female, warm (default)
- `af_bella` - American female, expressive
- `af_nova` - American female, clear
- `am_michael` - American male
- `bm_george` - British male
- And many more...

### Chunking Mode
Change in `app.py` `get_llm_response_streaming()`:
```python
chunker = TextChunker(mode='phrase')  # 'phrase' or 'sentence'
```

- `'phrase'`: Breaks on commas, colons (faster streaming)
- `'sentence'`: Breaks only on periods, exclamation marks (slower but more natural)

## Troubleshooting

### No Audio Playback
- Check browser console for decoding errors
- Verify Kokoro TTS installed: `pip install kokoro>=0.7.13`
- Check audio permissions in browser

### Text Shows But No Audio
- Check backend console for TTS errors
- Verify TTS engine loaded successfully
- Check for `[STAGE 5 - TTS]` logs in backend

### Laggy UI
- Text should appear immediately (Stage 4)
- Audio generation is parallel (Stage 5)
- If text waits for audio, check `socketio.emit('assistant_token')` is called before TTS

## Performance

- **TTS Latency**: ~100-500ms per phrase chunk
- **LLM Streaming**: Tokens appear immediately
- **Audio Playback**: Queued and plays sequentially
- **Memory**: TTS model ~500MB, shared across sessions

## Next Steps

1. Test with actual voice input
2. Monitor console logs for bottlenecks
3. Adjust chunking mode if needed
4. Consider adding visual indicators for each stage
5. Add error handling for TTS failures
