# Voice Agent Architecture (AGENTS)

This document explains how the real-time voice agent is structured, how audio flows through the system, what events are emitted, and how to extend or customize each stage.

## Overview

- Input: Microphone audio (mono, 16 kHz preferred)
- Pipeline: VAD → Transcription → LLM → Streaming TTS → Gapless playback
- Realtime: Everything streams; the UI updates as soon as each stage has data
- Sessions: Each WebSocket connection (sid) maintains its own state

```
Client Mic → Audio Chunks → [VAD] → [ASR Partial/Final] → [LLM Stream]
                                              ↘
                                    [Streaming TTS Chunks] → Client Playback
```

## Key Components

- VAD: Silero via `vad_detector.py` (factory in `create_vad_detector`)
- ASR: Parakeet TDT 0.6B MLX via `parakeet_mlx.from_pretrained`
- LLM: Qwen3 MLX (4B 4‑bit) via `mlx_lm.load` + `stream_generate`
- TTS: Kokoro via `streaming_tts.py` (`StreamingTTS`, `TextChunker`)
- App: Flask + Flask-SocketIO in `app.py`
- UI: Single-page HTML in `templates/index.html`

## Server: Pipeline Stages

1) VAD (Stage 1)
- Detects speech presence per incoming chunk and turn completion.
- Logs: `[STAGE 1 - VAD] Speech detected in chunk (sid: ...)`
- Config: See `vad_config.py` for `silence_threshold`, streaming windows, overlaps.

2) Transcription (Stage 2)
- Partial mode: Emits every few seconds during speech.
- Final mode: After turn completion; now uses a full-utterance buffer so the LLM receives the entire spoken turn.
- Logs: `Starting transcription (is_final=...)`, `Transcribed text: '...'`.

3) User Message (Stage 3)
- Emits the final transcript to the UI immediately: `user_message`.

4) LLM (Stage 4)
- Streams tokens; first token is logged.
- Emits tokens (`assistant_token`) and periodic full text syncs (`assistant_progress`).
- Completes with `assistant_complete`.

5) TTS (Stage 5)
- Text chunking: `TextChunker` breaks LLM text into sentences/phrases.
- For each chunk, `StreamingTTS` generates WAV bytes and streams them to the client (`assistant_audio`).

## Socket Events

From server → client
- `stream_started`: Recording initialized
- `partial_transcription`: Interim ASR text
- `user_message`: Final ASR text
- `assistant_start`: Begin assistant reply
- `assistant_token`: Streamed LLM token
- `assistant_progress`: Periodic full-text sync of LLM output
- `assistant_complete`: LLM response finished
- `assistant_audio`: Base64-encoded WAV chunk (24 kHz)
- `assistant_cancel`: Stop audio/text (barge-in or stop)
- `pipeline_stage`: UX hints for stage changes
- `error`: Error details

From client → server
- `start_stream`: Begin mic streaming
- `audio_chunk`: PCM16 chunk + `sampleRate`
- `stop_stream`: Stop mic streaming

## Session State (per sid)

- `audio_buffers`: Rolling buffer for partial ASR windows
- `utterance_audio_full`: Full-utterance buffer for final ASR
- `utterance_active`: Whether we are in a speaking turn
- `conversation_history`: Chat history used to build prompts
- `last_partial_text`: De-duplicate partials
- `current_generation_id`, `generation_active`: Control barge-in/cancellation
- `vad_detectors`, `vad_modes`: VAD instance & mode (Silero)
- `silence_counters`: Consecutive silent chunks (Silero)

## VAD Details

- Implementation: Silero (torch.hub) wrapped in `vad_detector.py`
- Turn completion: Triggered by N silent chunks (configurable)
- Configs: `vad_config.py` — `STREAMING_INTERVAL_SECONDS`, `MIN_BUFFER_SECONDS`, `OVERLAP_SECONDS`, `silence_threshold`

## Transcription (ASR)

- Model: `mlx-community/parakeet-tdt-0.6b-v3`
- Sample rate: Server resamples any input to 16 kHz
- Partial streaming: Uses rolling buffer with overlap for stability
- Final transcription: Uses `utterance_audio_full` to avoid truncation of earlier speech

## LLM

- Model: `Qwen/Qwen3-4B-MLX-4bit`
- Prompting: Uses tokenizer chat template when available
- Streaming: `stream_generate` yields tokens; we emit to UI immediately and in batches for reliability

## Streaming TTS

- Engine: Kokoro via `StreamingTTS`
- Voice: e.g., `af_heart` (configurable)
- Chunking: `TextChunker(mode='phrase')` for responsiveness; tune for sentence-only
- Output: WAV mono 24 kHz per chunk, base64-encoded

## Frontend Playback (Gapless)

- Maintains a single `AudioContext` and a rolling `nextStartTime`
- Decodes each WAV chunk and schedules it with `source.start(nextStartTime)`
- Advances `nextStartTime += buffer.duration` for seamless playback
- Barge-in/stop: Stops and disconnects all scheduled sources and clears queue

## Barge-In / Cancellation

- If user speech is detected while assistant audio/text is active, the server increments `current_generation_id` and emits `assistant_cancel`.
- The client stops playback immediately and ignores further TTS chunks from that generation.

## Extending

- VAD: Add a subclass in `vad_detector.py` and register in `create_vad_detector`. Tweak `vad_config.py`.
- ASR: Swap Parakeet with your model; keep the same transcribe interface.
- LLM: Replace in `app.py` (`load`, `stream_generate`) and adjust prompt template.
- TTS: Implement a class with `generate_audio_chunk(text)` yielding WAV bytes.

## Quick Start

- Python: 3.10+
- Install deps: `pip install -r requirements.txt`
- Run: `python app.py` then open the app in your browser (default port 3003)

## Troubleshooting

- TTS gaps or restart-sounding playback: Confirm the client gapless scheduler is active (`nextStartTime` logic) and that chunks arrive in order.
- Final transcript missing early words: Ensure `utterance_audio_full` is used for final ASR (already implemented).
- Torch warnings about RNN dropout/weight_norm: These are from dependencies and are benign.

---

This file is a concise reference for the agent’s moving parts. See also:
- `STREAMING_TTS_INTEGRATION.md` for step-by-step TTS streaming details
- `VAD_TUNING_GUIDE.md` for VAD parameters and trade-offs
- `CLAUDE.md` for a high-level narrative of the streaming flow
