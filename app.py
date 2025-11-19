#!/usr/bin/env python3
import os
import wave
import tempfile
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from parakeet_mlx import from_pretrained
from mlx_lm import load, stream_generate
from collections import defaultdict
import threading
import torch
from vad_detector import create_vad_detector
from vad_config import get_vad_config, STREAMING_INTERVAL_SECONDS, MIN_BUFFER_SECONDS, OVERLAP_SECONDS
from streaming_tts import StreamingTTS, TextChunker, audio_to_base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=50000000)

# Load transcription model
transcription_model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

# Load LLM model
print("Loading Qwen/Qwen3-0.6B-MLX-4bit model...")
llm_model, llm_tokenizer = load("Qwen/Qwen3-4B-MLX-4bit")
print("Model loaded successfully!")

# TTS will be loaded lazily on first use
tts_engine = None

# VAD models will be loaded per-session based on user preference
# Default and only VAD mode
DEFAULT_VAD_MODE = "silero"

# Session-based audio buffers for streaming
audio_buffers = defaultdict(list)
buffer_locks = defaultdict(threading.Lock)
accumulated_text = defaultdict(str)  # Track accumulated transcription per session
conversation_history = defaultdict(list)  # Track conversation history for each session
last_partial_text = defaultdict(str)  # Track last partial transcription to avoid duplicates
vad_detectors = {}  # Store VAD detector per session
vad_modes = defaultdict(lambda: DEFAULT_VAD_MODE)  # Track VAD mode per session
silence_counters = defaultdict(int)  # Track consecutive silent chunks (for Silero only)
# New: maintain a full-buffer per utterance to ensure final transcription sees all speech
utterance_audio_full = defaultdict(list)
utterance_active = defaultdict(bool)
# Track assistant generation per session for cancellation/barge-in
current_generation_id = defaultdict(int)
generation_active = defaultdict(bool)

# Load configuration from vad_config.py
STREAMING_CHUNK_SIZE = int(STREAMING_INTERVAL_SECONDS * 16000)  # Convert seconds to samples
MIN_BUFFER_SIZE = int(MIN_BUFFER_SECONDS * 16000)
OVERLAP_SAMPLES = int(OVERLAP_SECONDS * 16000)

def _resample_linear(x, sr_in, sr_out):
    """Lightweight linear resampler"""
    if sr_in == sr_out or x.size == 0:
        return x
    duration = x.shape[0] / float(sr_in)
    n_out = max(1, int(round(duration * sr_out)))
    t_in = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
    t_out = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)

def get_vad_detector(sid):
    """Get or create VAD detector for session"""
    if sid not in vad_detectors:
        vad_mode = vad_modes[sid]
        # Get configuration for this VAD type
        config = get_vad_config(vad_mode)
        vad_detectors[sid] = create_vad_detector(vad_mode, **config)
    return vad_detectors[sid]

def get_llm_response_streaming(conversation_messages, sid, gen_id=None):
    """Generate LLM response from conversation history with streaming TTS"""
    global tts_engine

    # Lazy load TTS engine on first use
    if tts_engine is None:
        print("Loading Kokoro TTS model...")
        tts_engine = StreamingTTS(voice='af_heart', speed=1.0, use_gpu=False)
        print("TTS model loaded!")

    try:
        # Preload system prompt for Mint persona
        system_prompt = (
            "You are Mint, a lightweight conversational agent here to help with anything. "
            "Please respond in a natural, conversational voice style. "
            "Avoid using Markdown formatting like bolding, headers, or lists, as your response will be spoken aloud. "
            "Keep responses concise and friendly."
        )
        
        # Prepend system prompt to conversation history for generation
        messages_with_system = [{"role": "system", "content": system_prompt}] + conversation_messages

        # Apply chat template with conversation history
        if llm_tokenizer.chat_template is not None:
            prompt = llm_tokenizer.apply_chat_template(
                messages_with_system,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking for faster response
            )
        else:
            # Fallback if no chat template
            prompt = f"System: {system_prompt}\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_messages])

        # Create text chunker for TTS
        chunker = TextChunker(mode='phrase')  # Use 'phrase' for faster streaming

        # Stream generate response
        full_response = ""
        token_count = 0
        for response in stream_generate(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=512,
        ):
            # Check for cancellation (barge-in)
            if gen_id is not None and current_generation_id.get(sid, 0) != gen_id:
                print(f"[CANCEL] Stopping token stream for sid={sid}, gen_id={gen_id}")
                break
            # Extract the text from the response object
            token = response.text if hasattr(response, 'text') else str(response)
            full_response += token
            token_count += 1

            # Log first token
            if token_count == 1:
                print(f"[STAGE 4 - LLM] First token generated: '{token}'")

            # Emit text token to display in UI IMMEDIATELY
            socketio.emit('assistant_token', {'token': token}, room=sid)
            # Yield to allow transports to flush
            socketio.sleep(0)

            # Periodic full-text sync to fix any dropped token events
            if token_count % 12 == 0:
                socketio.emit('assistant_progress', {'text': full_response}, room=sid)
                socketio.sleep(0)

            # Add token to chunker and generate audio for complete chunks
            for text_chunk in chunker.add_token(token):
                # STAGE 5: TTS Generation
                print(f"[STAGE 5 - TTS] Generating audio for chunk: '{text_chunk[:30]}...'")
                socketio.emit('pipeline_stage', {
                    'stage': 'tts_start',
                    'message': f'Generating speech for: {text_chunk[:30]}...'
                }, room=sid)

                # Generate audio for this chunk
                for audio_bytes in tts_engine.generate_audio_chunk(text_chunk):
                    # Check for cancellation (barge-in)
                    if gen_id is not None and current_generation_id.get(sid, 0) != gen_id:
                        print(f"[CANCEL] Stopping TTS stream for sid={sid}, gen_id={gen_id}")
                        break
                    # Convert to base64 and emit to frontend
                    audio_b64 = audio_to_base64(audio_bytes)
                    print(f"[STAGE 5 - TTS] Sending audio chunk ({len(audio_bytes)} bytes)")
                    socketio.emit('assistant_audio', {
                        'audio': audio_b64,
                        'sample_rate': 24000
                    }, room=sid)
                    # Yield so client can start playback immediately
                    socketio.sleep(0)

        # Process any remaining text in the buffer
        remaining = chunker.flush()
        if remaining:
            for audio_bytes in tts_engine.generate_audio_chunk(remaining):
                if gen_id is not None and current_generation_id.get(sid, 0) != gen_id:
                    print(f"[CANCEL] Stopping remaining TTS for sid={sid}, gen_id={gen_id}")
                    break
                audio_b64 = audio_to_base64(audio_bytes)
                socketio.emit('assistant_audio', {
                    'audio': audio_b64,
                    'sample_rate': 24000
                }, room=sid)
                socketio.sleep(0)

        # Final progress sync
        socketio.emit('assistant_progress', {'text': full_response}, room=sid)
        socketio.sleep(0)

        return full_response.strip()
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


def _run_llm_and_tts_for_sid(sid, gen_id):
    """Background task: stream LLM tokens and TTS audio for this sid."""
    try:
        # Snapshot conversation without holding per-sid lock to avoid KeyErrors
        # if the client disconnects while this task runs.
        history = list(conversation_history.get(sid, []))

        # Stream tokens + audio
        llm_response = get_llm_response_streaming(history, sid, gen_id=gen_id)

        # If not cancelled, append and emit completion
        if current_generation_id.get(sid, 0) == gen_id:
            try:
                conversation_history[sid].append({"role": "assistant", "content": llm_response})
            except Exception:
                pass

            print(f"[STAGE 4 - LLM] LLM complete: '{llm_response[:50]}...'")
            socketio.emit('pipeline_stage', {
                'stage': 'llm_complete',
                'message': 'LLM response complete'
            }, room=sid)
            socketio.emit('assistant_complete', {'text': llm_response}, room=sid)
    except Exception as e:
        print(f"Background LLM task error (sid={sid}): {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Mark generation as inactive if this task corresponds to current gen
        if current_generation_id.get(sid, 0) == gen_id:
            generation_active[sid] = False

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    with buffer_locks[sid]:
        if sid in audio_buffers:
            del audio_buffers[sid]
        if sid in utterance_audio_full:
            del utterance_audio_full[sid]
        if sid in utterance_active:
            del utterance_active[sid]
        if sid in accumulated_text:
            del accumulated_text[sid]
        if sid in conversation_history:
            del conversation_history[sid]
        if sid in silence_counters:
            del silence_counters[sid]
        if sid in last_partial_text:
            del last_partial_text[sid]
        if sid in vad_detectors:
            del vad_detectors[sid]
        if sid in vad_modes:
            del vad_modes[sid]
        if sid in buffer_locks:
            del buffer_locks[sid]

@socketio.on('set_vad_mode')
def handle_set_vad_mode(data):
    """Deprecated: Smart Turn removed. Always use Silero."""
    sid = request.sid
    with buffer_locks[sid]:
        vad_modes[sid] = 'silero'
        if sid in vad_detectors:
            del vad_detectors[sid]
        try:
            _ = get_vad_detector(sid)
            emit('vad_mode_changed', {'mode': 'silero', 'message': 'Using Silero VAD'})
        except Exception as e:
            emit('error', {'message': f'Error loading Silero VAD: {str(e)}'})

@socketio.on('start_stream')
def handle_start_stream():
    sid = request.sid
    with buffer_locks[sid]:
        audio_buffers[sid] = []
        utterance_audio_full[sid] = []
        utterance_active[sid] = False
        accumulated_text[sid] = ""
        silence_counters[sid] = 0
        last_partial_text[sid] = ""
        # Initialize VAD detector for this session
        vad_detector = get_vad_detector(sid)
        vad_detector.reset()
    emit('stream_started', {
        'message': 'Stream initialized',
        'vad_mode': 'silero'
    })

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    sid = request.sid
    try:
        audio_blob = data['audio']
        sample_rate = data.get('sampleRate', 48000)

        audio_array = np.frombuffer(audio_blob, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0

        if sample_rate != 16000:
            audio_float = _resample_linear(audio_float, sample_rate, 16000)

        # Decide action and copy audio out with minimal locking
        action = 'none'
        is_final = False
        audio_to_process = None
        chunk_has_speech = False

        with buffer_locks[sid]:
            # Get VAD detector for this session
            vad_detector = get_vad_detector(sid)

            # Always add to buffer first
            audio_buffers[sid].append(audio_float)
            combined_audio = np.concatenate(audio_buffers[sid]) if audio_buffers[sid] else np.array([], dtype=np.float32)

            # Check if current chunk has speech
            chunk_has_speech = vad_detector.has_speech(audio_float)

            # Manage full-utterance buffer lifecycle
            if chunk_has_speech and not utterance_active[sid]:
                # Start a new utterance
                utterance_active[sid] = True
                utterance_audio_full[sid] = []
            if utterance_active[sid]:
                # Keep collecting audio (including intervening short silences)
                utterance_audio_full[sid].append(audio_float)

            # Barge-in: if user starts speaking while assistant is active, cancel current generation
            if chunk_has_speech and generation_active.get(sid, False):
                # Increment generation id to cancel any in-flight streams
                current_generation_id[sid] += 1
                generation_active[sid] = False
                print(f"[BARGE-IN] Cancelling assistant for sid={sid}, new gen_id={current_generation_id[sid]}")
                socketio.emit('assistant_cancel', {'reason': 'barge_in'}, room=sid)

            # STAGE 1: VAD Detection
            if chunk_has_speech:
                print(f"[STAGE 1 - VAD] Speech detected in chunk (sid: {sid})")
                socketio.emit('pipeline_stage', {
                    'stage': 'vad',
                    'message': 'Speech detected'
                }, room=sid)

            # Update silence counter (Silero only)
            if chunk_has_speech:
                silence_counters[sid] = 0
            else:
                silence_counters[sid] += 1

            # Minimum buffer size check
            if len(combined_audio) >= MIN_BUFFER_SIZE:
                # Check if we have speech in the full buffer
                buffer_has_speech = vad_detector.has_speech(combined_audio)

                # If no speech detected in buffer, clear it and wait
                if not buffer_has_speech:
                    # Use detector's configured silence_threshold
                    if silence_counters[sid] > getattr(vad_detector, 'silence_threshold', 8):
                        audio_buffers[sid] = []
                        silence_counters[sid] = 0
                        last_partial_text[sid] = ""
                else:
                    # STREAMING MODE: Transcribe while speaking (every 2 seconds)
                    should_stream_partial = (
                        chunk_has_speech and
                        len(combined_audio) >= STREAMING_CHUNK_SIZE
                    )

                    # FINAL MODE: User stopped speaking
                    is_turn_complete = vad_detector.is_turn_complete(audio_float)
                    should_finalize = (
                        is_turn_complete and
                        len(combined_audio) >= MIN_BUFFER_SIZE
                    )

                    if should_stream_partial or should_finalize:
                        is_final = should_finalize
                        action = 'final' if is_final else 'partial'
                        # Copy audio out for processing without holding the lock
                        # For partials, use rolling window; for final, use the full utterance buffer
                        if is_final and utterance_audio_full[sid]:
                            try:
                                audio_to_process = np.concatenate(utterance_audio_full[sid]).copy()
                            except Exception:
                                audio_to_process = combined_audio.copy()
                        else:
                            audio_to_process = combined_audio.copy()
                        # For partials, keep overlap to improve continuity
                        if not is_final:
                            if len(combined_audio) > OVERLAP_SAMPLES:
                                audio_buffers[sid] = [combined_audio[-OVERLAP_SAMPLES:]]
                        else:
                            # For final, reset buffer and partial text for next turn
                            audio_buffers[sid] = []
                            last_partial_text[sid] = ""
                            utterance_active[sid] = False
                            utterance_audio_full[sid] = []

        # If nothing to do, return quickly
        if action == 'none' or audio_to_process is None:
            return

        # Write temp wav outside the lock
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                pcm16 = np.clip(audio_to_process, -1.0, 1.0)
                pcm16 = (pcm16 * 32767.0).astype(np.int16)
                wf.writeframes(pcm16.tobytes())

        try:
            # STAGE 2: Transcription
            print(f"[STAGE 2 - TRANSCRIPTION] Starting transcription (is_final={is_final}, sid={sid})")
            socketio.emit('pipeline_stage', {
                'stage': 'transcription_start',
                'message': 'Transcribing audio...',
                'is_final': is_final
            }, room=sid)
            socketio.sleep(0)

            result = transcription_model.transcribe(tmp_path)
            text = getattr(result, "text", str(result))

            if text and text.strip():
                print(f"[STAGE 2 - TRANSCRIPTION] Transcribed text: '{text}' (is_final={is_final})")
                socketio.emit('pipeline_stage', {
                    'stage': 'transcription_complete',
                    'message': f'Transcribed: {text[:50]}...',
                    'is_final': is_final
                }, room=sid)
                socketio.sleep(0)

                if is_final:
                    # Add user message to conversation history
                    try:
                        conversation_history[sid].append({"role": "user", "content": text})
                    except Exception:
                        pass

                    # Send final user message to frontend IMMEDIATELY
                    print(f"[STAGE 3 - USER MESSAGE] Displaying user message in UI")
                    emit('user_message', {'text': text, 'is_final': True})
                    socketio.sleep(0)

                    # STAGE 3: LLM Processing
                    print(f"[STAGE 4 - LLM] Starting LLM processing")
                    socketio.emit('pipeline_stage', {
                        'stage': 'llm_start',
                        'message': 'Processing with LLM...'
                    }, room=sid)
                    socketio.sleep(0)

                    # Signal start of assistant response
                    emit('assistant_start')
                    # Mark generation active and assign new generation id
                    current_generation_id[sid] += 1
                    gen_id = current_generation_id[sid]
                    generation_active[sid] = True
                    socketio.sleep(0)

                    # Offload LLM+TTS to a background task for live streaming
                    socketio.start_background_task(_run_llm_and_tts_for_sid, sid, gen_id)
                else:
                    # PARTIAL transcription - only emit if different from last partial
                    if text != last_partial_text[sid]:
                        emit('partial_transcription', {'text': text, 'is_final': False})
                        last_partial_text[sid] = text
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        # Log exceptions for visibility instead of swallowing silently
        print(f"Error in audio_chunk handler: {e}")
        import traceback
        traceback.print_exc()

@socketio.on('stop_stream')
def handle_stop_stream():
    sid = request.sid
    try:
        with buffer_locks[sid]:
            vad_detector = get_vad_detector(sid)

            # Prefer full utterance buffer if available
            candidate_audio = None
            if utterance_active.get(sid, False) and utterance_audio_full.get(sid):
                try:
                    candidate_audio = np.concatenate(utterance_audio_full[sid])
                except Exception:
                    candidate_audio = None
            if candidate_audio is None and sid in audio_buffers and audio_buffers[sid]:
                combined_audio = np.concatenate(audio_buffers[sid])
                if vad_detector.has_speech(combined_audio):
                    candidate_audio = combined_audio

            if candidate_audio is not None and candidate_audio.size > 0:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    with wave.open(tmp_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        pcm16 = np.clip(candidate_audio, -1.0, 1.0)
                        pcm16 = (pcm16 * 32767.0).astype(np.int16)
                        wf.writeframes(pcm16.tobytes())

                    try:
                        result = transcription_model.transcribe(tmp_path)
                        text = getattr(result, "text", str(result))

                        if text and text.strip():
                            if accumulated_text[sid]:
                                accumulated_text[sid] += " " + text
                            else:
                                accumulated_text[sid] = text

                    finally:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass

            final_text = accumulated_text.get(sid, "")
            emit('final_transcription', {'text': final_text, 'final': True})

            # Cancel any active assistant generation (stop audio & text streaming)
            current_generation_id[sid] += 1
            generation_active[sid] = False
            socketio.emit('assistant_cancel', {'reason': 'stop'}, room=sid)

            audio_buffers[sid] = []
            utterance_audio_full[sid] = []
            utterance_active[sid] = False
            accumulated_text[sid] = ""

    except Exception as e:
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3003))
    socketio.run(app, host='localhost', port=port, debug=False, allow_unsafe_werkzeug=True)
