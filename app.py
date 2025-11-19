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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=50000000)

# Load transcription model
transcription_model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

# Load Silero VAD model
print("Loading Silero VAD model...")
vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, _, _, _, _) = vad_utils
print("Silero VAD loaded successfully!")

# Load LLM model
print("Loading Qwen/Qwen3-0.6B-MLX-4bit model...")
llm_model, llm_tokenizer = load("Qwen/Qwen3-0.6B-MLX-4bit")
print("Model loaded successfully!")

# Session-based audio buffers for streaming
audio_buffers = defaultdict(list)
buffer_locks = defaultdict(threading.Lock)
accumulated_text = defaultdict(str)  # Track accumulated transcription per session
conversation_history = defaultdict(list)  # Track conversation history for each session
silence_counters = defaultdict(int)  # Track consecutive silent chunks
SILENCE_THRESHOLD = 10  # Number of silent chunks before considering speech ended

def _resample_linear(x, sr_in, sr_out):
    """Lightweight linear resampler"""
    if sr_in == sr_out or x.size == 0:
        return x
    duration = x.shape[0] / float(sr_in)
    n_out = max(1, int(round(duration * sr_out)))
    t_in = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
    t_out = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)

def detect_end_of_speech(audio_data, threshold=0.5):
    """
    Detect if there's sustained silence (end of speech)

    Returns:
        tuple: (has_speech_now, is_end_of_speech)
    """
    has_speech_now = has_speech(audio_data, threshold)

    # End of speech is detected when we have no speech in current chunk
    # but we had speech before (buffer is not empty)
    is_end_of_speech = not has_speech_now

    return has_speech_now, is_end_of_speech

def has_speech(audio_data, threshold=0.5):
    """
    Silero VAD-based speech detection

    Args:
        audio_data: numpy array of audio samples (float32, -1 to 1)
        threshold: confidence threshold for speech detection (0.0-1.0)

    Returns:
        bool: True if speech is detected
    """
    if len(audio_data) == 0:
        return False

    # Quick energy check to avoid unnecessary VAD processing
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms < 0.001:  # Very low energy, definitely no speech
        return False

    try:
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_data).float()

        # Get speech timestamps from Silero VAD
        # sampling_rate must be 16000 or 8000
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            sampling_rate=16000,
            threshold=threshold,
            min_speech_duration_ms=250,  # Minimum 250ms of speech
            min_silence_duration_ms=100,  # Minimum 100ms of silence to split
            return_seconds=False
        )

        # If we found any speech segments, return True
        return len(speech_timestamps) > 0

    except Exception as e:
        print(f"VAD error: {e}")
        # Fallback to energy-based detection
        return rms > 0.01

def get_llm_response_streaming(conversation_messages, sid):
    """Generate LLM response from conversation history with streaming"""
    try:
        # Apply chat template with conversation history
        if llm_tokenizer.chat_template is not None:
            prompt = llm_tokenizer.apply_chat_template(
                conversation_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking for faster response
            )
        else:
            # Fallback if no chat template
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_messages])

        # Stream generate response
        full_response = ""
        for response in stream_generate(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=512,
        ):
            # Extract the text from the response object
            token = response.text if hasattr(response, 'text') else str(response)
            full_response += token
            # Emit each token as it's generated
            socketio.emit('assistant_token', {'token': token}, room=sid)

        return full_response.strip()
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return f"Error: {str(e)}"

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
        if sid in accumulated_text:
            del accumulated_text[sid]
        if sid in conversation_history:
            del conversation_history[sid]
        if sid in silence_counters:
            del silence_counters[sid]
        if sid in buffer_locks:
            del buffer_locks[sid]

@socketio.on('start_stream')
def handle_start_stream():
    sid = request.sid
    with buffer_locks[sid]:
        audio_buffers[sid] = []
        accumulated_text[sid] = ""
        silence_counters[sid] = 0
    emit('stream_started', {'message': 'Stream initialized'})

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

        with buffer_locks[sid]:
            # Always add to buffer first
            audio_buffers[sid].append(audio_float)
            combined_audio = np.concatenate(audio_buffers[sid])

            # Check if current chunk has speech
            chunk_has_speech = has_speech(audio_float)

            if chunk_has_speech:
                # Reset silence counter when speech detected
                silence_counters[sid] = 0
            else:
                # Increment silence counter
                silence_counters[sid] += 1

            # Minimum buffer size check (1.5 seconds at 16kHz)
            min_samples = 24000
            if len(combined_audio) < min_samples:
                return

            # Check if we have speech in the full buffer
            buffer_has_speech = has_speech(combined_audio)

            # If no speech detected in buffer, clear it and wait
            if not buffer_has_speech:
                if silence_counters[sid] > SILENCE_THRESHOLD:
                    audio_buffers[sid] = []
                    silence_counters[sid] = 0
                return

            # If we detected sustained silence after speech, process the buffer
            if silence_counters[sid] >= SILENCE_THRESHOLD and len(combined_audio) >= min_samples:
                # Process the speech segment
                pass  # Continue to transcription below
            else:
                # Keep accumulating
                return

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                with wave.open(tmp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    pcm16 = np.clip(combined_audio, -1.0, 1.0)
                    pcm16 = (pcm16 * 32767.0).astype(np.int16)
                    wf.writeframes(pcm16.tobytes())

            try:
                result = transcription_model.transcribe(tmp_path)
                text = getattr(result, "text", str(result))

                if text and text.strip():
                    # Add user message to conversation history
                    conversation_history[sid].append({"role": "user", "content": text})

                    # Send user message to frontend
                    emit('user_message', {'text': text})

                    # Signal start of assistant response
                    emit('assistant_start')

                    # Generate LLM response with streaming
                    llm_response = get_llm_response_streaming(conversation_history[sid], sid)

                    # Add assistant response to conversation history
                    conversation_history[sid].append({"role": "assistant", "content": llm_response})

                    # Signal end of assistant response
                    emit('assistant_complete', {'text': llm_response})

                    # Clear audio buffer for next segment
                    audio_buffers[sid] = []

            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    except Exception as e:
        pass

@socketio.on('stop_stream')
def handle_stop_stream():
    sid = request.sid
    try:
        with buffer_locks[sid]:
            if sid in audio_buffers and audio_buffers[sid]:
                combined_audio = np.concatenate(audio_buffers[sid])

                if has_speech(combined_audio):
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        with wave.open(tmp_path, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(16000)
                            pcm16 = np.clip(combined_audio, -1.0, 1.0)
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

            audio_buffers[sid] = []
            accumulated_text[sid] = ""

    except Exception as e:
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3003))
    socketio.run(app, host='localhost', port=port, debug=False, allow_unsafe_werkzeug=True)
