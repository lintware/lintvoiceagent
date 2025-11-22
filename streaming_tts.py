#!/usr/bin/env python3
"""
Streaming TTS Module for Voice Agent

Provides real-time text-to-speech streaming using Kokoro TTS.
Converts text chunks into audio as they arrive from the LLM.
"""

from kokoro import KModel, KPipeline
import numpy as np
import base64
import io
import wave
import re


class StreamingTTS:
    """Handles streaming text-to-speech conversion."""

    def __init__(self, voice='af_heart', speed=1.0, use_gpu=False):
        """
        Initialize the streaming TTS engine.

        Args:
            voice: Voice preset (e.g., 'af_heart', 'af_bella', 'am_michael')
            speed: Speech speed multiplier (0.5 - 2.0)
            use_gpu: Whether to use GPU (if available)
        """
        self.voice = voice
        self.speed = speed
        self.lang_code = voice[0]  # 'a' for American, 'b' for British

        # Initialize model and pipeline
        device = 'cuda' if use_gpu else 'cpu'
        self.model = KModel().to(device).eval()
        self.pipeline = KPipeline(lang_code=self.lang_code, model=False)

        # Load voice pack
        self.voice_pack = self.pipeline.load_voice(voice)

        # Text buffer for incomplete sentences
        self.text_buffer = ""

    def generate_audio_chunk(self, text):
        """
        Generate audio for a text chunk.

        Args:
            text: Text to convert to speech

        Yields:
            bytes: WAV audio data for each chunk
        """
        if not text or not text.strip():
            return

        # Generate audio using the pipeline
        for _, ps, _ in self.pipeline(text, self.voice, self.speed):
            ref_s = self.voice_pack[len(ps)-1]
            audio = self.model(ps, ref_s, self.speed)

            # Convert to WAV format
            audio_np = audio.numpy()
            wav_bytes = self._audio_to_wav_bytes(audio_np, sample_rate=24000)

            yield wav_bytes

    def _audio_to_wav_bytes(self, audio_data, sample_rate=24000):
        """
        Convert numpy audio array to WAV bytes.

        Args:
            audio_data: Numpy array of audio samples
            sample_rate: Sample rate in Hz

        Returns:
            bytes: WAV file as bytes
        """
        # Ensure audio is in the right format
        if audio_data.dtype != np.int16:
            # Convert float32 to int16
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)   # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return wav_buffer.getvalue()

    def add_text(self, token):
        """
        Add a token to the buffer and return complete sentences.

        Args:
            token: Text token from LLM

        Returns:
            str or None: Complete sentence if available, None otherwise
        """
        self.text_buffer += token

        # Check for sentence boundaries
        # Look for . ! ? followed by space or newline
        sentence_pattern = r'([.!?;]+[\s\n]+)'
        match = re.search(sentence_pattern, self.text_buffer)

        if match:
            # Found a complete sentence
            end_pos = match.end()
            sentence = self.text_buffer[:end_pos].strip()
            self.text_buffer = self.text_buffer[end_pos:]
            return sentence

        return None

    def flush(self):
        """
        Get any remaining text in the buffer.

        Returns:
            str: Remaining buffered text
        """
        remaining = self.text_buffer.strip()
        self.text_buffer = ""
        return remaining


class TextChunker:
    """Helper class to chunk text into speakable phrases."""

    # Sentence-ending punctuation
    SENTENCE_END = r'[.!?;]+[\s\n]+'

    # Phrase-breaking punctuation (for faster streaming)
    PHRASE_BREAK = r'[,:][\s]+'

    # Minimum characters before considering a break
    MIN_CHUNK_SIZE = 20

    def __init__(self, mode='sentence'):
        """
        Initialize chunker.

        Args:
            mode: 'sentence' for full sentences, 'phrase' for faster streaming
        """
        self.mode = mode
        self.buffer = ""

    def add_token(self, token):
        """
        Add a token and return complete chunks.

        Args:
            token: Text token from LLM

        Yields:
            str: Complete chunks ready for TTS
        """
        self.buffer += token

        # Choose pattern based on mode
        if self.mode == 'phrase':
            # Break on both sentences and phrases
            pattern = f'({self.SENTENCE_END}|{self.PHRASE_BREAK})'
        else:
            # Only break on sentences
            pattern = f'({self.SENTENCE_END})'

        # Keep finding complete chunks
        while True:
            # Only look for breaks if we have minimum chunk size
            if len(self.buffer) < self.MIN_CHUNK_SIZE:
                break

            match = re.search(pattern, self.buffer)
            if not match:
                break

            # Found a chunk
            end_pos = match.end()
            chunk = self.buffer[:end_pos].strip()
            self.buffer = self.buffer[end_pos:]

            if chunk:
                yield chunk

    def flush(self):
        """
        Get any remaining text.

        Returns:
            str or None: Remaining text if any
        """
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None


def audio_to_base64(wav_bytes):
    """
    Convert WAV bytes to base64 for transmission.

    Args:
        wav_bytes: WAV file as bytes

    Returns:
        str: Base64-encoded audio
    """
    return base64.b64encode(wav_bytes).decode('utf-8')
