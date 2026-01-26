#!/usr/bin/env python3.11
"""Jarvis with Wake Word - Always listening for "Jarvis" activation.

Uses Picovoice Porcupine for local wake word detection (no cloud calls),
then OpenAI for STT/TTS and Claude Daemon for conversation.
"""

import asyncio
import base64
import json
import os
import struct
import sys
import time

import numpy as np
import pvporcupine
import pyaudio
import websockets
from openai import OpenAI

from claude_client import (
    ClaudeDaemon,
    DaemonNotRunningError,
    DaemonBusyError,
    TextMessage,
    ToolUseMessage,
    ToolResultMessage,
    DoneMessage,
    ErrorMessage,
)

# Audio settings
SAMPLE_RATE = 16000  # Porcupine requires 16kHz
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Wake word settings
WAKE_WORD_PATH = os.path.join(os.path.dirname(__file__), "Jarvis_en_mac_v4_0_0.ppn")

# Silence detection for end of speech
SILENCE_THRESHOLD = 500  # Audio level below this = silence
SILENCE_DURATION = 1.5   # Seconds of silence to end recording
MAX_RECORDING_DURATION = 30  # Max seconds to record

# OpenAI Realtime for STT
REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

EXIT_PHRASES = ["goodbye", "shut down", "shutdown", "exit", "quit", "stop listening"]


def find_best_audio_devices():
    """Find best audio devices. Priority: AirPods > TONOR/USB mic > MacBook built-in."""
    p = pyaudio.PyAudio()

    airpods_in, airpods_out = None, None
    usb_mic_in = None
    macbook_in, macbook_out = None, None

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info['name'].lower()

        # AirPods - highest priority
        if 'airpods' in name:
            if info['maxInputChannels'] > 0 and airpods_in is None:
                airpods_in = i
            if info['maxOutputChannels'] > 0 and airpods_out is None:
                airpods_out = i

        # USB microphone (TONOR, etc.) - second priority for input
        elif 'usb' in name or 'tonor' in name:
            if info['maxInputChannels'] > 0 and usb_mic_in is None:
                usb_mic_in = i

        # MacBook built-in - fallback
        elif 'macbook' in name:
            if info['maxInputChannels'] > 0 and macbook_in is None:
                macbook_in = i
            if info['maxOutputChannels'] > 0 and macbook_out is None:
                macbook_out = i

    p.terminate()

    # Determine best devices
    if airpods_in is not None:
        return airpods_in, airpods_out, "AirPods"
    elif usb_mic_in is not None:
        return usb_mic_in, macbook_out, "TONOR USB mic + MacBook speakers"
    elif macbook_in is not None:
        return macbook_in, macbook_out, "MacBook built-in"
    else:
        return None, None, "system default"


def find_airpods():
    """Find AirPods if connected. Returns (input_index, output_index) or (None, None)."""
    p = pyaudio.PyAudio()
    input_idx = None
    output_idx = None

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info['name'].lower()

        if 'airpods' in name:
            if info['maxInputChannels'] > 0 and input_idx is None:
                input_idx = i
            if info['maxOutputChannels'] > 0 and output_idx is None:
                output_idx = i

    p.terminate()
    return input_idx, output_idx


class JarvisWakeWord:
    """Voice assistant with wake word activation."""

    def __init__(self, voice="onyx", input_device=None, output_device=None):
        self.voice = voice
        self.input_device = input_device
        self.output_device = output_device

        # API keys
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.picovoice_key = os.environ.get("PICOVOICE_ACCESS_KEY")
        if not self.picovoice_key:
            raise ValueError("PICOVOICE_ACCESS_KEY not set")

        self.openai_client = OpenAI()

        # Porcupine wake word
        self.porcupine = None

        # Audio
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

        # State
        self.running = False
        self.listening_for_wake = True

        # Claude daemon
        self.claude = None

    async def connect_daemon(self):
        """Connect to Claude Daemon."""
        try:
            self.claude = ClaudeDaemon()
            status = await self.claude.status()
            print(f"Connected to daemon (queries: {status.queries})")
            return True
        except DaemonNotRunningError as e:
            print(f"\033[91mError: {e}\033[0m")
            print("\033[93mStart with: ./clauded start\033[0m")
            return False

    def init_porcupine(self):
        """Initialize Porcupine wake word detection."""
        self.porcupine = pvporcupine.create(
            access_key=self.picovoice_key,
            keyword_paths=[WAKE_WORD_PATH]
        )
        print(f"Wake word loaded: Jarvis")

    def start_audio(self):
        """Start audio input stream for wake word detection."""
        input_kwargs = {
            "format": FORMAT,
            "channels": CHANNELS,
            "rate": SAMPLE_RATE,  # 16kHz for Porcupine
            "input": True,
            "frames_per_buffer": self.porcupine.frame_length
        }
        if self.input_device is not None:
            input_kwargs["input_device_index"] = self.input_device

        # Output at 24kHz for OpenAI TTS
        output_kwargs = {
            "format": FORMAT,
            "channels": CHANNELS,
            "rate": 24000,
            "output": True,
            "frames_per_buffer": 2400
        }
        if self.output_device is not None:
            output_kwargs["output_device_index"] = self.output_device

        self.input_stream = self.audio.open(**input_kwargs)
        self.output_stream = self.audio.open(**output_kwargs)

    def play_chime(self, freq=880, duration=0.15):
        """Play activation chime."""
        samples = int(24000 * duration)
        t = np.linspace(0, duration, samples, False)
        wave = np.sin(2 * np.pi * freq * t) * 0.3
        # Fade out
        fade = np.linspace(1, 0, samples)
        wave = (wave * fade * 32767).astype(np.int16)
        self.output_stream.write(wave.tobytes())

    def get_audio_level(self, audio_data):
        """Get RMS audio level from raw bytes."""
        if len(audio_data) < 2:
            return 0
        shorts = struct.unpack(f'{len(audio_data)//2}h', audio_data)
        return int(np.sqrt(np.mean(np.square(shorts))))

    def calibrate_silence_threshold(self, duration=1.0):
        """Measure ambient noise level and set threshold."""
        print("\033[90m[Calibrating...]\033[0m", end="", flush=True)
        levels = []
        samples = int(duration * SAMPLE_RATE / self.porcupine.frame_length)

        for _ in range(samples):
            audio_data = self.input_stream.read(
                self.porcupine.frame_length,
                exception_on_overflow=False
            )
            levels.append(self.get_audio_level(audio_data))

        avg_noise = sum(levels) / len(levels) if levels else 500
        # Simple: threshold = avg * 1.7 (worked at 1615 with avg ~950)
        self.silence_threshold = int(avg_noise * 1.7)
        print(f" avg={int(avg_noise)} thr={self.silence_threshold}")
        print(f" threshold={self.silence_threshold}")
        return self.silence_threshold

    async def record_until_silence(self):
        """Record audio until silence detected. Returns audio bytes."""
        print("\033[92m[Listening...]\033[0m", end="", flush=True)

        frames = []
        silence_start = None
        recording_start = time.time()
        speech_detected = False

        # Need to read at 16kHz for consistency, will resample for OpenAI
        while self.running:
            audio_data = self.input_stream.read(
                self.porcupine.frame_length,
                exception_on_overflow=False
            )
            frames.append(audio_data)

            level = self.get_audio_level(audio_data)
            threshold = getattr(self, 'silence_threshold', SILENCE_THRESHOLD)

            # Must detect speech before we start looking for silence
            if level > threshold * 1.2:
                speech_detected = True
                silence_start = None
            elif speech_detected and level < threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print(" \033[90m[done]\033[0m")
                    break

            # Max duration check
            if time.time() - recording_start > MAX_RECORDING_DURATION:
                print(" \033[90m[max time]\033[0m")
                break

            await asyncio.sleep(0)  # Yield to event loop

        return b''.join(frames)

    async def transcribe_audio(self, audio_data):
        """Transcribe audio using OpenAI Realtime API."""
        if not audio_data:
            return ""

        try:
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "OpenAI-Beta": "realtime=v1"
            }

            async with websockets.connect(REALTIME_URL, additional_headers=headers) as ws:
                # Wait for session
                msg = await ws.recv()
                event = json.loads(msg)
                if event["type"] != "session.created":
                    return ""

                # Configure for transcription
                await ws.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "input_audio_format": "pcm16",
                        "input_audio_transcription": {"model": "whisper-1"},
                        "turn_detection": None
                    }
                }))

                # Resample from 16kHz to 24kHz for OpenAI
                audio_24k = self.resample_audio(audio_data, 16000, 24000)

                # Send audio
                audio_b64 = base64.b64encode(audio_24k).decode("utf-8")
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }))

                # Commit and request response
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await ws.send(json.dumps({"type": "response.create"}))

                # Wait for transcription
                transcript = ""
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    event = json.loads(msg)

                    if event["type"] == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "").strip()
                        break
                    elif event["type"] == "response.done":
                        break
                    elif event["type"] == "error":
                        print(f"\033[91mSTT error: {event.get('error', {}).get('message')}\033[0m")
                        break

                return transcript

        except Exception as e:
            print(f"\033[91mTranscription error: {e}\033[0m")
            return ""

    def resample_audio(self, audio_data, from_rate, to_rate):
        """Simple resampling using linear interpolation."""
        shorts = np.frombuffer(audio_data, dtype=np.int16)
        duration = len(shorts) / from_rate
        new_length = int(duration * to_rate)
        indices = np.linspace(0, len(shorts) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(shorts)), shorts)
        return resampled.astype(np.int16).tobytes()

    async def query_claude(self, prompt: str):
        """Send query to Claude and speak response."""
        try:
            print(f"\033[94mClaude:\033[0m ", end="", flush=True)
            full_response = []

            async for msg in self.claude.query(prompt):
                if isinstance(msg, TextMessage):
                    print(msg.content, end="", flush=True)
                    full_response.append(msg.content)

                elif isinstance(msg, ToolUseMessage):
                    print(f"\n\033[90m[{msg.tool}]\033[0m", end="", flush=True)

                elif isinstance(msg, ToolResultMessage):
                    status = "✓" if msg.success else "✗"
                    print(f"\033[90m[{status}]\033[0m", end="", flush=True)

                elif isinstance(msg, DoneMessage):
                    print(f"\n\033[90m[${msg.cost_usd:.4f}]\033[0m")

                elif isinstance(msg, ErrorMessage):
                    print(f"\n\033[91mError: {msg.message}\033[0m")
                    await self.speak("Sorry, an error occurred.")
                    return

            # Speak the full response
            response_text = "".join(full_response).strip()
            if response_text:
                await self.speak(response_text)

        except DaemonBusyError:
            print("\033[91mDaemon busy\033[0m")
            await self.speak("I'm still working on something. One moment.")
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")
            await self.speak("Sorry, something went wrong.")

    async def speak(self, text: str):
        """Speak text using OpenAI TTS."""
        if not text.strip():
            return

        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text,
                response_format="pcm"
            )

            for chunk in response.iter_bytes(chunk_size=4096):
                if not self.running:
                    break
                self.output_stream.write(chunk)

        except Exception as e:
            print(f"\033[91mTTS error: {e}\033[0m")

    async def run(self):
        """Main run loop."""
        try:
            # Connect to daemon
            if not await self.connect_daemon():
                return

            # Initialize Porcupine
            self.init_porcupine()

            # Start audio
            self.start_audio()

            # Calibrate silence threshold based on ambient noise
            self.calibrate_silence_threshold()

            self.running = True

            print(f"\n\033[93m{'=' * 50}\033[0m")
            print(f"\033[93m  Jarvis Ready - Say 'Jarvis' to activate\033[0m")
            print(f"\033[93m  Voice: {self.voice}\033[0m")
            print(f"\033[93m{'=' * 50}\033[0m")
            print(f"\033[93m  Say 'goodbye' to exit\033[0m")
            print(f"\033[93m{'=' * 50}\033[0m\n")

            print("\033[90m[Listening for 'Jarvis'...]\033[0m")

            # Main loop
            frame_count = 0
            while self.running:
                # Read audio frame for wake word detection
                pcm = self.input_stream.read(
                    self.porcupine.frame_length,
                    exception_on_overflow=False
                )
                pcm_unpacked = struct.unpack_from(
                    "h" * self.porcupine.frame_length, pcm
                )

                # Check for wake word
                keyword_index = self.porcupine.process(pcm_unpacked)

                # Heartbeat every ~5 seconds (frame_length is 512 at 16kHz = 32ms per frame)
                frame_count += 1
                if frame_count % 156 == 0:  # ~5 seconds
                    print(".", end="", flush=True)

                if keyword_index >= 0:
                    print("\n\033[96m[Jarvis activated]\033[0m")
                    self.play_chime()

                    # Record until silence
                    audio_data = await self.record_until_silence()

                    # Transcribe
                    transcript = await self.transcribe_audio(audio_data)

                    if transcript:
                        print(f"\n\033[92mYou: {transcript}\033[0m")

                        # Check for exit
                        if any(phrase in transcript.lower() for phrase in EXIT_PHRASES):
                            print("\n\033[93mGoodbye!\033[0m")
                            await self.speak("Goodbye!")
                            self.running = False
                            break

                        # Query Claude
                        await self.query_claude(transcript)
                    else:
                        print("\033[90m[No speech detected]\033[0m")

                    print("\n\033[90m[Listening for 'Jarvis'...]\033[0m")

                await asyncio.sleep(0)  # Yield to event loop

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        if self.audio:
            self.audio.terminate()
        if self.porcupine:
            self.porcupine.delete()
        if self.claude:
            await self.claude.close()


def list_devices():
    """List audio devices."""
    p = pyaudio.PyAudio()
    print("INPUT:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    print("\nOUTPUT:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    p.terminate()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Jarvis with wake word activation")
    parser.add_argument("--voice", "-v", default="onyx",
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
    parser.add_argument("--input", "-i", type=int, help="Input device index")
    parser.add_argument("--output", "-o", type=int, help="Output device index")
    parser.add_argument("--list-devices", "-l", action="store_true")
    parser.add_argument("--no-airpods", action="store_true")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    input_device = args.input
    output_device = args.output

    # Auto-detect best audio devices
    if input_device is None and output_device is None and not args.no_airpods:
        input_device, output_device, device_name = find_best_audio_devices()
        if input_device is not None:
            print(f"\033[96m🎤 Using: {device_name}\033[0m")

    jarvis = JarvisWakeWord(args.voice, input_device, output_device)
    asyncio.run(jarvis.run())


if __name__ == "__main__":
    main()
