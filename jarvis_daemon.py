#!/usr/bin/env python3.11
"""Jarvis - Voice interface using Claude Daemon + OpenAI voice.

Uses the Claude Daemon for persistent conversation memory and tool access,
with OpenAI Realtime API for high-quality STT and TTS.
"""

import asyncio
import base64
import json
import os
import select
import sys
import termios
import tty

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
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 2400  # 100ms at 24kHz
FORMAT = pyaudio.paInt16

REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

EXIT_PHRASES = ["bye", "goodbye", "exit", "quit", "shut down", "shutdown"]


class JarvisDaemon:
    """Voice assistant using Claude Daemon for backend."""

    def __init__(self, voice="onyx", input_device=None, output_device=None):
        self.voice = voice
        self.input_device = input_device
        self.output_device = output_device

        # OpenAI API key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.openai_client = OpenAI()

        # Audio
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.realtime_ws = None

        # State
        self.running = False
        self.recording = False  # True when SPACE held
        self.speaking = False   # True when TTS playing
        self.audio_sent = False  # Track if audio was sent during recording
        self.audio_buffer = asyncio.Queue()
        self.tts_queue = asyncio.Queue()  # Queue for sentences to speak

        # Claude daemon client
        self.claude = None

    async def connect_daemon(self):
        """Connect to Claude Daemon."""
        try:
            self.claude = ClaudeDaemon()
            status = await self.claude.status()
            print(f"Connected to Claude Daemon (queries: {status.queries}, cost: ${status.session_cost_usd:.4f})")
            return True
        except DaemonNotRunningError as e:
            print(f"\033[91mError: {e}\033[0m")
            print("\033[93mStart the daemon with: ./clauded start\033[0m")
            return False

    async def connect_realtime(self):
        """Connect to OpenAI Realtime API for STT."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.realtime_ws = await websockets.connect(REALTIME_URL, additional_headers=headers)

        # Wait for session.created
        msg = await self.realtime_ws.recv()
        event = json.loads(msg)
        if event["type"] != "session.created":
            raise Exception(f"Unexpected event: {event['type']}")

        # Configure for STT only (no turn detection - we use push-to-talk)
        await self.realtime_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": "Transcribe speech accurately.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": None  # Push-to-talk mode
            }
        }))
        print("Connected to OpenAI Realtime (push-to-talk)")

    def start_audio_streams(self):
        """Initialize audio input/output streams."""
        input_kwargs = {
            "format": FORMAT, "channels": CHANNELS, "rate": SAMPLE_RATE,
            "input": True, "frames_per_buffer": CHUNK_SIZE
        }
        if self.input_device is not None:
            input_kwargs["input_device_index"] = self.input_device

        output_kwargs = {
            "format": FORMAT, "channels": CHANNELS, "rate": SAMPLE_RATE,
            "output": True, "frames_per_buffer": CHUNK_SIZE
        }
        if self.output_device is not None:
            output_kwargs["output_device_index"] = self.output_device

        self.input_stream = self.audio.open(**input_kwargs)
        self.output_stream = self.audio.open(**output_kwargs)

    async def capture_audio(self):
        """Capture mic audio when recording."""
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                audio_data = await loop.run_in_executor(
                    None,
                    lambda: self.input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                )

                # Only send audio when recording (push-to-talk)
                if self.recording:
                    audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                    await self.realtime_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64
                    }))
                    if not self.audio_sent:
                        print(f"\033[90m[Audio streaming...]\033[0m")
                    self.audio_sent = True

            except Exception as e:
                if self.running:
                    print(f"Audio error: {e}")
                break

    async def handle_realtime(self):
        """Handle Realtime API events."""
        while self.running:
            try:
                msg = await self.realtime_ws.recv()
                event = json.loads(msg)
                event_type = event.get("type", "")

                if event_type == "conversation.item.input_audio_transcription.completed":
                    text = event.get("transcript", "").strip()
                    if text:
                        print(f"\033[90m[Transcribed]\033[0m")
                        await self.on_transcript(text)

                elif event_type == "input_audio_buffer.committed":
                    print(f"\033[90m[Processing...]\033[0m")

                elif event_type == "error":
                    error = event.get("error", {})
                    err_msg = error.get("message", "")
                    # Suppress harmless buffer errors
                    if "buffer is empty" not in err_msg.lower() and "buffer too small" not in err_msg.lower():
                        print(f"\033[91mRealtime Error: {err_msg}\033[0m")

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                if self.running:
                    print(f"Realtime error: {e}")
                break

    async def on_transcript(self, text):
        """Handle transcribed speech - send to Claude Daemon."""
        print(f"\n\033[92mYou: {text}\033[0m")

        # Check for exit phrases
        text_lower = text.lower()
        for phrase in EXIT_PHRASES:
            if phrase in text_lower:
                print("\n\033[93mGoodbye!\033[0m")
                await self.speak("Goodbye!")
                self.running = False
                return

        # Send to Claude Daemon and stream response
        await self.query_claude(text)

    async def query_claude(self, prompt: str):
        """Send query to Claude Daemon and speak the response."""
        try:
            response_text = []
            current_sentence = ""

            print("\033[94mClaude: \033[0m", end="", flush=True)

            async for msg in self.claude.query(prompt):
                if isinstance(msg, TextMessage):
                    # Accumulate text
                    current_sentence += msg.content
                    print(msg.content, end="", flush=True)

                    # Check for complete sentences to speak
                    while True:
                        # Look for sentence endings
                        for i, char in enumerate(current_sentence):
                            if char in '.!?' and i < len(current_sentence) - 1:
                                next_char = current_sentence[i + 1]
                                if next_char in ' \n':
                                    # Found a complete sentence
                                    sentence = current_sentence[:i + 1].strip()
                                    current_sentence = current_sentence[i + 2:]
                                    if sentence and len(sentence) > 3:
                                        response_text.append(sentence)
                                        # Queue for TTS (processed in order by tts_worker)
                                        await self.queue_speech(sentence)
                                    break
                        else:
                            break

                elif isinstance(msg, ToolUseMessage):
                    print(f"\n\033[90m[Using {msg.tool}...]\033[0m", end="", flush=True)

                elif isinstance(msg, ToolResultMessage):
                    status = "done" if msg.success else "failed"
                    print(f"\033[90m[{status}]\033[0m", end="", flush=True)

                elif isinstance(msg, DoneMessage):
                    print(f"\n\033[90m[${msg.cost_usd:.4f}]\033[0m")

                elif isinstance(msg, ErrorMessage):
                    print(f"\n\033[91mError: {msg.message}\033[0m")
                    await self.speak(f"Sorry, I encountered an error: {msg.message}")

            # Queue any remaining text
            if current_sentence.strip():
                response_text.append(current_sentence.strip())
                await self.queue_speech(current_sentence.strip())

            # Wait for TTS queue to drain
            while not self.tts_queue.empty():
                await asyncio.sleep(0.1)
            # Wait for current speech to finish
            while self.speaking:
                await asyncio.sleep(0.1)

            print()  # Newline after response

        except DaemonBusyError:
            print("\033[91mDaemon is busy with another query\033[0m")
            await self.speak("I'm still processing a previous request. Please wait.")
        except DaemonNotRunningError:
            print("\033[91mDaemon connection lost\033[0m")
            await self.speak("I lost connection to the backend. Please restart me.")
            self.running = False
        except Exception as e:
            print(f"\033[91mQuery error: {e}\033[0m")
            await self.speak(f"Sorry, something went wrong: {e}")

    async def queue_speech(self, text: str):
        """Queue text for TTS (non-blocking)."""
        if text.strip():
            await self.tts_queue.put(text)

    async def speak(self, text: str):
        """Speak text immediately using OpenAI TTS (blocking)."""
        if not text.strip():
            return

        self.speaking = True
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
                await self.audio_buffer.put(chunk)

        except Exception as e:
            print(f"\033[91mTTS error: {e}\033[0m")
        finally:
            self.speaking = False

    async def tts_worker(self):
        """Process TTS queue in order."""
        while self.running:
            try:
                text = await asyncio.wait_for(self.tts_queue.get(), timeout=0.1)
                await self.speak(text)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.running:
                    print(f"TTS worker error: {e}")

    async def play_audio(self):
        """Play TTS audio from buffer."""
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                audio_data = await asyncio.wait_for(self.audio_buffer.get(), timeout=0.1)
                await loop.run_in_executor(None, lambda: self.output_stream.write(audio_data))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.running:
                    print(f"Playback error: {e}")
                break

    async def keyboard_handler(self):
        """Handle keyboard - SPACE to toggle recording."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)  # cbreak mode - still see output

            while self.running:
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if readable:
                    key = sys.stdin.read(1)

                    if key == ' ':
                        if not self.recording:
                            # Start recording
                            self.recording = True
                            self.audio_sent = False
                            # Interrupt any TTS playback
                            self.speaking = False
                            while not self.audio_buffer.empty():
                                try:
                                    self.audio_buffer.get_nowait()
                                except:
                                    break
                            print("\r\033[93m[Recording... press SPACE to send]\033[0m", end="", flush=True)
                        else:
                            # Stop and commit
                            self.recording = False
                            print("\r\033[K", end="", flush=True)
                            if self.audio_sent:
                                await self.realtime_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            else:
                                print("\r\033[90m(no audio captured)\033[0m")

                    elif key == 'q' or key == '\x03':
                        self.running = False
                        break

                await asyncio.sleep(0.01)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    async def run(self):
        """Main run loop."""
        try:
            # Connect to daemon first
            if not await self.connect_daemon():
                return

            # Connect to OpenAI Realtime
            await self.connect_realtime()

            # Start audio streams
            self.start_audio_streams()

            self.running = True

            print(f"\n\033[93m{'=' * 50}\033[0m")
            print(f"\033[93m  Jarvis (Daemon Mode) Ready\033[0m")
            print(f"\033[93m  Voice: {self.voice}\033[0m")
            print(f"\033[93m{'=' * 50}\033[0m")
            print(f"\033[93m  Press SPACE to start recording\033[0m")
            print(f"\033[93m  Press SPACE again to send\033[0m")
            print(f"\033[93m  Press 'q' to quit\033[0m")
            print(f"\033[93m  Say 'goodbye' to exit\033[0m")
            print(f"\033[93m{'=' * 50}\033[0m\n")

            # Run all tasks
            await asyncio.gather(
                self.capture_audio(),
                self.handle_realtime(),
                self.play_audio(),
                self.tts_worker(),
                self.keyboard_handler(),
            )

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.running = False
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        if self.audio:
            self.audio.terminate()
        if self.realtime_ws:
            await self.realtime_ws.close()
        if self.claude:
            await self.claude.close()


def list_devices():
    """List available audio devices."""
    p = pyaudio.PyAudio()
    print("INPUT DEVICES:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    print("\nOUTPUT DEVICES:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    p.terminate()


def find_airpods():
    """Find AirPods if connected. Returns (input_index, output_index) or (None, None)."""
    p = pyaudio.PyAudio()
    input_idx = None
    output_idx = None

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info['name'].lower()

        # Check for AirPods (matches "airpods", "airpods pro", "airpods max", etc.)
        if 'airpods' in name:
            if info['maxInputChannels'] > 0 and input_idx is None:
                input_idx = i
            if info['maxOutputChannels'] > 0 and output_idx is None:
                output_idx = i

    p.terminate()
    return input_idx, output_idx


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Jarvis - Voice interface using Claude Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python jarvis_daemon.py                    # Auto-detects AirPods if connected
  python jarvis_daemon.py -v nova            # Use nova voice
  python jarvis_daemon.py --no-airpods       # Use default system audio
  python jarvis_daemon.py --list-devices     # List audio devices
  python jarvis_daemon.py -i 2 -o 3          # Specify audio devices manually

AirPods are automatically detected and used when connected.
Use --no-airpods to disable this and use built-in mic/speakers.

Note: Requires the Claude Daemon to be running.
Start it with: ./clauded start
        """
    )
    parser.add_argument("--voice", "-v", default="onyx",
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                        help="TTS voice (default: onyx)")
    parser.add_argument("--input", "-i", type=int, metavar="INDEX",
                        help="Input device index")
    parser.add_argument("--output", "-o", type=int, metavar="INDEX",
                        help="Output device index")
    parser.add_argument("--list-devices", "-l", action="store_true",
                        help="List available audio devices")
    parser.add_argument("--no-airpods", action="store_true",
                        help="Disable automatic AirPods detection")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    input_device = args.input
    output_device = args.output

    # Auto-detect AirPods if no devices specified and not disabled
    if input_device is None and output_device is None and not args.no_airpods:
        airpods_in, airpods_out = find_airpods()
        if airpods_in is not None or airpods_out is not None:
            input_device = airpods_in
            output_device = airpods_out
            print(f"\033[96m🎧 AirPods detected - using for audio\033[0m")

    jarvis = JarvisDaemon(args.voice, input_device, output_device)
    asyncio.run(jarvis.run())


if __name__ == "__main__":
    main()
