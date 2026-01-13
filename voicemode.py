#!/usr/bin/env python3
"""Voice mode for learning and reviewing markdown notes using OpenAI Realtime API."""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import pyaudio
import websockets

# Audio settings (OpenAI Realtime API requirements)
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 2400  # 100ms at 24kHz
OUTPUT_BUFFER_SIZE = 4800  # 200ms output buffer for smoother playback
FORMAT = pyaudio.paInt16
PREBUFFER_CHUNKS = 3  # Buffer 3 chunks (~300ms) before starting playback

# WebSocket URL
REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

# Exit phrases (lowercase for matching)
EXIT_PHRASES = [
    "ok bye bye", "okay bye bye", "bye bye",
    "shut down", "shutdown",
    "exit", "quit", "stop",
    "goodbye", "good bye",
    "end session", "close session"
]


class VoiceMode:
    def __init__(self, markdown_path: str, voice: str = "alloy",
                 input_device: int = None, output_device: int = None):
        self.markdown_path = Path(markdown_path)
        self.voice = voice
        self.input_device = input_device
        self.output_device = output_device
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Load markdown content
        self.markdown_content = self.markdown_path.read_text()

        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

        # State
        self.ws = None
        self.running = False
        self.audio_buffer = asyncio.Queue()
        self.is_playing = False

    def build_system_prompt(self) -> str:
        return f"""You are helping the user learn and review the following material. Your role is to:
- Quiz them on key concepts and characters
- Explain ideas when asked
- Answer questions about the material
- Help them internalize and remember the content
- Be conversational and encouraging

Keep responses concise for voice - aim for 1-3 sentences unless they ask for more detail.

---
{self.markdown_content}
---"""

    async def connect(self):
        """Connect to OpenAI Realtime API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }

        self.ws = await websockets.connect(
            REALTIME_URL,
            additional_headers=headers
        )
        print("Connected to OpenAI Realtime API")

        # Wait for session.created
        msg = await self.ws.recv()
        event = json.loads(msg)
        if event["type"] == "session.created":
            print("Session created")

        # Configure session
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": self.build_system_prompt(),
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 1000
                }
            }
        }))
        print(f"Session configured with voice: {self.voice}")
        print(f"Loaded: {self.markdown_path.name}")
        print("\nSpeak to start the conversation. Ctrl+C to quit.\n")

    def start_audio_streams(self):
        """Initialize audio input/output streams."""
        input_kwargs = {
            "format": FORMAT,
            "channels": CHANNELS,
            "rate": SAMPLE_RATE,
            "input": True,
            "frames_per_buffer": CHUNK_SIZE
        }
        if self.input_device is not None:
            input_kwargs["input_device_index"] = self.input_device

        output_kwargs = {
            "format": FORMAT,
            "channels": CHANNELS,
            "rate": SAMPLE_RATE,
            "output": True,
            "frames_per_buffer": OUTPUT_BUFFER_SIZE
        }
        if self.output_device is not None:
            output_kwargs["output_device_index"] = self.output_device

        self.input_stream = self.audio.open(**input_kwargs)
        self.output_stream = self.audio.open(**output_kwargs)

    async def capture_audio(self):
        """Capture audio from microphone and send to WebSocket."""
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                # Read audio chunk in executor to avoid blocking
                audio_data = await loop.run_in_executor(
                    None,
                    lambda: self.input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                )

                # Convert to base64 and send
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                await self.ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }))

            except Exception as e:
                if self.running:
                    print(f"Audio capture error: {e}")
                break

    async def play_audio(self):
        """Play audio chunks from the buffer with pre-buffering for smoothness."""
        loop = asyncio.get_event_loop()
        prebuffer = []
        buffering = True

        while self.running:
            try:
                audio_data = await asyncio.wait_for(
                    self.audio_buffer.get(),
                    timeout=0.1
                )

                if buffering:
                    # Accumulate chunks before starting playback
                    prebuffer.append(audio_data)
                    if len(prebuffer) >= PREBUFFER_CHUNKS:
                        # Play all buffered audio
                        self.is_playing = True
                        combined = b''.join(prebuffer)
                        await loop.run_in_executor(
                            None,
                            lambda c=combined: self.output_stream.write(c)
                        )
                        prebuffer = []
                        buffering = False
                else:
                    # Normal playback - batch a couple chunks if available
                    chunks = [audio_data]
                    while not self.audio_buffer.empty() and len(chunks) < 2:
                        try:
                            chunks.append(self.audio_buffer.get_nowait())
                        except asyncio.QueueEmpty:
                            break
                    combined = b''.join(chunks)
                    await loop.run_in_executor(
                        None,
                        lambda c=combined: self.output_stream.write(c)
                    )

            except asyncio.TimeoutError:
                # Flush any remaining prebuffer on timeout
                if prebuffer:
                    self.is_playing = True
                    combined = b''.join(prebuffer)
                    await loop.run_in_executor(
                        None,
                        lambda c=combined: self.output_stream.write(c)
                    )
                    prebuffer = []
                buffering = True  # Reset for next response
                self.is_playing = False
                continue
            except Exception as e:
                if self.running:
                    print(f"Audio playback error: {e}")
                break

    async def handle_events(self):
        """Handle incoming WebSocket events."""
        while self.running:
            try:
                msg = await self.ws.recv()
                event = json.loads(msg)
                event_type = event.get("type", "")

                if event_type == "response.audio.delta":
                    # Decode and queue audio for playback
                    audio_b64 = event.get("delta", "")
                    if audio_b64:
                        audio_data = base64.b64decode(audio_b64)
                        await self.audio_buffer.put(audio_data)

                elif event_type == "response.audio_transcript.delta":
                    # Print AI response transcript
                    text = event.get("delta", "")
                    print(f"\033[94m{text}\033[0m", end="", flush=True)

                elif event_type == "response.audio_transcript.done":
                    print()  # Newline after transcript

                elif event_type == "input_audio_buffer.speech_started":
                    # User started speaking - clear audio buffer for interruption
                    while not self.audio_buffer.empty():
                        try:
                            self.audio_buffer.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    # Print user transcript
                    text = event.get("transcript", "")
                    if text:
                        print(f"\033[92mYou: {text}\033[0m")
                        # Check for exit phrases
                        text_lower = text.lower().strip()
                        for phrase in EXIT_PHRASES:
                            if phrase in text_lower:
                                print("\n\033[93mGoodbye! Ending session...\033[0m")
                                self.running = False
                                return

                elif event_type == "error":
                    error = event.get("error", {})
                    print(f"\033[91mError: {error.get('message', 'Unknown error')}\033[0m")

            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break
            except Exception as e:
                if self.running:
                    print(f"Event handling error: {e}")
                break

    async def run(self):
        """Main run loop."""
        try:
            await self.connect()
            self.start_audio_streams()
            self.running = True

            # Run all tasks concurrently
            await asyncio.gather(
                self.capture_audio(),
                self.play_audio(),
                self.handle_events()
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
        if self.ws:
            await self.ws.close()


def list_devices():
    """List available audio devices."""
    p = pyaudio.PyAudio()
    print("INPUT DEVICES:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    print()
    print("OUTPUT DEVICES:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    p.terminate()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voice mode for learning markdown notes")
    parser.add_argument("markdown_file", nargs="?", help="Path to markdown file")
    parser.add_argument("--voice", "-v", default="alloy",
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                        help="Voice to use (default: alloy)")
    parser.add_argument("--input", "-i", type=int, help="Input device index")
    parser.add_argument("--output", "-o", type=int, help="Output device index")
    parser.add_argument("--list-devices", "-l", action="store_true",
                        help="List available audio devices")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    if not args.markdown_file:
        parser.print_help()
        sys.exit(1)

    if not Path(args.markdown_file).exists():
        print(f"Error: File not found: {args.markdown_file}")
        sys.exit(1)

    vm = VoiceMode(args.markdown_file, args.voice, args.input, args.output)
    asyncio.run(vm.run())


if __name__ == "__main__":
    main()
