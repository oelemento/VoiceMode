#!/usr/bin/env python3
"""Read epub books aloud with voice Q&A using OpenAI Realtime API."""

import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import pyaudio
import websockets

# Audio settings
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 2400
FORMAT = pyaudio.paInt16

# WebSocket URL
REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

# Commands
EXIT_PHRASES = ["bye bye", "stop reading", "exit", "quit", "shut down", "goodbye"]
RESUME_PHRASES = ["continue", "keep reading", "resume", "go on", "keep going"]


def extract_text_from_epub(epub_path: str) -> list[dict]:
    """Extract chapters from epub file."""
    book = epub.read_epub(epub_path)
    chapters = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')

            # Get title if available
            title = None
            title_tag = soup.find(['h1', 'h2', 'title'])
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'\n{3,}', '\n\n', text)  # Clean up excessive newlines

            if text and len(text) > 100:  # Skip very short sections
                chapters.append({
                    'title': title or f"Section {len(chapters) + 1}",
                    'text': text
                })

    return chapters


class BookReader:
    def __init__(self, epub_path: str, voice: str = "onyx",
                 input_device: int = None, output_device: int = None):
        self.epub_path = Path(epub_path)
        self.voice = voice
        self.input_device = input_device
        self.output_device = output_device
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Load book
        print(f"Loading: {self.epub_path.name}")
        self.chapters = extract_text_from_epub(str(epub_path))
        print(f"Found {len(self.chapters)} chapters")

        self.current_chapter = 0
        self.current_position = 0  # Character position in current chapter
        self.chunk_size = 500  # Characters to read at a time

        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

        # State
        self.ws = None
        self.running = False
        self.reading = False  # Currently reading aloud
        self.waiting_for_response = False
        self.audio_buffer = asyncio.Queue()
        self.pending_text = ""  # Text waiting to be read

    def get_context_window(self) -> str:
        """Get recent text for context (last ~2000 chars)."""
        if not self.chapters:
            return ""

        chapter = self.chapters[self.current_chapter]
        start = max(0, self.current_position - 2000)
        end = min(len(chapter['text']), self.current_position + 500)

        context = chapter['text'][start:end]
        return f"[Chapter: {chapter['title']}]\n\n{context}"

    def get_next_chunk(self) -> str | None:
        """Get next chunk of text to read."""
        if self.current_chapter >= len(self.chapters):
            return None

        chapter = self.chapters[self.current_chapter]

        if self.current_position >= len(chapter['text']):
            # Move to next chapter
            self.current_chapter += 1
            self.current_position = 0
            if self.current_chapter >= len(self.chapters):
                return None
            chapter = self.chapters[self.current_chapter]
            return f"\n\nChapter: {chapter['title']}\n\n" + chapter['text'][:self.chunk_size]

        chunk = chapter['text'][self.current_position:self.current_position + self.chunk_size]
        self.current_position += len(chunk)
        return chunk

    def build_system_prompt(self) -> str:
        return f"""You are reading the book "{self.epub_path.stem}" aloud to the user.

Your modes:
1. READING MODE: When told to read, speak the text naturally as an audiobook narrator. Read exactly what's given - don't summarize or paraphrase.

2. Q&A MODE: When the user interrupts with a question, answer based on the book content you've read so far. Be helpful and concise.

Current position context:
{self.get_context_window()}

Instructions:
- Read the text naturally with good pacing
- When user asks a question, answer it helpfully based on the book
- If user says "continue" or "keep reading", resume reading from where you left off
- Keep Q&A answers brief (1-3 sentences) unless user asks for more detail"""

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
                    "silence_duration_ms": 800
                }
            }
        }))
        print(f"Session configured with voice: {self.voice}")

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
            "frames_per_buffer": CHUNK_SIZE
        }
        if self.output_device is not None:
            output_kwargs["output_device_index"] = self.output_device

        self.input_stream = self.audio.open(**input_kwargs)
        self.output_stream = self.audio.open(**output_kwargs)

    async def send_text_to_read(self, text: str):
        """Send text for the AI to read aloud."""
        await self.ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": f"Please read this aloud naturally: {text}"
                }]
            }
        }))
        await self.ws.send(json.dumps({"type": "response.create"}))
        self.waiting_for_response = True

    async def reading_loop(self):
        """Manage the reading flow."""
        # Start with first chunk
        await asyncio.sleep(1)  # Let connection settle

        print("\n\033[93mStarting to read. Interrupt anytime to ask questions.\033[0m")
        print("\033[93mSay 'continue' to resume reading after questions.\033[0m\n")

        self.reading = True
        chunk = self.get_next_chunk()
        if chunk:
            await self.send_text_to_read(chunk)

    async def capture_audio(self):
        """Capture audio from microphone and send to WebSocket."""
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                audio_data = await loop.run_in_executor(
                    None,
                    lambda: self.input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                )

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
        """Play audio chunks from the buffer."""
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                audio_data = await asyncio.wait_for(
                    self.audio_buffer.get(),
                    timeout=0.1
                )

                await loop.run_in_executor(
                    None,
                    lambda: self.output_stream.write(audio_data)
                )

            except asyncio.TimeoutError:
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
                    audio_b64 = event.get("delta", "")
                    if audio_b64:
                        audio_data = base64.b64decode(audio_b64)
                        await self.audio_buffer.put(audio_data)

                elif event_type == "response.audio_transcript.delta":
                    text = event.get("delta", "")
                    print(f"\033[94m{text}\033[0m", end="", flush=True)

                elif event_type == "response.audio_transcript.done":
                    print()

                elif event_type == "response.done":
                    self.waiting_for_response = False
                    # If we're in reading mode, continue with next chunk
                    if self.reading:
                        await asyncio.sleep(0.3)  # Brief pause between chunks
                        chunk = self.get_next_chunk()
                        if chunk:
                            await self.send_text_to_read(chunk)
                        else:
                            print("\n\033[93mFinished reading the book!\033[0m")
                            self.reading = False

                elif event_type == "input_audio_buffer.speech_started":
                    # User interrupted - clear audio buffer
                    self.reading = False  # Pause reading mode
                    while not self.audio_buffer.empty():
                        try:
                            self.audio_buffer.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    # Cancel any pending response
                    await self.ws.send(json.dumps({"type": "response.cancel"}))

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    text = event.get("transcript", "")
                    if text:
                        print(f"\033[92mYou: {text}\033[0m")
                        text_lower = text.lower().strip()

                        # Check for exit
                        for phrase in EXIT_PHRASES:
                            if phrase in text_lower:
                                print("\n\033[93mGoodbye!\033[0m")
                                self.running = False
                                return

                        # Check for resume reading
                        for phrase in RESUME_PHRASES:
                            if phrase in text_lower:
                                print("\033[93mResuming reading...\033[0m")
                                self.reading = True
                                # Update context and continue
                                await self.ws.send(json.dumps({
                                    "type": "session.update",
                                    "session": {
                                        "instructions": self.build_system_prompt()
                                    }
                                }))
                                chunk = self.get_next_chunk()
                                if chunk:
                                    await self.send_text_to_read(chunk)
                                break

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

            await asyncio.gather(
                self.reading_loop(),
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
    parser = argparse.ArgumentParser(description="Read epub books aloud with voice Q&A")
    parser.add_argument("epub_file", nargs="?", help="Path to epub file")
    parser.add_argument("--voice", "-v", default="onyx",
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                        help="Voice to use (default: onyx)")
    parser.add_argument("--input", "-i", type=int, help="Input device index")
    parser.add_argument("--output", "-o", type=int, help="Output device index")
    parser.add_argument("--list-devices", "-l", action="store_true",
                        help="List available audio devices")
    parser.add_argument("--chapter", "-c", type=int, default=0,
                        help="Start from chapter number (0-indexed)")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    if not args.epub_file:
        parser.print_help()
        sys.exit(1)

    if not Path(args.epub_file).exists():
        print(f"Error: File not found: {args.epub_file}")
        sys.exit(1)

    reader = BookReader(args.epub_file, args.voice, args.input, args.output)
    reader.current_chapter = args.chapter
    asyncio.run(reader.run())


if __name__ == "__main__":
    main()
