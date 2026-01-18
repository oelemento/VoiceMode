#!/usr/bin/env python3
"""Read epub books aloud with voice Q&A using OpenAI Realtime API."""

import asyncio
import base64
import json
import os
import re
import sys
import select
import termios
import tty
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

# Navigation command pattern (AI outputs this to navigate)
NAVIGATE_PATTERN = re.compile(r'\[NAVIGATE:\s*(.+?)\]', re.IGNORECASE)


def clean_text(text: str) -> str:
    """Clean extracted text from common EPUB artifacts."""
    # Remove page numbers (standalone numbers or "Page X")
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\nPage\s+\d+\n', '\n', text, flags=re.IGNORECASE)

    # Remove sequences of numbers (like 1 2 3 4 5... from TOC or index)
    text = re.sub(r'(\d+\s+){5,}', '', text)

    # Remove footnote markers like [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)

    # Remove standalone numbers on lines (page numbers)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)

    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove common header/footer artifacts
    text = re.sub(r'\n(Contents|Table of Contents|Index)\n', '\n', text, flags=re.IGNORECASE)

    return text.strip()


def extract_text_from_epub(epub_path: str) -> list[dict]:
    """Extract chapters from epub file."""
    book = epub.read_epub(epub_path)
    chapters = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')

            # Remove script, style, and nav elements
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()

            # Get title if available
            title = None
            title_tag = soup.find(['h1', 'h2', 'title'])
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            text = clean_text(text)

            if text and len(text) > 100:  # Skip very short sections
                chapters.append({
                    'title': title or f"Section {len(chapters) + 1}",
                    'text': text
                })

    return chapters


def clean_pdf_text(text: str) -> str:
    """Clean PDF-extracted text from common artifacts."""
    # Remove markdown image references
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
    # Clean up table remnants (pipes)
    text = re.sub(r'\|+', ' ', text)
    return text.strip()


def split_markdown_into_articles(md_text: str) -> list[dict]:
    """Split markdown text into articles based on # headings."""
    articles = []

    # Try splitting on H1 headings first
    h1_pattern = re.compile(r'^# (.+)$', re.MULTILINE)
    h1_matches = list(h1_pattern.finditer(md_text))

    if len(h1_matches) >= 2:
        for i, match in enumerate(h1_matches):
            title = match.group(1).strip()
            start = match.end()
            end = h1_matches[i + 1].start() if i + 1 < len(h1_matches) else len(md_text)
            text = md_text[start:end].strip()
            if text and len(text) > 100:
                articles.append({'title': title, 'text': clean_pdf_text(text)})

    # Fallback: try H2 headings
    if not articles:
        h2_pattern = re.compile(r'^## (.+)$', re.MULTILINE)
        h2_matches = list(h2_pattern.finditer(md_text))
        if len(h2_matches) >= 2:
            for i, match in enumerate(h2_matches):
                title = match.group(1).strip()
                start = match.end()
                end = h2_matches[i + 1].start() if i + 1 < len(h2_matches) else len(md_text)
                text = md_text[start:end].strip()
                if text and len(text) > 100:
                    articles.append({'title': title, 'text': clean_pdf_text(text)})

    # Final fallback: treat as single document
    if not articles:
        articles.append({
            'title': 'Document',
            'text': clean_pdf_text(md_text)
        })

    return articles


def detect_articles_with_gemini(full_text: str) -> list[str]:
    """Use Gemini CLI to detect article titles from PDF text.

    Samples multiple sections of the document for better coverage.
    Returns list of article titles found in the text.
    """
    import subprocess
    import shutil

    # Check if gemini CLI is available
    if not shutil.which('gemini'):
        return []

    # Sample from different parts of the document for better coverage
    text_len = len(full_text)
    samples = []

    # Take samples from beginning, middle sections, and end
    chunk_size = 10000
    if text_len > chunk_size * 5:
        # Long document: sample 5 sections
        positions = [0, text_len // 4, text_len // 2, 3 * text_len // 4, text_len - chunk_size]
        for pos in positions:
            samples.append(full_text[pos:pos + chunk_size])
        sample_text = '\n\n---\n\n'.join(samples)
    else:
        # Short document: use all of it
        sample_text = full_text[:50000]

    prompt = """Analyze this magazine/newspaper text and list ALL article titles you can find.
Article titles are typically:
- Short headlines (under 15 words) that introduce a story
- Often clever, punny, or attention-grabbing
- NOT section headers like "Business", "Politics", "United States"
- NOT navigation text like "Share", "Save", "Listen to this story"

Return ONLY the article titles, one per line. No numbering, no quotes, no explanations.

Text samples from the document:
""" + sample_text

    try:
        result = subprocess.run(
            ['gemini', '-o', 'text', prompt],
            capture_output=True,
            text=True,
            timeout=90
        )
        if result.returncode == 0 and result.stdout.strip():
            titles = [line.strip() for line in result.stdout.strip().split('\n')
                     if line.strip() and len(line.strip()) > 5 and len(line.strip()) < 100]
            # Deduplicate while preserving order
            seen = set()
            unique_titles = []
            for t in titles:
                t_lower = t.lower()
                if t_lower not in seen:
                    seen.add(t_lower)
                    unique_titles.append(t)
            return unique_titles
    except Exception as e:
        print(f"Gemini error: {e}")

    return []


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract articles/sections from PDF file.

    Uses Gemini CLI for article detection if available.
    Falls back to heuristic extraction otherwise.
    """
    import pymupdf

    doc = pymupdf.open(pdf_path)

    # Extract all text first
    all_pages_text = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text().strip()
        if text:
            all_pages_text.append((page_num, text))

    full_text = '\n\n'.join(text for _, text in all_pages_text)

    # Try Gemini-based article detection
    print("Detecting articles with Gemini...")
    article_titles = detect_articles_with_gemini(full_text)

    if article_titles and len(article_titles) >= 3:
        print(f"Gemini found {len(article_titles)} articles")
        # Split text based on detected titles
        articles = []
        for title in article_titles:
            # Find where this title appears in the text
            title_lower = title.lower()
            idx = full_text.lower().find(title_lower)
            if idx != -1:
                # Find the next title to determine end of this article
                end_idx = len(full_text)
                for next_title in article_titles:
                    if next_title != title:
                        next_idx = full_text.lower().find(next_title.lower(), idx + len(title))
                        if next_idx != -1 and next_idx < end_idx:
                            end_idx = next_idx

                article_text = full_text[idx:end_idx].strip()
                if len(article_text) > 200:
                    articles.append({
                        'title': title,
                        'text': clean_pdf_text(article_text)
                    })

        if articles:
            doc.close()
            return articles

    # Fallback: Try pymupdf4llm for better layout handling
    try:
        import pymupdf4llm
        md_text = pymupdf4llm.to_markdown(doc, show_progress=False)
        doc.close()
        return split_markdown_into_articles(md_text)
    except Exception:
        pass  # Fall through to basic extraction

    # Basic extraction: group pages into sections based on content
    # For magazines, look for article titles (short capitalized lines followed by body text)
    articles = []
    current_article = {'title': None, 'text': [], 'start_page': 0}

    # Skip words that indicate navigation/UI elements
    skip_patterns = ['menu', 'share', 'save', 'insider', 'advertisement', 'photograph',
                     'listen to', 'for you', 'weekly edition', '●', '0:00',
                     'get the economist', 'ios or android', 'united states', 'china',
                     'business', 'finance', 'europe', 'middle east', 'americas',
                     'the world this week', 'min read', 'back to top', 'world in brief',
                     'reuse this content', 'sign up', 'from the january', 'from the february',
                     'from the march', 'from the april', 'from the may', 'from the june',
                     'from the july', 'from the august', 'from the september', 'from the october',
                     'from the november', 'from the december', 'politics', 'science']

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text().strip()

        if not text or len(text) < 100:
            continue

        lines = text.split('\n')
        page_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip navigation/UI elements
            if any(skip in line.lower() for skip in skip_patterns):
                continue

            # Detect potential article title:
            # - Short line (5-80 chars)
            # - Contains at least 2 words
            # - Followed by longer content
            is_potential_title = (
                5 < len(line) < 80 and
                len(line.split()) >= 2 and
                not line.endswith('.') and  # Titles usually don't end with period
                line[0].isupper()  # Starts with capital
            )

            if is_potential_title and len(page_content) == 0:
                # This might be a new article title
                # Save previous article if it has enough content
                if current_article['title'] and len('\n'.join(current_article['text'])) > 500:
                    articles.append({
                        'title': f"{current_article['title']} (p.{current_article['start_page']+1})",
                        'text': clean_pdf_text('\n'.join(current_article['text']))
                    })
                current_article = {'title': line, 'text': [], 'start_page': page_num}
            else:
                page_content.append(line)

        if page_content:
            current_article['text'].extend(page_content)

    # Add last article
    if current_article['title'] and len('\n'.join(current_article['text'])) > 500:
        articles.append({
            'title': f"{current_article['title']} (p.{current_article['start_page']+1})",
            'text': clean_pdf_text('\n'.join(current_article['text']))
        })

    doc.close()

    # Deduplicate articles with similar titles (keep longest content)
    seen_titles = {}
    for article in articles:
        # Normalize title for comparison (remove page numbers)
        base_title = re.sub(r'\s*\(p\.\d+\)$', '', article['title']).strip().lower()
        if base_title in seen_titles:
            # Keep the one with more content
            if len(article['text']) > len(seen_titles[base_title]['text']):
                seen_titles[base_title] = article
        else:
            seen_titles[base_title] = article
    articles = list(seen_titles.values())

    # If too few articles found, fall back to page-based chunking
    if len(articles) < 3:
        doc = pymupdf.open(pdf_path)
        articles = []
        # Group pages into chunks of ~5 pages
        chunk_size = 5
        for start_page in range(0, doc.page_count, chunk_size):
            end_page = min(start_page + chunk_size, doc.page_count)
            chunk_text = '\n'.join(doc[i].get_text() for i in range(start_page, end_page))
            if len(chunk_text) > 200:
                # Try to find a title from first few lines
                first_lines = [l.strip() for l in chunk_text.split('\n')[:10] if l.strip()]
                title = next((l for l in first_lines if 10 < len(l) < 80), f"Pages {start_page+1}-{end_page}")
                articles.append({
                    'title': title,
                    'text': clean_pdf_text(chunk_text)
                })
        doc.close()

    return articles


class BookReader:
    def __init__(self, file_path: str, voice: str = "onyx",
                 input_device: int = None, output_device: int = None):
        self.file_path = Path(file_path)
        self.voice = voice
        self.input_device = input_device
        self.output_device = output_device
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Load document based on file type
        print(f"Loading: {self.file_path.name}")
        suffix = self.file_path.suffix.lower()
        if suffix == '.pdf':
            self.chapters = extract_text_from_pdf(str(file_path))
        elif suffix == '.epub':
            self.chapters = extract_text_from_epub(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .epub or .pdf")
        print(f"Found {len(self.chapters)} articles/chapters")

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
        self.is_playing = False  # Echo suppression flag
        self.old_settings = None  # Terminal settings for keyboard input
        self.current_transcript = ""  # Accumulate AI response for navigation parsing
        self.paused = False  # User paused - don't capture mic

    def get_context_window(self) -> str:
        """Get recent text for context (last ~2000 chars)."""
        if not self.chapters:
            return ""

        chapter = self.chapters[self.current_chapter]
        start = max(0, self.current_position - 2000)
        end = min(len(chapter['text']), self.current_position + 500)

        context = chapter['text'][start:end]
        return f"[Chapter: {chapter['title']}]\n\n{context}"

    def get_chapter_list(self) -> str:
        """Get formatted list of chapters for the AI."""
        lines = []
        for i, ch in enumerate(self.chapters):
            marker = " <-- CURRENT" if i == self.current_chapter else ""
            lines.append(f"  {i}: {ch['title']}{marker}")
        return "\n".join(lines)

    def go_to_chapter(self, target: str) -> tuple[bool, str]:
        """Navigate to a chapter by number or name. Returns (success, message)."""
        target = target.strip()

        # Try as number first
        try:
            chapter_num = int(target)
            if 0 <= chapter_num < len(self.chapters):
                self.current_chapter = chapter_num
                self.current_position = 0
                return True, f"Navigated to chapter {chapter_num}: {self.chapters[chapter_num]['title']}"
            else:
                return False, f"Chapter {chapter_num} not found. Valid range: 0-{len(self.chapters)-1}"
        except ValueError:
            pass

        # Try as name (fuzzy match)
        target_lower = target.lower()
        for i, ch in enumerate(self.chapters):
            title_lower = ch['title'].lower()
            # Exact match or contains
            if target_lower == title_lower or target_lower in title_lower:
                self.current_chapter = i
                self.current_position = 0
                return True, f"Navigated to chapter {i}: {ch['title']}"

        # Try word matching
        target_words = set(target_lower.split())
        best_match = -1
        best_score = 0
        for i, ch in enumerate(self.chapters):
            title_words = set(ch['title'].lower().split())
            score = len(target_words & title_words)
            if score > best_score:
                best_score = score
                best_match = i

        if best_match >= 0 and best_score > 0:
            self.current_chapter = best_match
            self.current_position = 0
            return True, f"Navigated to chapter {best_match}: {self.chapters[best_match]['title']}"

        return False, f"Could not find chapter matching '{target}'"

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
            chunk = chapter['text'][:self.chunk_size]
            self.current_position = len(chunk)  # Fix: update position
            return f"\n\nChapter: {chapter['title']}\n\n" + chunk

        chunk = chapter['text'][self.current_position:self.current_position + self.chunk_size]
        self.current_position += len(chunk)
        return chunk

    def build_system_prompt(self) -> str:
        return f"""You are reading "{self.file_path.stem}" aloud to the user.

AVAILABLE CHAPTERS:
{self.get_chapter_list()}

Your modes:
1. READING MODE: When told to read, speak the text naturally as an audiobook narrator. Read exactly what's given - don't summarize or paraphrase.

2. Q&A MODE: When the user interrupts with a question, answer based on the book content you've read so far. Be helpful and concise.

3. NAVIGATION MODE: When the user asks to go to a specific chapter, skip ahead, go back, or navigate anywhere in the book:
   - First, say something brief like "Okay, going to chapter X" or "Jumping to the introduction"
   - Then output the navigation command EXACTLY like this: [NAVIGATE: chapter_name_or_number]
   - Examples: [NAVIGATE: 3] or [NAVIGATE: Introduction] or [NAVIGATE: Chapter One]
   - The system will handle the navigation and start reading from that chapter.

Current position context:
{self.get_context_window()}

Instructions:
- Read the text naturally with good pacing
- When user asks a question, answer it helpfully based on the book
- If user says "continue" or "keep reading", resume reading from where you left off
- If user asks to go to a chapter (e.g., "go to chapter 5", "skip to the introduction", "jump to part two"), use [NAVIGATE: target]
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
            "channels": CHANNELS,  # Mono
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

    async def keyboard_listener(self):
        """Listen for keyboard input (spacebar to pause/resume)."""
        loop = asyncio.get_event_loop()

        # Set terminal to raw mode for single key detection
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while self.running:
                # Check if input is available
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = await loop.run_in_executor(None, sys.stdin.read, 1)

                    if key == ' ':  # Spacebar
                        if self.reading and not self.paused:
                            print("\n\033[93m[PAUSED - Press SPACE to resume, or speak your question]\033[0m")
                            self.reading = False
                            self.paused = True
                            # Clear output audio buffer
                            while not self.audio_buffer.empty():
                                try:
                                    self.audio_buffer.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            # Cancel current response and clear server-side input buffer
                            await self.ws.send(json.dumps({"type": "response.cancel"}))
                            await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                            self.is_playing = False
                            self.waiting_for_response = False
                            # Wait for echoes to die down, then clear again
                            await asyncio.sleep(0.5)
                            await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                            self.paused = False  # Now allow mic input for questions
                        elif self.paused:
                            # Still settling, ignore
                            pass
                        else:
                            print("\033[93m[RESUMING...]\033[0m")
                            self.reading = True
                            # Clear any stray audio that accumulated while paused
                            await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                            # Wait briefly for any active response to finish
                            await asyncio.sleep(0.2)
                            # Only get next chunk if not already waiting
                            if not self.waiting_for_response:
                                chunk = self.get_next_chunk()
                                if chunk:
                                    await self.send_text_to_read(chunk)

                    elif key == 'q':  # Q to quit
                        print("\n\033[93mQuitting...\033[0m")
                        self.running = False
                        break
                else:
                    await asyncio.sleep(0.1)
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    async def reading_loop(self):
        """Manage the reading flow."""
        # Start with first chunk
        await asyncio.sleep(1)  # Let connection settle

        print("\n\033[93mStarting to read. Press SPACE to pause and ask questions.\033[0m")
        print("\033[93mPress Q to quit.\033[0m\n")

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

                # Only capture mic when paused for Q&A (not while reading)
                # This disables voice interruption - use SPACE to pause instead
                if not self.is_playing and not self.paused and not self.reading:
                    audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                    await self.ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64
                    }))

            except Exception as e:
                if self.running:
                    print(f"Audio capture error: {e}")
                break

    def mono_to_stereo(self, mono_data: bytes) -> bytes:
        """Convert mono audio to stereo by duplicating channels."""
        import array
        mono = array.array('h', mono_data)
        stereo = array.array('h')
        for sample in mono:
            stereo.append(sample)  # Left
            stereo.append(sample)  # Right
        return stereo.tobytes()

    async def play_audio(self):
        """Play audio chunks from the buffer."""
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                audio_data = await asyncio.wait_for(
                    self.audio_buffer.get(),
                    timeout=0.1
                )

                self.is_playing = True
                await loop.run_in_executor(
                    None,
                    lambda d=audio_data: self.output_stream.write(d)
                )

            except asyncio.TimeoutError:
                self.is_playing = False
                continue
            except Exception as e:
                self.is_playing = False
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
                    self.current_transcript += text
                    print(f"\033[94m{text}\033[0m", end="", flush=True)

                elif event_type == "response.audio_transcript.done":
                    print()
                    # Check for navigation command in the transcript
                    match = NAVIGATE_PATTERN.search(self.current_transcript)
                    if match:
                        target = match.group(1)
                        print(f"\033[93m[Navigation requested: {target}]\033[0m")
                        success, message = self.go_to_chapter(target)
                        print(f"\033[93m{message}\033[0m")
                        if success:
                            # Update session with new context
                            await self.ws.send(json.dumps({
                                "type": "session.update",
                                "session": {
                                    "instructions": self.build_system_prompt()
                                }
                            }))
                            # Set reading mode - response.done will send the first chunk
                            self.reading = True
                    self.current_transcript = ""  # Reset for next response

                elif event_type == "response.done":
                    self.waiting_for_response = False
                    # If we're in reading mode, wait for audio to finish then continue
                    if self.reading:
                        # Wait for audio buffer to drain
                        while not self.audio_buffer.empty():
                            await asyncio.sleep(0.1)
                        await asyncio.sleep(0.5)  # Brief pause between chunks

                        # Check we're still in reading mode (user might have interrupted)
                        if self.reading:
                            chunk = self.get_next_chunk()
                            if chunk:
                                await self.send_text_to_read(chunk)
                            else:
                                print("\n\033[93mFinished reading the book!\033[0m")
                                self.reading = False

                elif event_type == "input_audio_buffer.speech_started":
                    # User interrupted (or stray audio detected)
                    self.reading = False  # Pause reading mode
                    self.waiting_for_response = True  # VAD will trigger a response
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
                    msg = error.get('message', 'Unknown error')
                    # Suppress harmless cancellation errors
                    if "Cancellation failed" not in msg:
                        print(f"\033[91mError: {msg}\033[0m")

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
                self.handle_events(),
                self.keyboard_listener()
            )

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.running = False
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        # Restore terminal settings
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass
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
    parser = argparse.ArgumentParser(description="Read epub/pdf files aloud with voice Q&A")
    parser.add_argument("file", nargs="?", help="Path to epub or pdf file")
    parser.add_argument("--voice", "-v", default="ash",
                        choices=["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"],
                        help="Voice to use (default: ash)")
    parser.add_argument("--input", "-i", type=int, help="Input device index")
    parser.add_argument("--output", "-o", type=int, help="Output device index")
    parser.add_argument("--list-devices", "-l", action="store_true",
                        help="List available audio devices")
    parser.add_argument("--chapter", "-c", type=int, default=0,
                        help="Start from chapter/article number (0-indexed)")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    if not args.file:
        parser.print_help()
        sys.exit(1)

    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    reader = BookReader(args.file, args.voice, args.input, args.output)
    reader.current_chapter = args.chapter
    asyncio.run(reader.run())


if __name__ == "__main__":
    main()
