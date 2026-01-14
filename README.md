# VoiceMode

Voice conversations for learning and reviewing markdown notes using OpenAI's Realtime API.

## Installation

```bash
pip install websockets pyaudio
```

Requires Python 3.9+ and an OpenAI API key with access to `gpt-4o-realtime-preview`.

```bash
export OPENAI_API_KEY="your-key-here"
```

### Use from anywhere

Add an alias to your shell config (`~/.zshrc` or `~/.bashrc`):

```bash
alias voicemode="python3.11 /path/to/VoiceMode/voicemode.py"
```

Then reload: `source ~/.zshrc`

Now run from any folder:

```bash
voicemode ~/Notes/my-note.md
```

## Usage

```bash
python voicemode.py <markdown_file> [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--voice`, `-v` | Voice: alloy, echo, fable, onyx, nova, shimmer (default: alloy) |
| `--input`, `-i` | Input device index (microphone) |
| `--output`, `-o` | Output device index (speakers/headphones) |
| `--list-devices`, `-l` | List available audio devices |

### Examples

```bash
# Basic usage
python voicemode.py my-notes.md

# List audio devices
python voicemode.py --list-devices

# Use specific devices
python voicemode.py my-notes.md --input 4 --output 2

# Use a different voice
python voicemode.py my-notes.md --voice nova
```

## Voice Commands

Say any of these to exit:
- "bye bye", "ok bye bye"
- "shut down", "shutdown"
- "exit", "quit", "stop"
- "goodbye"
- "end session"

## Tips

- **Use headphones** to prevent echo feedback
- **External microphone** works best for clear input
- The AI waits 1 second of silence before responding
- Interrupt anytime by speaking over the AI
- Press `Ctrl+C` to force quit

## How It Works

1. Loads your markdown file as system context
2. Connects to OpenAI Realtime API via WebSocket
3. Streams your voice to OpenAI with server-side voice activity detection
4. Streams AI audio responses back to your speakers
5. Supports natural interruptions

## Transcripts

Session transcripts are saved to the `transcripts/` folder.

---

# ReadBook

Read epub books aloud with voice Q&A - interrupt anytime to ask questions.

## Usage

```bash
python readbook.py <epub_file> [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--voice`, `-v` | Voice: alloy, echo, fable, onyx, nova, shimmer (default: onyx) |
| `--input`, `-i` | Input device index (microphone) |
| `--output`, `-o` | Output device index (speakers/headphones) |
| `--chapter`, `-c` | Start from chapter number (0-indexed) |
| `--list-devices`, `-l` | List available audio devices |

### Examples

```bash
# Read a book
python readbook.py "My Book.epub"

# Start from chapter 3
python readbook.py book.epub --chapter 3

# Use specific voice
python readbook.py book.epub --voice nova
```

### Voice Commands

While reading:
- **Interrupt anytime** - Just speak to pause and ask a question
- **"Continue"** / **"Keep reading"** - Resume reading
- **"Stop reading"** / **"Exit"** - End session

### Example Questions

- "Who is this character?"
- "What did he mean by that?"
- "Summarize what just happened"
- "Why is this important?"
