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

---

# Jarvis

Voice interface for Claude Code - speak commands, hear responses, approve actions by voice.

## Usage

```bash
python3.11 jarvis.py [options]
```

### Modes

**Push-to-talk (default)**: Press SPACE to start recording, press SPACE again to send. Most reliable - no echo issues.

**Hands-free**: Always listening with voice activity detection. Has 1 second delay after responses to prevent echo feedback.

### Options

| Option | Description |
|--------|-------------|
| `--voice`, `-v` | TTS voice: alloy, echo, fable, onyx, nova, shimmer (default: onyx) |
| `--hands-free`, `-f` | Always listening mode (vs push-to-talk) |
| `--input`, `-i` | Input device index (microphone) |
| `--output`, `-o` | Output device index (speakers/headphones) |
| `--list-devices`, `-l` | List available audio devices |

### Examples

```bash
# Push-to-talk mode (recommended)
python3.11 jarvis.py

# Hands-free mode (always listening)
python3.11 jarvis.py --hands-free

# Use different voice
python3.11 jarvis.py --voice nova

# List audio devices
python3.11 jarvis.py --list-devices
```

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Toggle recording (press to start, press again to send) |
| `q` | Quit |
| `Ctrl+C` | Force quit |

### Voice Commands

- **Any request**: "Add error handling to the login function"
- **Navigation**: "Go to my VoiceMode project"
- **Approval**: "Yes" / "Yeah" / "Go ahead" when Claude asks for permission |
- **Deny**: "No" / "Cancel" / "Stop" |
- **Exit**: "Goodbye" / "Exit" / "Quit" / "Shut down"

### How It Works

1. Starts Claude CLI as a persistent subprocess (PTY)
2. Uses OpenAI Realtime API for speech-to-text
3. Sends transcribed speech to Claude as commands
4. Reads Claude's output and speaks it using OpenAI TTS (onyx voice)
5. Detects permission prompts (`[Y/n]`, `Allow?`, etc.) and accepts voice approval
6. Echo suppression: mic is muted while speaking + 1 second settling time

### Tips

- **Use headphones** to minimize echo in hands-free mode
- **Push-to-talk is more reliable** - press SPACE, speak, release
- In hands-free mode, wait 1 second after Jarvis speaks before responding
- Claude's code blocks are not spoken (only prose)
- Say "yes" or "go ahead" to approve file edits and commands

---

# Claude Daemon

A persistent Claude Code daemon with conversation memory and tool access, controllable via Unix socket.

## Installation

```bash
pip install claude-agent-sdk
```

## Usage

### Start the Daemon

```bash
cd /path/to/your/project
./clauded start
```

The daemon automatically loads `CLAUDE.md` from the current directory if present, including it in Claude's system prompt.

### Commands

| Command | Description |
|---------|-------------|
| `./clauded start` | Start daemon in background |
| `./clauded stop` | Graceful shutdown |
| `./clauded status` | Check if running + session info |
| `./clauded reset` | Clear conversation memory |
| `./clauded query "prompt"` | Send query, stream response |
| `./clauded interactive` | REPL mode (persistent memory) |

### Examples

```bash
# Start daemon in your project folder
cd ~/Projects/my-app
./clauded start

# Send a query
./clauded query "What files are in this directory?"

# Follow-up (remembers context)
./clauded query "Explain the main one"

# Check status
./clauded status

# Interactive REPL
./clauded interactive

# Reset conversation memory
./clauded reset

# Stop daemon
./clauded stop
```

### CLAUDE.md Support

If a `CLAUDE.md` file exists in the working directory when the daemon starts, its contents are automatically included in Claude's system prompt. This lets you provide project-specific instructions.

```bash
# Your project structure
my-project/
  CLAUDE.md      # Project instructions (loaded by daemon)
  src/
  ...

# Start daemon from project root
cd my-project
./clauded start  # Logs: "Loaded CLAUDE.md (X bytes)"
```

### Python Client

```python
from claude_client import ClaudeDaemon, TextMessage

async with ClaudeDaemon() as claude:
    # Stream responses
    async for msg in claude.query("What files are here?"):
        if isinstance(msg, TextMessage):
            print(msg.content, end="")

    # Follow-up with context
    response = await claude.query_text("Explain the first one")
```

---

# Jarvis Daemon

Voice interface using Claude Daemon for persistent memory and tool access.

## Usage

```bash
# First, start the daemon
./clauded start

# Then start Jarvis
python3.11 jarvis_daemon.py [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--voice`, `-v` | TTS voice: alloy, echo, fable, onyx, nova, shimmer (default: onyx) |
| `--input`, `-i` | Input device index (microphone) |
| `--output`, `-o` | Output device index (speakers/headphones) |
| `--list-devices`, `-l` | List available audio devices |

### Examples

```bash
# Default voice
python3.11 jarvis_daemon.py

# Use different voice
python3.11 jarvis_daemon.py -v nova

# List audio devices
python3.11 jarvis_daemon.py --list-devices
```

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Start/stop recording |
| `q` | Quit |

### Voice Commands

- **Exit**: "Goodbye" / "Exit" / "Quit" / "Shut down"

### Features

- **Persistent memory**: Conversation continues across Jarvis restarts (daemon keeps context)
- **Tool access**: Claude can read/write files, run commands, search the web
- **Streaming TTS**: Speaks sentences as they arrive for low latency
- **Tool visibility**: Shows when Claude uses tools

### How It Works

1. Connects to Claude Daemon (must be running)
2. Uses OpenAI Realtime API for speech-to-text (Whisper)
3. Sends transcribed speech to daemon
4. Streams response text to OpenAI TTS
5. Plays audio as it arrives

### Comparison: jarvis.py vs jarvis_daemon.py

| Feature | jarvis.py | jarvis_daemon.py |
|---------|-----------|------------------|
| Backend | Claude CLI (PTY) | Claude Daemon |
| Memory | Per-session | Persistent (across restarts) |
| CLAUDE.md | No | Yes (via daemon) |
| Tool access | Yes | Yes |
| Permission prompts | Voice approval | Bypassed (daemon mode) |
| Hands-free mode | Yes | No (push-to-talk only) |
