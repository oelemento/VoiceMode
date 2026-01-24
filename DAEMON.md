# Claude Code Daemon

A long-running daemon that keeps Claude Code alive with full conversation memory, controllable by any external program via Unix socket.

## Overview

The Claude Code Daemon solves a common problem: Claude Code sessions are ephemeral, losing context between invocations. This daemon maintains a persistent Claude session, allowing you to:

- Send multiple queries that build on each other
- Integrate Claude into scripts, automation, and other programs
- Control Claude from any language that can write to a Unix socket

```
┌─────────────────┐                      ┌──────────────────────┐
│  Any program    │     Unix Socket      │  claude_daemon.py    │
│  - Python       │◄───────────────────►│                      │
│  - Shell        │     JSON messages    │  ┌────────────────┐  │
│  - Node.js      │                      │  │ Claude Agent   │  │
│  - etc.         │                      │  │ SDK Session    │  │
└─────────────────┘                      │  │ (persistent)   │  │
                                         │  └────────────────┘  │
┌─────────────────┐                      │                      │
│  clauded CLI    │◄───────────────────►│  Memory persists     │
│  (control tool) │                      │  across all queries  │
└─────────────────┘                      └──────────────────────┘
```

## Installation

```bash
pip install claude-agent-sdk
```

Ensure the three daemon files are in your project:
- `claude_daemon.py` - Main daemon process
- `clauded` - CLI control script (must be executable)
- `claude_client.py` - Python client library

## Quick Start

```bash
# Start the daemon
./clauded start

# Send a query
./clauded query "What files are in this directory?"

# Follow-up (remembers context)
./clauded query "Which one is the largest?"

# Check status
./clauded status

# Clear conversation memory
./clauded reset

# Stop the daemon
./clauded stop
```

## CLI Reference

### `clauded start`

Start the daemon in background. Creates a Unix socket at `/tmp/claude_daemon.sock`.

```bash
./clauded start
# Output: Daemon started (socket: /tmp/claude_daemon.sock)
```

### `clauded stop`

Gracefully stop the daemon.

```bash
./clauded stop
# Output: Daemon stopped
```

### `clauded status`

Check if the daemon is running and view session info.

```bash
./clauded status
# Output:
# Status: ready
# Queries: 5
# Session cost: $0.0523
# Uptime: 0h 15m 32s
# Socket: /tmp/claude_daemon.sock
```

### `clauded query "prompt"`

Send a query and stream the response to stdout. Tool usage is shown on stderr.

```bash
./clauded query "Read the first 10 lines of README.md"
# Output streams as Claude responds
```

### `clauded reset`

Clear conversation memory and start fresh. Useful when switching contexts.

```bash
./clauded reset
# Output: Session reset - conversation memory cleared
```

### `clauded interactive`

Enter an interactive REPL mode with persistent context.

```bash
./clauded interactive
# Claude Daemon Interactive Mode
# Type 'quit' or 'exit' to leave, 'reset' to clear memory
# ----------------------------------------
#
# > What is 2 + 2?
# 2 + 2 = 4
#
# > Multiply that by 10
# 4 × 10 = 40
```

## Python Client

The `claude_client.py` module provides a Python interface to the daemon.

### Basic Usage

```python
import asyncio
from claude_client import ClaudeDaemon, TextMessage

async def main():
    async with ClaudeDaemon() as claude:
        # Simple query - returns just the text
        response = await claude.query_text("What is Python?")
        print(response)

        # Follow-up with context
        response = await claude.query_text("Give me an example")
        print(response)

asyncio.run(main())
```

### Streaming Responses

```python
async def main():
    async with ClaudeDaemon() as claude:
        async for msg in claude.query("Explain recursion"):
            if isinstance(msg, TextMessage):
                print(msg.content, end="", flush=True)
```

### Message Types

```python
from claude_client import (
    TextMessage,      # Text response from Claude
    ToolUseMessage,   # Claude is using a tool
    ToolResultMessage,# Tool execution result
    DoneMessage,      # Query complete (includes cost)
    ErrorMessage,     # An error occurred
    StatusMessage,    # Daemon status info
    BusyMessage,      # Daemon is busy with another query
)
```

### Status and Reset

```python
async with ClaudeDaemon() as claude:
    # Check status
    status = await claude.status()
    print(f"State: {status.state}")
    print(f"Queries: {status.queries}")
    print(f"Cost: ${status.session_cost_usd:.4f}")

    # Reset conversation
    await claude.reset()
```

### One-Shot Query Function

```python
from claude_client import ask

# Simple one-liner for quick queries
response = await ask("What time is it in Tokyo?")
```

### Error Handling

```python
from claude_client import DaemonNotRunningError, DaemonBusyError

try:
    async with ClaudeDaemon() as claude:
        response = await claude.query_text("Hello")
except DaemonNotRunningError:
    print("Start the daemon first: ./clauded start")
except DaemonBusyError:
    print("Daemon is busy, try again later")
```

## Socket Protocol

For integrating from other languages, the daemon uses a simple JSON protocol over Unix socket.

**Socket Path:** `/tmp/claude_daemon.sock` (or `$CLAUDE_DAEMON_SOCK`)

**Format:** NDJSON (newline-delimited JSON)

### Client → Daemon Messages

```json
{"type": "query", "prompt": "Your question here"}
{"type": "status"}
{"type": "reset"}
{"type": "shutdown"}
```

### Daemon → Client Messages

```json
{"type": "text", "content": "Response text..."}
{"type": "tool_use", "tool": "Read", "input": {"file_path": "/path/to/file"}}
{"type": "tool_result", "tool": "tool_use_id", "success": true}
{"type": "done", "cost_usd": 0.0123}
{"type": "error", "message": "Error description"}
{"type": "busy"}
{"type": "status", "state": "ready", "queries": 5, "session_cost_usd": 0.05, "uptime_seconds": 3600}
```

### Example: Shell Integration

```bash
# Using netcat/socat
echo '{"type":"query","prompt":"Hello"}' | nc -U /tmp/claude_daemon.sock

# Using socat for full interaction
socat - UNIX-CONNECT:/tmp/claude_daemon.sock
```

### Example: Node.js Client

```javascript
const net = require('net');

const client = net.createConnection('/tmp/claude_daemon.sock');

client.on('data', (data) => {
    const lines = data.toString().split('\n').filter(Boolean);
    for (const line of lines) {
        const msg = JSON.parse(line);
        if (msg.type === 'text') {
            process.stdout.write(msg.content);
        } else if (msg.type === 'done') {
            console.log(`\nCost: $${msg.cost_usd}`);
            client.end();
        }
    }
});

client.write(JSON.stringify({type: 'query', prompt: 'Hello'}) + '\n');
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_DAEMON_SOCK` | `/tmp/claude_daemon.sock` | Socket file path |
| `CLAUDE_DAEMON_PID` | `/tmp/claude_daemon.pid` | PID file path |

### Tool Access

The daemon is configured with full tool access:

- **Read** - Read any file
- **Write** - Create/overwrite files
- **Edit** - Edit existing files
- **Bash** - Execute shell commands
- **Glob** - Find files by pattern
- **Grep** - Search file contents
- **Task** - Spawn sub-agents
- **WebFetch** - Fetch web content
- **WebSearch** - Search the web

All tools are auto-approved (`permission_mode="bypassPermissions"`).

### Customizing

Edit `claude_daemon.py` to modify the SDK options:

```python
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Edit", "Bash", ...],
    permission_mode="bypassPermissions",  # or "default", "acceptEdits"
    system_prompt="Your custom system prompt",
    cwd=Path("/your/working/directory"),
    max_turns=10,  # Limit agent turns
    max_budget_usd=5.0,  # Cost limit
)
```

## Logs

Daemon logs are written to `/tmp/claude_daemon.log` when started via `clauded start`.

```bash
tail -f /tmp/claude_daemon.log
```

## Troubleshooting

### "Daemon not running"

```bash
./clauded start
```

### "Connection refused"

The daemon may have crashed. Check logs and restart:

```bash
cat /tmp/claude_daemon.log
./clauded start
```

### "Daemon is busy"

Only one query can run at a time. Wait for the current query to complete.

### Socket permission denied

The socket is created with mode 0600 (owner only). Run commands as the same user who started the daemon.

### Stale socket file

If the daemon crashed without cleanup:

```bash
rm /tmp/claude_daemon.sock /tmp/claude_daemon.pid
./clauded start
```
