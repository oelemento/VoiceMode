#!/usr/bin/env python3.11
"""
Claude Code Daemon - A long-running daemon that keeps Claude Code alive
with full conversation memory, controllable via Unix socket.

Protocol (NDJSON):
Client -> Daemon:
    {"type": "query", "prompt": "..."}
    {"type": "status"}
    {"type": "reset"}
    {"type": "shutdown"}

Daemon -> Client:
    {"type": "text", "content": "..."}
    {"type": "tool_use", "tool": "...", "input": {...}}
    {"type": "tool_result", "tool": "...", "success": true/false}
    {"type": "done", "cost_usd": 0.0123}
    {"type": "error", "message": "..."}
    {"type": "busy"}
    {"type": "status", "state": "ready/busy", "queries": N, "session_cost_usd": X}
"""

import asyncio
import json
import os
import signal
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Any
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Socket path - can be overridden with CLAUDE_DAEMON_SOCK env var
SOCKET_PATH = os.environ.get('CLAUDE_DAEMON_SOCK', '/tmp/claude_daemon.sock')
PID_FILE = os.environ.get('CLAUDE_DAEMON_PID', '/tmp/claude_daemon.pid')


@dataclass
class QueryRequest:
    """A query to be processed by the SDK worker."""
    prompt: str
    response_callback: Callable[[dict], Any]


@dataclass
class ResetRequest:
    """Request to reset the session."""
    response_callback: Callable[[dict], Any]


class ClaudeDaemon:
    """Main daemon class that manages the Claude session and socket server."""

    def __init__(self):
        self.is_busy = False
        self.query_count = 0
        self.total_cost_usd = 0.0
        self.started_at = datetime.now()
        self.server: Optional[asyncio.Server] = None
        self.shutdown_event = asyncio.Event()

        # Queue for SDK operations - ensures all SDK calls happen in one task
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.sdk_task: Optional[asyncio.Task] = None

    def load_claude_md(self) -> str:
        """Load CLAUDE.md from current directory if it exists."""
        claude_md_path = Path.cwd() / "CLAUDE.md"
        if claude_md_path.exists():
            try:
                content = claude_md_path.read_text()
                logger.info(f"Loaded CLAUDE.md ({len(content)} bytes)")
                return content
            except Exception as e:
                logger.warning(f"Failed to read CLAUDE.md: {e}")
        return ""

    async def sdk_worker(self):
        """
        Worker task that handles all SDK operations.
        This ensures all SDK operations happen in a single asyncio task,
        avoiding issues with anyio cancel scopes crossing task boundaries.
        """
        from claude_agent_sdk import (
            ClaudeSDKClient, ClaudeAgentOptions,
            AssistantMessage, TextBlock, ToolUseBlock,
            ToolResultBlock, ResultMessage, UserMessage
        )

        # Build system prompt with CLAUDE.md if present
        base_prompt = """You are Jarvis, a voice assistant running as a persistent daemon. You have full access to the filesystem and can execute commands.

## Response Style
**Be extremely concise.** The user is listening, not reading.
- Give short, direct answers (1-3 sentences when possible)
- Prioritize the most important information first
- Skip explanations unless asked
- No preambles like "Sure, I can help with that"
- No lists longer than 3-4 items unless specifically requested
- When summarizing schedules/tasks, mention only the key items

## User Context
- Name: Olivier Elemento
- Obsidian vault: /Users/ole2001/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/
- Weekly notes: 10 Weekly/YYYY-Wnn.md (e.g., 2026-W04.md for week 4)
"""
        claude_md = self.load_claude_md()
        if claude_md:
            system_prompt = f"{base_prompt}\n\n# Project Instructions (CLAUDE.md)\n\n{claude_md}"
        else:
            system_prompt = base_prompt

        options = ClaudeAgentOptions(
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task", "WebFetch", "WebSearch"],
            permission_mode="bypassPermissions",
            system_prompt=system_prompt,
            cwd=Path.cwd(),
        )

        while not self.shutdown_event.is_set():
            # Use the client within context manager for this session
            logger.info("Starting new SDK session")

            try:
                async with ClaudeSDKClient(options=options) as client:
                    logger.info("SDK session started")

                    while not self.shutdown_event.is_set():
                        try:
                            # Wait for a request with timeout
                            request = await asyncio.wait_for(
                                self.request_queue.get(),
                                timeout=1.0
                            )
                        except asyncio.TimeoutError:
                            continue

                        if isinstance(request, ResetRequest):
                            # Break out of this session to start a new one
                            await request.response_callback({
                                "type": "done",
                                "message": "Session reset"
                            })
                            self.query_count = 0
                            self.total_cost_usd = 0.0
                            logger.info("Session reset requested, starting new session")
                            break

                        elif isinstance(request, QueryRequest):
                            self.is_busy = True
                            self.query_count += 1
                            used_tools = False  # Track if tools were used

                            try:
                                await client.query(request.prompt)

                                async for message in client.receive_messages():
                                    if isinstance(message, AssistantMessage):
                                        for block in message.content:
                                            if isinstance(block, TextBlock):
                                                await request.response_callback({
                                                    "type": "text",
                                                    "content": block.text
                                                })
                                            elif isinstance(block, ToolUseBlock):
                                                used_tools = True
                                                await request.response_callback({
                                                    "type": "tool_use",
                                                    "tool": block.name,
                                                    "input": block.input
                                                })

                                    elif isinstance(message, UserMessage):
                                        if hasattr(message, 'content') and not isinstance(message.content, str):
                                            for block in message.content:
                                                if isinstance(block, ToolResultBlock):
                                                    await request.response_callback({
                                                        "type": "tool_result",
                                                        "tool": block.tool_use_id,
                                                        "success": not getattr(block, 'is_error', False)
                                                    })

                                    elif isinstance(message, ResultMessage):
                                        cost = getattr(message, 'total_cost_usd', 0.0) or 0.0
                                        self.total_cost_usd += cost
                                        await request.response_callback({
                                            "type": "done",
                                            "cost_usd": cost
                                        })
                                        break

                            except Exception as e:
                                error_msg = str(e)
                                logger.error(f"Query error: {error_msg}")

                                # Check for tool_use ID collision - needs session reset
                                if "tool_use" in error_msg and "unique" in error_msg:
                                    logger.warning("Tool use ID collision detected, resetting session...")
                                    await request.response_callback({
                                        "type": "error",
                                        "message": "Session reset due to tool conflict. Please retry your query."
                                    })
                                    self.is_busy = False
                                    break  # Break inner loop to start fresh SDK session
                                else:
                                    import traceback
                                    traceback.print_exc()
                                    await request.response_callback({
                                        "type": "error",
                                        "message": error_msg
                                    })
                            finally:
                                self.is_busy = False

            except Exception as e:
                logger.error(f"SDK session error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)  # Brief pause before retrying

        logger.info("SDK worker shutting down")

    async def send_response(self, writer: asyncio.StreamWriter, data: dict):
        """Send a JSON response to the client."""
        try:
            line = json.dumps(data) + '\n'
            writer.write(line.encode('utf-8'))
            await writer.drain()
        except Exception as e:
            logger.error(f"Failed to send response: {e}")

    async def handle_query(self, writer: asyncio.StreamWriter, prompt: str):
        """Handle a query request from a client."""
        if self.is_busy:
            await self.send_response(writer, {"type": "busy"})
            return

        # Create a response event and callback
        done_event = asyncio.Event()

        async def response_callback(data: dict):
            await self.send_response(writer, data)
            if data.get('type') in ('done', 'error'):
                done_event.set()

        # Submit the query to the worker
        await self.request_queue.put(QueryRequest(
            prompt=prompt,
            response_callback=response_callback
        ))

        # Wait for completion
        await done_event.wait()

    async def handle_status(self, writer: asyncio.StreamWriter):
        """Handle a status request."""
        await self.send_response(writer, {
            "type": "status",
            "state": "busy" if self.is_busy else "ready",
            "queries": self.query_count,
            "session_cost_usd": self.total_cost_usd,
            "uptime_seconds": (datetime.now() - self.started_at).total_seconds()
        })

    async def handle_reset(self, writer: asyncio.StreamWriter):
        """Reset the conversation - signal the worker to restart."""
        if self.is_busy:
            await self.send_response(writer, {
                "type": "error",
                "message": "Cannot reset while busy"
            })
            return

        done_event = asyncio.Event()

        async def response_callback(data: dict):
            await self.send_response(writer, data)
            done_event.set()

        await self.request_queue.put(ResetRequest(
            response_callback=response_callback
        ))

        await done_event.wait()

    async def handle_shutdown(self, writer: asyncio.StreamWriter):
        """Handle graceful shutdown request."""
        await self.send_response(writer, {
            "type": "done",
            "message": "Shutting down"
        })
        self.shutdown_event.set()

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a connected client."""
        addr = writer.get_extra_info('peername') or 'unix-socket'
        logger.info(f"Client connected: {addr}")

        try:
            while True:
                data = await reader.readline()
                if not data:
                    break

                try:
                    msg = json.loads(data.decode('utf-8').strip())
                except json.JSONDecodeError as e:
                    await self.send_response(writer, {
                        "type": "error",
                        "message": f"Invalid JSON: {e}"
                    })
                    continue

                msg_type = msg.get('type', '')

                if msg_type == 'query':
                    prompt = msg.get('prompt', '')
                    if prompt:
                        await self.handle_query(writer, prompt)
                    else:
                        await self.send_response(writer, {
                            "type": "error",
                            "message": "Missing prompt"
                        })

                elif msg_type == 'status':
                    await self.handle_status(writer)

                elif msg_type == 'reset':
                    await self.handle_reset(writer)

                elif msg_type == 'shutdown':
                    await self.handle_shutdown(writer)
                    break

                else:
                    await self.send_response(writer, {
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    })

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except:
                pass
            logger.info(f"Client disconnected: {addr}")

    async def start_server(self):
        """Start the Unix socket server."""
        # Remove existing socket file
        socket_path = Path(SOCKET_PATH)
        if socket_path.exists():
            socket_path.unlink()

        # Start the SDK worker task
        self.sdk_task = asyncio.create_task(self.sdk_worker())

        # Create the server
        self.server = await asyncio.start_unix_server(
            self.handle_client,
            path=SOCKET_PATH
        )

        # Set socket permissions (readable/writable by owner)
        os.chmod(SOCKET_PATH, 0o600)

        logger.info(f"Daemon listening on {SOCKET_PATH}")

        # Write PID file
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))

        async with self.server:
            # Wait for shutdown signal
            await self.shutdown_event.wait()

        # Cleanup
        if self.sdk_task:
            self.sdk_task.cancel()
            try:
                await self.sdk_task
            except asyncio.CancelledError:
                pass

        socket_path.unlink(missing_ok=True)
        Path(PID_FILE).unlink(missing_ok=True)
        logger.info("Daemon stopped")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        def handle_signal(signum):
            logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))


async def main():
    """Main entry point."""
    # Check SDK availability
    try:
        from claude_agent_sdk import ClaudeSDKClient
        logger.info("Claude SDK available")
    except ImportError as e:
        logger.error(f"Failed to import claude_agent_sdk: {e}")
        logger.error("Install with: pip install claude-agent-sdk")
        sys.exit(1)

    daemon = ClaudeDaemon()

    # Setup signal handlers
    daemon.setup_signal_handlers()

    # Start the server
    await daemon.start_server()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted")
