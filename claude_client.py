#!/usr/bin/env python3.11
"""
Python client library for the Claude Code Daemon.

Usage:
    from claude_client import ClaudeDaemon

    async with ClaudeDaemon() as claude:
        # Stream responses
        async for msg in claude.query("What files are here?"):
            print(msg)

        # Follow-up with full context
        async for msg in claude.query("Now explain the main one"):
            print(msg)

        # Get status
        status = await claude.status()
        print(f"Queries: {status['queries']}")

        # Reset when needed
        await claude.reset()
"""

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

# Socket path
SOCKET_PATH = os.environ.get('CLAUDE_DAEMON_SOCK', '/tmp/claude_daemon.sock')


@dataclass
class TextMessage:
    """Text response from Claude."""
    content: str
    type: str = "text"


@dataclass
class ToolUseMessage:
    """Tool use notification."""
    tool: str
    input: dict
    type: str = "tool_use"


@dataclass
class ToolResultMessage:
    """Tool result notification."""
    tool: str
    success: bool
    type: str = "tool_result"


@dataclass
class DoneMessage:
    """Query completion message."""
    cost_usd: float
    message: Optional[str] = None
    type: str = "done"


@dataclass
class ErrorMessage:
    """Error message."""
    message: str
    type: str = "error"


@dataclass
class StatusMessage:
    """Daemon status."""
    state: str
    queries: int
    session_cost_usd: float
    uptime_seconds: float
    type: str = "status"


@dataclass
class BusyMessage:
    """Daemon is busy."""
    type: str = "busy"


# Type alias for all message types
Message = TextMessage | ToolUseMessage | ToolResultMessage | DoneMessage | ErrorMessage | StatusMessage | BusyMessage


def _parse_message(data: dict) -> Message:
    """Parse a JSON response into a typed message."""
    msg_type = data.get('type', '')

    if msg_type == 'text':
        return TextMessage(content=data.get('content', ''))
    elif msg_type == 'tool_use':
        return ToolUseMessage(tool=data.get('tool', ''), input=data.get('input', {}))
    elif msg_type == 'tool_result':
        return ToolResultMessage(tool=data.get('tool', ''), success=data.get('success', False))
    elif msg_type == 'done':
        return DoneMessage(cost_usd=data.get('cost_usd', 0.0), message=data.get('message'))
    elif msg_type == 'error':
        return ErrorMessage(message=data.get('message', 'Unknown error'))
    elif msg_type == 'status':
        return StatusMessage(
            state=data.get('state', 'unknown'),
            queries=data.get('queries', 0),
            session_cost_usd=data.get('session_cost_usd', 0.0),
            uptime_seconds=data.get('uptime_seconds', 0.0)
        )
    elif msg_type == 'busy':
        return BusyMessage()
    else:
        return ErrorMessage(message=f"Unknown message type: {msg_type}")


class DaemonNotRunningError(Exception):
    """Raised when the daemon is not running."""
    pass


class DaemonBusyError(Exception):
    """Raised when the daemon is busy with another query."""
    pass


class ClaudeDaemon:
    """
    Client for interacting with the Claude Code Daemon.

    Usage:
        async with ClaudeDaemon() as claude:
            async for msg in claude.query("Hello"):
                if isinstance(msg, TextMessage):
                    print(msg.content)
    """

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _connect(self):
        """Connect to the daemon socket."""
        if self._writer is not None:
            return

        if not Path(self.socket_path).exists():
            raise DaemonNotRunningError(
                f"Daemon socket not found at {self.socket_path}. "
                "Start the daemon with: clauded start"
            )

        try:
            self._reader, self._writer = await asyncio.open_unix_connection(self.socket_path)
        except ConnectionRefusedError:
            raise DaemonNotRunningError("Daemon connection refused. It may have crashed.")
        except Exception as e:
            raise DaemonNotRunningError(f"Failed to connect to daemon: {e}")

    async def close(self):
        """Close the connection."""
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except:
                pass
            self._writer = None
            self._reader = None

    async def _send(self, msg: dict):
        """Send a message to the daemon."""
        await self._connect()
        if not self._writer:
            raise DaemonNotRunningError("Not connected to daemon")
        line = json.dumps(msg) + '\n'
        self._writer.write(line.encode('utf-8'))
        await self._writer.drain()

    async def _receive_one(self) -> Optional[Message]:
        """Receive a single message."""
        if not self._reader:
            return None

        line = await self._reader.readline()
        if not line:
            return None

        try:
            data = json.loads(line.decode().strip())
            return _parse_message(data)
        except json.JSONDecodeError:
            return None

    async def query(self, prompt: str) -> AsyncIterator[Message]:
        """
        Send a query and yield response messages.

        Args:
            prompt: The prompt to send to Claude

        Yields:
            Message objects (TextMessage, ToolUseMessage, etc.)

        Raises:
            DaemonNotRunningError: If daemon is not running
            DaemonBusyError: If daemon is busy with another query
        """
        await self._send({"type": "query", "prompt": prompt})

        while True:
            msg = await self._receive_one()
            if msg is None:
                break

            if isinstance(msg, BusyMessage):
                raise DaemonBusyError("Daemon is busy with another query")

            yield msg

            if isinstance(msg, (DoneMessage, ErrorMessage)):
                break

    async def query_text(self, prompt: str) -> str:
        """
        Send a query and return just the text response.

        Args:
            prompt: The prompt to send to Claude

        Returns:
            The concatenated text response

        Raises:
            DaemonNotRunningError: If daemon is not running
            DaemonBusyError: If daemon is busy with another query
            Exception: If an error occurred
        """
        text_parts = []

        async for msg in self.query(prompt):
            if isinstance(msg, TextMessage):
                text_parts.append(msg.content)
            elif isinstance(msg, ErrorMessage):
                raise Exception(msg.message)

        return ''.join(text_parts)

    async def status(self) -> StatusMessage:
        """
        Get daemon status.

        Returns:
            StatusMessage with daemon state info

        Raises:
            DaemonNotRunningError: If daemon is not running
        """
        await self._send({"type": "status"})
        msg = await self._receive_one()

        if isinstance(msg, StatusMessage):
            return msg
        elif isinstance(msg, ErrorMessage):
            raise Exception(msg.message)
        else:
            raise Exception("Unexpected response from daemon")

    async def reset(self) -> bool:
        """
        Reset the conversation memory.

        Returns:
            True if reset was successful

        Raises:
            DaemonNotRunningError: If daemon is not running
        """
        await self._send({"type": "reset"})
        msg = await self._receive_one()

        if isinstance(msg, DoneMessage):
            return True
        elif isinstance(msg, ErrorMessage):
            raise Exception(msg.message)
        else:
            return False

    async def shutdown(self) -> bool:
        """
        Request daemon shutdown.

        Returns:
            True if shutdown was initiated
        """
        await self._send({"type": "shutdown"})
        msg = await self._receive_one()
        return isinstance(msg, DoneMessage)


# Convenience function for simple queries
async def ask(prompt: str, socket_path: str = SOCKET_PATH) -> str:
    """
    Simple one-shot query function.

    Args:
        prompt: The prompt to send
        socket_path: Optional custom socket path

    Returns:
        The text response

    Example:
        response = await ask("What is 2 + 2?")
        print(response)
    """
    async with ClaudeDaemon(socket_path) as claude:
        return await claude.query_text(prompt)


# Example usage
if __name__ == '__main__':
    async def demo():
        print("Claude Daemon Client Demo")
        print("-" * 40)

        try:
            async with ClaudeDaemon() as claude:
                # Check status
                status = await claude.status()
                print(f"Daemon status: {status.state}")
                print(f"Session queries: {status.queries}")
                print()

                # Send a query
                print("Sending: What is 2 + 2?")
                print("Response: ", end="", flush=True)

                async for msg in claude.query("What is 2 + 2?"):
                    if isinstance(msg, TextMessage):
                        print(msg.content, end="", flush=True)
                    elif isinstance(msg, DoneMessage):
                        print(f"\n[Cost: ${msg.cost_usd:.4f}]")

                print()

                # Follow-up
                print("Sending follow-up: Multiply that by 10")
                print("Response: ", end="", flush=True)

                async for msg in claude.query("Multiply that by 10"):
                    if isinstance(msg, TextMessage):
                        print(msg.content, end="", flush=True)
                    elif isinstance(msg, DoneMessage):
                        print(f"\n[Cost: ${msg.cost_usd:.4f}]")

        except DaemonNotRunningError as e:
            print(f"Error: {e}")
            print("Start the daemon with: clauded start")

    asyncio.run(demo())
