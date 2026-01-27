#!/usr/bin/env python3.11
"""Jarvis Status Window - Small floating indicator showing current state.

Uses multiprocessing to run tkinter in a separate process, avoiding
macOS main-thread requirements that conflict with audio processing.
"""

import multiprocessing as mp
import tkinter as tk
from typing import Optional


def _run_status_window(state_queue: mp.Queue, shutdown_event):
    """Run the status window in a separate process."""
    root = tk.Tk()
    root.title("Jarvis")

    # State colors and labels
    STATES = {
        "off": ("#dc3545", "Daemon Off"),      # Red
        "idle": ("#6c757d", "Idle"),           # Gray
        "activated": ("#ffc107", "Activated"), # Yellow
        "recording": ("#28a745", "Recording"), # Green
        "processing": ("#007bff", "Processing"), # Blue
        "speaking": ("#9b59b6", "Speaking"),   # Purple
    }

    current_state = "off"

    # Window properties
    root.overrideredirect(True)  # No title bar
    root.attributes("-topmost", True)  # Always on top
    root.configure(bg="#1a1a1a")

    # Size and position (top-right corner)
    window_width = 220
    window_height = 250
    margin = 20

    screen_width = root.winfo_screenwidth()
    x = screen_width - window_width - margin
    y = margin

    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Create frame for content (vertical layout)
    frame = tk.Frame(root, bg="#1a1a1a")
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Large colored indicator circle
    circle_size = 150
    canvas = tk.Canvas(frame, width=circle_size, height=circle_size, bg="#1a1a1a", highlightthickness=0)
    canvas.pack(pady=(0, 15))

    color = STATES[current_state][0]
    indicator = canvas.create_oval(5, 5, circle_size-5, circle_size-5, fill=color, outline="")

    # Status label below circle
    text = STATES[current_state][1]
    label = tk.Label(frame, text=text, fg="white", bg="#1a1a1a", font=("SF Pro", 18, "bold"))
    label.pack()

    def poll_updates():
        nonlocal current_state

        # Check for shutdown
        if shutdown_event.is_set():
            root.destroy()
            return

        # Process state updates
        try:
            while True:
                state = state_queue.get_nowait()
                if state in STATES:
                    current_state = state
                    color, text = STATES[state]
                    canvas.itemconfig(indicator, fill=color)
                    label.config(text=text)
        except:
            pass

        root.after(50, poll_updates)

    poll_updates()
    root.mainloop()


class JarvisStatusWindow:
    """Status window that runs in a separate process."""

    def __init__(self):
        self._process: Optional[mp.Process] = None
        self._state_queue: Optional[mp.Queue] = None
        self._shutdown_event: Optional[mp.Event] = None

    def start(self):
        """Start the status window process."""
        self._state_queue = mp.Queue()
        self._shutdown_event = mp.Event()
        self._process = mp.Process(
            target=_run_status_window,
            args=(self._state_queue, self._shutdown_event),
            daemon=True
        )
        self._process.start()

    def set_state(self, state: str):
        """Update the displayed state. Safe to call from any thread."""
        if self._state_queue:
            try:
                self._state_queue.put_nowait(state)
            except:
                pass

    def close(self):
        """Close the status window."""
        if self._shutdown_event:
            self._shutdown_event.set()
        if self._process and self._process.is_alive():
            self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.terminate()


# Standalone test
if __name__ == "__main__":
    import time

    print("Testing status window...")
    status = JarvisStatusWindow()
    status.start()

    states = ["off", "idle", "activated", "recording", "processing", "speaking"]

    for state in states:
        print(f"  State: {state}")
        status.set_state(state)
        time.sleep(1.5)

    print("Cycling through states...")
    for _ in range(3):
        for state in states:
            status.set_state(state)
            time.sleep(0.3)

    print("Done. Closing...")
    status.close()
    print("Closed.")
