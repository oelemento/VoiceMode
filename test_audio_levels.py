#!/usr/bin/env python3.11
"""Test audio levels and silence detection for different mics."""

import struct
import time
import numpy as np
import pyaudio

SAMPLE_RATE = 16000
FRAME_SIZE = 512  # Same as Porcupine


def get_audio_level(audio_data):
    """Get RMS audio level from raw bytes."""
    if len(audio_data) < 2:
        return 0
    shorts = struct.unpack(f'{len(audio_data)//2}h', audio_data)
    return int(np.sqrt(np.mean(np.square(shorts))))


def list_input_devices():
    """List available input devices."""
    p = pyaudio.PyAudio()
    print("Available INPUT devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    p.terminate()


def monitor_audio(device_index=None, duration=10):
    """Monitor audio levels for a device."""
    p = pyaudio.PyAudio()

    if device_index is not None:
        info = p.get_device_info_by_index(device_index)
        print(f"\nMonitoring: {info['name']}")
    else:
        print("\nMonitoring: default device")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=FRAME_SIZE
    )

    print(f"Recording for {duration} seconds...")
    print("Speak, then stay silent to see the difference.\n")

    levels = []
    start_time = time.time()

    # Visual bar width
    bar_width = 50

    try:
        while time.time() - start_time < duration:
            audio_data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            level = get_audio_level(audio_data)
            levels.append(level)

            # Visual bar (scale to 5000 max)
            bar_len = min(int(level / 5000 * bar_width), bar_width)
            bar = "█" * bar_len + "░" * (bar_width - bar_len)

            # Color based on level
            if level < 500:
                color = "\033[90m"  # Gray - very quiet
            elif level < 1500:
                color = "\033[92m"  # Green - quiet
            elif level < 3000:
                color = "\033[93m"  # Yellow - medium
            else:
                color = "\033[91m"  # Red - loud

            print(f"\r{color}[{bar}] {level:5d}\033[0m", end="", flush=True)

    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Stats
    print("\n\n--- Statistics ---")
    print(f"Min:    {min(levels):5d}")
    print(f"Max:    {max(levels):5d}")
    print(f"Avg:    {int(sum(levels)/len(levels)):5d}")

    # Find noise floor (bottom 10%)
    sorted_levels = sorted(levels)
    noise_floor = sorted_levels[len(sorted_levels) // 10]
    print(f"Noise floor (10th percentile): {noise_floor}")

    # Suggested threshold
    suggested = int(noise_floor * 1.7)
    print(f"\nSuggested silence threshold: {suggested}")
    print(f"(noise_floor * 1.7)")

    return levels


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test audio levels")
    parser.add_argument("--device", "-d", type=int, help="Input device index")
    parser.add_argument("--list", "-l", action="store_true", help="List devices")
    parser.add_argument("--duration", "-t", type=int, default=10, help="Duration in seconds")

    args = parser.parse_args()

    if args.list:
        list_input_devices()
        return

    monitor_audio(args.device, args.duration)


if __name__ == "__main__":
    main()
