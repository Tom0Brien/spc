"""Minimal FFmpeg-based video recorder for the interactive examples.

Self-contained (no hydrax dependency) so the standalone ``spc`` package can
record the MuJoCo viewer without pulling in the research repo.
"""

import os
import subprocess
from datetime import datetime


class VideoRecorder:
    """Pipe raw RGB frames to FFmpeg and encode an mp4."""

    def __init__(self, output_dir, width=720, height=480, fps=30.0):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps

        self.ffmpeg_process = None
        self.video_path = None
        self.is_recording = False

    def start(self):
        """Start the FFmpeg process. Returns True on success."""
        if self.is_recording:
            print("Warning: Recording already in progress")
            return True

        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(self.output_dir, f"simulation_{timestamp}.mp4")

        cmd = [
            "ffmpeg",
            "-y",  # overwrite output file
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",  # input from stdin pipe
            "-an",  # no audio
            "-vcodec", "h264",
            "-crf", "18",  # visually lossless
            "-preset", "medium",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            self.video_path,
        ]

        try:
            self.ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: FFmpeg not found. Video recording disabled.")
            self.is_recording = False
            return False

        self.is_recording = True
        print(f"Recording video to {self.video_path}")
        return True

    def add_frame(self, frame):
        """Write a raw RGB frame (bytes) to FFmpeg. Returns True on success."""
        if not self.is_recording or self.ffmpeg_process is None or self.ffmpeg_process.stdin is None:
            return False
        try:
            self.ffmpeg_process.stdin.write(frame)
            return True
        except (BrokenPipeError, IOError):
            print("Warning: Failed to write frame to video")
            self.is_recording = False
            return False

    def stop(self):
        """Flush and finalize the video. Returns True on success."""
        if not self.is_recording or self.ffmpeg_process is None:
            return False
        try:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            print(f"\nVideo saved to {self.video_path}")
            self.is_recording = False
            return True
        except (BrokenPipeError, IOError) as e:
            print(f"Warning: Error finalizing video: {e}")
            try:
                self.ffmpeg_process.terminate()
            except Exception:
                pass
            self.is_recording = False
            return False
