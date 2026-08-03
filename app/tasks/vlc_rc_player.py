"""Persistent-VLC-instance playback via VLC's RC (remote control) interface.

Reusing a single long-lived VLC process across clips -- instead of spawning
a fresh vlc.exe (and re-entering fullscreen from scratch) per clip, see
emotion_faced.play_clip's "close_reopen"/"hide" transition_mode history --
removes the per-clip process-spawn + decoder-init + fullscreen-entry cost,
which is the main remaining source of the flicker/flash on each clip
transition once the "hide" window-transition fix was already in place.

This revisits the exact area (a single VLC instance staying alive and being
driven programmatically) that caused reliability bugs before ("--no-one-
instance" was added because VLC's default "hand off to an already-running
instance via IPC" behavior sometimes played the wrong file or never returned
control). To avoid repeating that, this module owns its OWN dedicated VLC
process (never relies on VLC's IPC/one-instance handoff -- we launch exactly
one process and talk to it directly over a local TCP socket we opened
ourselves), and every operation is wrapped to raise VlcSessionError on any
problem rather than fail silently, so the caller (emotion_faced.play_clip)
can catch it and fall back to the old known-good per-clip subprocess
approach for the rest of the session.

The RC wire protocol was verified empirically against the installed VLC
(see scratch probe scripts used during development) rather than assumed from
memory -- notably: `status` reports numeric state codes and randomly
interleaves unsolicited "status change: (...)" event push lines into
whatever response you're reading (this happens even with --rc-quiet), so
`is_playing` (a clean bare "0"/"1" reply) is polled instead, and every poll's
raw bytes are appended to a running per-play() buffer that's regex-scanned
for standalone "0"/"1" lines -- the last one found is trusted, so a reply
digit that happens to land in the same recv() as several interleaved event
lines is still read correctly.
"""

from __future__ import annotations

import re
import socket
import subprocess
import time
from pathlib import Path

RC_HOST = "127.0.0.1"
RC_PORT = 4212  # arbitrary local-only port -- only this module's own VLC process ever binds to it
_POLL_INTERVAL_SEC = 0.15
_STARTUP_TIMEOUT_SEC = 10
_SOCKET_TIMEOUT_SEC = 5
_DRAIN_IDLE_SEC = 0.15  # how long to wait for "no more data pending" when draining the socket -- comfortably above localhost round-trip latency
_BARE_DIGIT_LINE = re.compile(r"(?m)^([01])\r?$")


class VlcSessionError(RuntimeError):
    pass


class PersistentVlcSession:
    """One VLC process + one RC control socket, reused across play() calls."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None

    def start(self, player_path: str) -> None:
        try:
            self._proc = subprocess.Popen(
                [
                    player_path,
                    "--fullscreen",
                    "--no-video-title-show",
                    "--no-one-instance",
                    "--extraintf", "rc",
                    "--rc-host", f"{RC_HOST}:{RC_PORT}",
                    "--rc-quiet",
                    "--no-repeat",
                    "--no-loop",
                    "--quiet",
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise VlcSessionError(f"Failed to launch VLC ({player_path}): {exc}") from exc
        deadline = time.time() + _STARTUP_TIMEOUT_SEC
        last_error: Exception | None = None
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise VlcSessionError(f"VLC exited during startup (returncode={self._proc.returncode}).")
            try:
                sock = socket.create_connection((RC_HOST, RC_PORT), timeout=_SOCKET_TIMEOUT_SEC)
                sock.settimeout(_SOCKET_TIMEOUT_SEC)
                self._sock = sock
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.2)
        self.stop()
        raise VlcSessionError(f"VLC RC interface never came up on {RC_HOST}:{RC_PORT}: {last_error}")

    def _send(self, command: str) -> None:
        if self._sock is None:
            raise VlcSessionError("VLC RC socket is not connected.")
        try:
            self._sock.sendall((command + "\n").encode("utf-8"))
        except OSError as exc:
            raise VlcSessionError(f"Failed to send RC command {command!r}: {exc}") from exc

    def _drain(self) -> str:
        """Reads whatever the RC socket has buffered right now, waiting up to
        _DRAIN_IDLE_SEC of silence before deciding there's nothing more."""
        if self._sock is None:
            raise VlcSessionError("VLC RC socket is not connected.")
        chunks = []
        self._sock.settimeout(_DRAIN_IDLE_SEC)
        try:
            while True:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        except socket.timeout:
            pass
        except OSError as exc:
            raise VlcSessionError(f"Failed to read from VLC RC socket: {exc}") from exc
        finally:
            self._sock.settimeout(_SOCKET_TIMEOUT_SEC)
        return b"".join(chunks).decode("utf-8", errors="ignore")

    def play(self, file_path: Path, timeout_sec: float) -> None:
        """Blocks until `file_path` finishes playing (or `timeout_sec`
        elapses -- mirrors the old per-clip subprocess timeout safety net)."""
        if self._proc is None or self._proc.poll() is not None:
            raise VlcSessionError("VLC process is not running.")
        self._send("clear")
        self._drain()
        # RC's `add` enqueues AND starts playing immediately.
        self._send(f"add {file_path.resolve().as_uri()}")
        self._drain()

        buffer = ""
        started = False
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise VlcSessionError(f"VLC process exited unexpectedly (returncode={self._proc.returncode}).")
            self._send("is_playing")
            buffer += self._drain()
            digits = _BARE_DIGIT_LINE.findall(buffer)
            if digits:
                if "1" in digits:
                    started = True
                if started and digits[-1] == "0":
                    return
            time.sleep(_POLL_INTERVAL_SEC)
        raise VlcSessionError(f"Timed out after {timeout_sec:.0f}s waiting for {file_path.name} to finish.")

    def stop(self) -> None:
        if self._sock is not None:
            try:
                self._send("quit")
            except VlcSessionError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
