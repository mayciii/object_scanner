"""Object detection with voice output.
Phone (IP Webcam / DroidCam) -> Ultralytics YOLO -> spoken announcements.
Keys: q = quit, m = mute/unmute
"""
import argparse
import queue
import threading
import time
from collections import Counter

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Optional: prefer win32com SAPI5 on Windows for maximum reliability.
# Falls back to pyttsx3-per-utterance if win32com is not installed.
# ---------------------------------------------------------------------------
try:
    import win32com.client as _win32com
    _USE_WIN32COM = True
    print("TTS backend: win32com SAPI5")
except ImportError:
    import pyttsx3
    _USE_WIN32COM = False
    print("TTS backend: pyttsx3 (fresh engine per utterance)")

# How long (seconds) a queued item may wait before being discarded as stale.
_QUEUE_TTL = 2.5

# Per-label cooldown: once a label is announced, don't speak it again
# for this many seconds — even if it briefly leaves and re-enters the frame.
ANNOUNCE_COOLDOWN = 3.0


def _speak_win32com(text: str, rate: int) -> None:
    """Speak *text* synchronously via Windows SAPI5 COM interface."""
    sapi = _win32com.Dispatch("SAPI.SpVoice")
    # SAPI rate: -10 (slowest) .. 0 (normal) .. +10 (fastest).
    # pyttsx3 rate≈170 wpm maps roughly to SAPI rate 0.
    sapi_rate = max(-10, min(10, (rate - 170) // 15))
    sapi.Rate = sapi_rate
    sapi.Speak(text)
    del sapi


def _speak_pyttsx3(text: str, rate: int) -> None:
    """Speak *text* by creating a fresh pyttsx3 engine for this utterance."""
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", rate)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    finally:
        # engine.stop() drains the event loop; then we let the object die.
        try:
            engine.stop()
        except Exception:
            pass
        del engine


def _speak(text: str, rate: int) -> None:
    if _USE_WIN32COM:
        _speak_win32com(text, rate)
    else:
        _speak_pyttsx3(text, rate)


# ---------------------------------------------------------------------------
# Speaker — one dedicated worker thread; main loop only puts to a queue.
# ---------------------------------------------------------------------------
class Speaker:
    """Offline TTS on its own thread so detection never blocks."""

    def __init__(self, rate: int = 170):
        # Queue items: (enqueue_time: float, text: str)  or  None (sentinel)
        self.q: queue.Queue = queue.Queue()
        self.rate = rate
        self.muted = False
        self._busy = False
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name="Speaker")
        self._thread.start()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        print("Speaker thread started.")
        while True:
            item = self.q.get()

            # None sentinel → clean shutdown
            if item is None:
                print("Speaker thread received shutdown signal.")
                self.q.task_done()
                break

            enqueue_time, text = item

            # Drop stale announcements that piled up while engine was busy.
            age = time.monotonic() - enqueue_time
            if age > _QUEUE_TTL:
                print(f"[Speaker] Dropped stale item (age={age:.2f}s): {text!r}")
                self.q.task_done()
                with self._state_lock:
                    if self.q.empty():
                        self._busy = False
                continue

            print(f"[Speaker] Speaking: {text!r}")
            try:
                _speak(text, self.rate)
                print(f"[Speaker] Done speaking: {text!r}")
            except Exception as exc:
                print(f"[Speaker] ERROR during speech: {exc}")
            finally:
                self.q.task_done()
                with self._state_lock:
                    if self.q.empty():
                        self._busy = False

    # ------------------------------------------------------------------
    def say(self, text: str) -> bool:
        """Queue *text* for speaking. Returns False if muted."""
        with self._state_lock:
            if self.muted:
                return False
            self.q.put_nowait((time.monotonic(), text))
            self._busy = True
        print(f"[Speaker] Queued: {text!r}")
        return True

    def busy(self) -> bool:
        with self._state_lock:
            return self._busy

    def shutdown(self, timeout: float = 3.0) -> None:
        """Signal the worker to exit and wait for it to finish."""
        self.q.put(None)
        self._thread.join(timeout=timeout)
        print("Speaker thread shut down.")


# ---------------------------------------------------------------------------
# Stream — always-latest-frame reader (avoids lag from buffered network video)
# ---------------------------------------------------------------------------
class Stream:
    """Always-latest-frame reader to avoid lag from buffered network video."""

    def __init__(self, source):
        self.source = int(source) if str(source).isdigit() else source
        self.frame = None
        self.ok = True
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True, name="Stream").start()

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        while self.ok:
            if not cap.isOpened():
                time.sleep(1)
                cap = cv2.VideoCapture(self.source)
                continue
            good, frame = cap.read()
            if not good:
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.source)
                continue
            with self.lock:
                self.frame = frame
        cap.release()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.ok = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def phrase(name: str, n: int) -> str:
    count = "one" if n == 1 else str(n)
    item = name if n == 1 else "people" if name == "person" else name + "s"
    return f"{count} {item}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="http://192.168.198.167:8080/video",
                   help="Phone stream URL, or 0 for local webcam")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    model = YOLO(args.model)
    stream = Stream(args.source)
    speaker = Speaker()
    announced_counts: dict[str, int] = {}
    last_seen: dict[str, float] = {}
    last_announced_time: dict[str, float] = {}   # label -> monotonic time it was last spoken

    print("Connecting to stream... press q to quit, m to mute.")
    try:
        while True:
            frame = stream.read()
            if frame is None:
                time.sleep(0.05)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            names = [model.names[int(c)] for c in result.boxes.cls]
            counts = Counter(names)

            now = time.monotonic()
            for name in counts:
                last_seen[name] = now

            # Clear entries that haven't been seen for 0.75 s (left the frame).
            missing = [
                name for name, seen_at in last_seen.items()
                if now - seen_at > 0.75
            ]
            for name in missing:
                last_seen.pop(name)
                # Reset count so the label is announced fresh when it reappears,
                # but keep last_announced_time so the cooldown is still enforced.
                announced_counts.pop(name, None)
                print(f"[Tracker] {name!r} left the frame; count reset.")

            # Announce only objects whose count changed or are newly visible,
            # AND whose per-label cooldown has expired.
            newly_seen = [
                (name, count) for name, count in counts.items()
                if count > announced_counts.get(name, 0)
                and now - last_announced_time.get(name, 0.0) >= ANNOUNCE_COOLDOWN
            ]
            if newly_seen:
                text = "I see " + ", ".join(
                    phrase(name, count) for name, count in newly_seen
                ) + "."
                if speaker.say(text):
                    for name, count in newly_seen:
                        announced_counts[name] = count
                        last_announced_time[name] = now
                        print(f"[Cooldown] {name!r} cooldown started ({ANNOUNCE_COOLDOWN}s).")
            elif counts and not speaker.busy():
                # "Still see" reminder — also apply the cooldown per label in scene.
                eligible = [
                    (name, count) for name, count in sorted(counts.items())
                    if now - last_announced_time.get(name, 0.0) >= ANNOUNCE_COOLDOWN
                ]
                if eligible:
                    text = "I still see " + ", ".join(
                        phrase(name, count) for name, count in eligible
                    ) + "."
                    if speaker.say(text):
                        for name, count in eligible:
                            last_announced_time[name] = now

            cv2.imshow("Detection", result.plot())
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                speaker.muted = not speaker.muted
                print("Muted." if speaker.muted else "Unmuted.")

    finally:
        stream.stop()
        speaker.shutdown()
        cv2.destroyAllWindows()
        print("Exited cleanly.")


if __name__ == "__main__":
    main()
