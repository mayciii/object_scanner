"""Single-threaded voice announcement worker with reliable speech engine handling."""

import queue
import threading

# Prefer native Windows SAPI5 COM interface if available; fall back to pyttsx3.
try:
    import win32com.client as _win32com
    _USE_WIN32COM = True
except ImportError:
    import pyttsx3
    _USE_WIN32COM = False


def _speak_win32com(text: str) -> None:
    """Speak synchronously via Windows SAPI5 COM interface."""
    sapi = _win32com.Dispatch("SAPI.SpVoice")
    sapi.Speak(text)
    del sapi


def _speak_pyttsx3(text: str) -> None:
    """Speak text using a fresh pyttsx3 engine per utterance to avoid SAPI5 freezing."""
    engine = pyttsx3.init()
    try:
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass
        del engine


def _speak(text: str) -> None:
    if _USE_WIN32COM:
        _speak_win32com(text)
    else:
        _speak_pyttsx3(text)


class VoiceWorker:
    """Runs speech generation in a dedicated thread to keep the GUI responsive."""

    def __init__(self, events: queue.Queue):
        self.events = events
        self.messages = queue.Queue()
        self._lock = threading.Lock()
        self._muted = False
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="VoiceWorker", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self._muted = muted
            if muted:
                self._clear_pending()

    def announce(self, text: str) -> None:
        with self._lock:
            if self._muted or self._stopping:
                return
            self.messages.put(text)

    def request_stop(self) -> None:
        with self._lock:
            self._stopping = True
            self._clear_pending()
            self.messages.put(None)

    def _clear_pending(self) -> None:
        while True:
            try:
                self.messages.get_nowait()
                self.messages.task_done()
            except queue.Empty:
                break

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        # Check backend readiness
        try:
            if not _USE_WIN32COM:
                test_eng = pyttsx3.init()
                test_eng.stop()
                del test_eng
            self.events.put(("voice_ready", None))
        except Exception as exc:
            self.events.put(("voice_error", f"Could not initialize voice: {exc}"))
            return

        while True:
            text = self.messages.get()
            try:
                if text is None:
                    return
                with self._lock:
                    muted = self._muted
                if not muted:
                    _speak(text)
            except Exception as exc:
                self.events.put(("voice_error", f"Voice playback failed: {exc}"))
            finally:
                self.messages.task_done()
