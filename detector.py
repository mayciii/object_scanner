"""YOLO inference and camera capture for Android IP Webcam."""

from dataclasses import dataclass
import queue
import threading
import time
from urllib.parse import urlparse

import cv2
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float


@dataclass(frozen=True)
class DetectionResult:
    frame: object
    detections: list[Detection]
    fps: float
    distinct_types: int
    new_labels: list[str]


def normalize_camera_source(source: str | int) -> str:
    """Normalize phone stream URL for Android IP Webcam.
    
    Laptop/local webcams (0, 1, 2) are excluded — only phone camera is supported.
    Handles:
    - Raw IP only: "192.168.1.15" -> "http://192.168.1.15:8080/video"
    - IP with port: "192.168.1.15:8080" -> "http://192.168.1.15:8080/video"
    - Root URL: "http://192.168.1.15:8080" -> "http://192.168.1.15:8080/video"
    - Root URL with slash: "http://192.168.1.15:8080/" -> "http://192.168.1.15:8080/video"
    - Full URL: "http://192.168.1.15:8080/video" -> kept as-is
    - Rejects integer indices (0, 1, 2) since laptop webcam is excluded.
    """
    text = str(source).strip()
    if not text:
        raise ValueError("Please enter your Android phone IP Webcam address (e.g. http://192.168.1.X:8080/video).")

    # Strip any trailing label like " (Hotspot)"
    if " (" in text:
        text = text.split(" (", 1)[0].strip()
    elif " - " in text and not text.startswith("http"):
        text = text.split(" - ", 1)[0].strip()

    # Reject local/laptop webcam indices
    if text.isdigit():
        raise ValueError(
            "Laptop webcam is disabled. Only Android phone IP Webcam is supported.\n"
            "Please enter your phone stream URL (e.g. http://192.168.1.X:8080/video)."
        )

    # Prepend http:// if missing
    if not (text.startswith("http://") or text.startswith("https://") or text.startswith("rtsp://")):
        text = f"http://{text}"

    # Parse and normalize endpoint for IP Webcam
    try:
        parsed = urlparse(text)
        netloc = parsed.netloc

        # If user gave IP without port (e.g. 192.168.1.15), default IP Webcam port is 8080
        if ":" not in netloc:
            netloc = f"{netloc}:8080"

        path = parsed.path.rstrip("/")
        # IP Webcam streams video at /video or /videofeed
        if not path or path == "":
            path = "/video"
        elif path == "/videofeed":
            path = "/video"

        text = f"{parsed.scheme}://{netloc}{path}"
    except Exception:
        if not text.endswith("/video"):
            text = f"{text.rstrip('/')}/video"

    return text


class ObjectDetector:
    """Loads YOLO once and reports newly appearing object labels."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.25,
        image_size: int = 640,
        label_cooldown: float = 4.0,
        model=None,
    ):
        self.model = model if model is not None else YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.label_cooldown = label_cooldown
        self._visible_labels: set[str] = set()
        self._last_announced: dict[str, float] = {}
        self._labels_seen: set[str] = set()
        self._last_frame_time: float | None = None

    def reset_presence(self) -> None:
        """Treat a camera reconnect as a new visibility session."""
        self._visible_labels.clear()
        self._last_frame_time = None

    def process(self, frame) -> DetectionResult:
        """Run inference, draw detections, and track label entry/exit."""
        result = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )[0]

        detections = [
            Detection(
                label=str(self.model.names[int(class_id)]),
                confidence=float(conf),
            )
            for class_id, conf in zip(result.boxes.cls, result.boxes.conf)
        ]

        now = time.monotonic()
        labels_now = {detection.label for detection in detections}

        # Announce when newly appearing (not in previous frame)
        # and when the per-label cooldown (4s) has passed.
        new_labels = [
            label
            for label in sorted(labels_now - self._visible_labels)
            if now - self._last_announced.get(label, float("-inf"))
            >= self.label_cooldown
        ]
        for label in new_labels:
            self._last_announced[label] = now

        self._visible_labels = labels_now
        self._labels_seen.update(labels_now)

        fps = 0.0
        if self._last_frame_time is not None:
            elapsed = now - self._last_frame_time
            if elapsed > 0:
                fps = 1.0 / elapsed
        self._last_frame_time = now

        return DetectionResult(
            frame=result.plot(),
            detections=detections,
            fps=fps,
            distinct_types=len(self._labels_seen),
            new_labels=new_labels,
        )


class DetectionWorker:
    """Owns phone camera streaming and YOLO inference on a background thread."""

    FRAME_TIMEOUT = 2.5

    def __init__(
        self,
        events: queue.Queue,
        updates: queue.Queue,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.25,
        image_size: int = 640,
    ):
        self.events = events
        self.updates = updates
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.commands = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="DetectionWorker", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def connect(self, source: str) -> None:
        self.commands.put(("connect", source))

    def disconnect(self) -> None:
        self.commands.put(("disconnect", None))

    def request_stop(self) -> None:
        self._stop.set()
        self.commands.put(("stop", None))

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @staticmethod
    def _open_camera(raw_source: str):
        source = normalize_camera_source(raw_source)

        # Android IP Webcam network stream over Wi-Fi
        # Give Wi-Fi network streams 5 seconds to connect and 3 seconds to read
        timeout_parameters = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            3000,
        ]
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, timeout_parameters)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap, source
        except (TypeError, cv2.error):
            pass

        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, source

    def _publish_update(self, update: DetectionResult) -> None:
        try:
            self.updates.put_nowait(update)
        except queue.Full:
            try:
                self.updates.get_nowait()
            except queue.Empty:
                pass
            self.updates.put_nowait(update)

    def _run(self) -> None:
        camera = None
        source_desc = None
        detector = None

        try:
            try:
                detector = ObjectDetector(
                    self.model_path, self.confidence, self.image_size
                )
                self.events.put(("model_ready", None))
            except Exception as exc:
                self.events.put(("model_error", str(exc)))

            last_frame_at = None
            last_read_error = None

            while not self._stop.is_set():
                try:
                    command, value = self.commands.get(
                        timeout=0.02 if camera is None else 0
                    )
                except queue.Empty:
                    command = None
                    value = None

                if command == "stop":
                    break

                if command == "disconnect":
                    if camera is not None:
                        camera.release()
                        camera = None
                        source_desc = None
                    if detector is not None:
                        detector.reset_presence()
                    self.events.put(("camera_disconnected", ""))

                elif command == "connect":
                    if camera is not None:
                        camera.release()
                        camera = None
                    if detector is not None:
                        detector.reset_presence()

                    raw_source = value
                    if detector is None:
                        self.events.put(
                            ("camera_disconnected", "The YOLO model is not loaded.")
                        )
                        continue

                    try:
                        camera, normalized = self._open_camera(raw_source)
                        source_desc = str(normalized)
                    except Exception as exc:
                        camera = None
                        self.events.put(
                            ("camera_disconnected", f"Invalid camera source: {exc}")
                        )
                        continue

                    try:
                        opened = camera.isOpened()
                    except cv2.error as exc:
                        camera.release()
                        camera = None
                        self.events.put(("camera_disconnected", str(exc)))
                        continue

                    if not opened:
                        camera.release()
                        camera = None
                        err_msg = (
                            f"Could not connect to IP Webcam at {source_desc}.\n"
                            "1) Ensure Android phone & laptop are on the same Wi-Fi.\n"
                            "2) Open IP Webcam app and tap 'Start server'.\n"
                            "3) Check that the IP and port match your phone screen."
                        )
                        self.events.put(("camera_disconnected", err_msg))
                        continue

                    last_frame_at = time.monotonic()
                    self.events.put(("camera_connected", source_desc))

                if camera is None:
                    continue

                try:
                    success, frame = camera.read()
                except cv2.error as exc:
                    success, frame = False, None
                    last_read_error = str(exc)

                now = time.monotonic()
                if not success or frame is None:
                    if (
                        last_frame_at is not None
                        and now - last_frame_at >= self.FRAME_TIMEOUT
                    ):
                        camera.release()
                        camera = None
                        self.events.put(
                            (
                                "camera_disconnected",
                                last_read_error
                                or f"IP Webcam stream lost ({source_desc}): no frames received for 2 seconds.",
                            )
                        )
                        source_desc = None
                        last_read_error = None
                    else:
                        self._stop.wait(0.02)
                    continue

                last_frame_at = now
                last_read_error = None

                try:
                    result = detector.process(frame)
                except Exception as exc:
                    self.events.put(("detection_error", str(exc)))
                    continue

                for label in result.new_labels:
                    self.events.put(("announce", f"{label} detected"))

                self._publish_update(result)

        except Exception as exc:
            self.events.put(("worker_error", str(exc)))
        finally:
            if camera is not None:
                try:
                    camera.release()
                except cv2.error as exc:
                    self.events.put(("worker_error", f"Could not release camera: {exc}"))
            self.events.put(("camera_disconnected", ""))
            self.events.put(("worker_stopped", None))
