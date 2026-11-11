"""Dark Tkinter interface for real-time YOLO object detection via Android IP Webcam."""

import json
from pathlib import Path
import queue
import socket
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from detector import DetectionWorker, normalize_camera_source
from voice import VoiceWorker


# Theme Palette
BACKGROUND = "#111820"
PANEL = "#1b2632"
CARD = "#141d26"
TEXT = "#e7edf3"
MUTED = "#9eacb9"
GREEN = "#39b878"
RED = "#d95757"
AMBER = "#d79b36"


def get_local_ip() -> str:
    """Find local network IP to assist with Android phone Wi-Fi setup."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.1.X"


class DetectorApp:
    """Desktop GUI managing Android phone streaming, detection displays, and voice alerts."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Voice Detector - Android IP Webcam")
        self.root.geometry("1160x740")
        self.root.minsize(880, 580)
        self.root.configure(bg=BACKGROUND)

        self.events: queue.Queue = queue.Queue()
        self.updates: queue.Queue = queue.Queue(maxsize=1)
        self.settings_path = Path(__file__).with_name("settings.json")
        self.camera_state = "disconnected"
        self.closing = False
        self.photo = None
        self._current_frame = None

        self._configure_style()
        self._build_layout()

        self.detector = DetectionWorker(
            self.events,
            self.updates,
            model_path="yolo11n.pt",
            confidence=0.25,
            image_size=640,
        )
        self.voice = VoiceWorker(self.events)
        self.detector.start()
        self.voice.start()

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(40, self._poll_workers)

    def _configure_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("App.TFrame", background=BACKGROUND)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("App.TLabel", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BACKGROUND, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Info.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 8))
        style.configure("PanelTitle.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 11, "bold"))
        style.configure(
            "App.TButton",
            background="#2b3947",
            foreground=TEXT,
            padding=(12, 7),
            borderwidth=0,
            font=("Segoe UI", 9, "bold"),
        )
        style.map("App.TButton", background=[("active", "#3a4b5a"), ("disabled", "#25313c")])
        style.configure("App.TCheckbutton", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))

    def _build_layout(self):
        # 1. Header with Phone Camera controls
        header = ttk.Frame(self.root, style="App.TFrame", padding=(16, 12, 16, 6))
        header.pack(fill="x")
        header.columnconfigure(1, weight=1)

        ttk.Label(header, text="ANDROID IP WEBCAM", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )

        last_source, presets = self._load_settings()
        self.source = ttk.Combobox(
            header,
            values=presets,
            font=("Segoe UI", 10),
        )
        self.source.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.source.set(last_source)
        self.source.bind("<Return>", lambda _event: self.toggle_camera())

        # Clickable status badge (toggles camera on click)
        self.status_button = tk.Button(
            header,
            text="Phone Camera: Disconnected",
            command=self.toggle_camera,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=7,
            bg=RED,
            fg="white",
            activebackground=RED,
            activeforeground="white",
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        self.status_button.grid(row=0, column=2, padx=(0, 8))

        # Plain Connect / Disconnect button
        self.connect_button = ttk.Button(
            header,
            text="Connect",
            command=self.toggle_camera,
            style="App.TButton",
            cursor="hand2",
        )
        self.connect_button.grid(row=0, column=3)

        # 2. Status message & Network helper info
        sub_bar = ttk.Frame(self.root, style="App.TFrame", padding=(16, 0, 16, 8))
        sub_bar.pack(fill="x")
        sub_bar.columnconfigure(0, weight=1)

        self.message = ttk.Label(
            sub_bar,
            text="Loading YOLO model…",
            style="Muted.TLabel",
        )
        self.message.grid(row=0, column=0, sticky="w")

        local_ip = get_local_ip()
        self.network_hint = ttk.Label(
            sub_bar,
            text=f"Same Wi-Fi network • Laptop IP: {local_ip} • IP Webcam Default: http://<phone-ip>:8080/video",
            style="Muted.TLabel",
        )
        self.network_hint.grid(row=0, column=1, sticky="e")

        # 3. Main content area: Live Preview (left) + Detection Panel (right)
        content = ttk.Frame(self.root, style="App.TFrame", padding=(16, 0, 16, 16))
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        # Video frame preview area (clickable to toggle camera)
        self.preview = tk.Label(
            content,
            text="Phone Camera disconnected.\n\nOpen IP Webcam on Android, tap 'Start server', and click here to connect.",
            bg="#080d12",
            fg=MUTED,
            font=("Segoe UI", 13),
            justify="center",
            cursor="hand2",
        )
        self.preview.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.preview.bind("<Configure>", self._resize_current_frame)
        self.preview.bind("<Button-1>", lambda _event: self.toggle_camera())

        # Side detection panel
        panel = ttk.Frame(content, style="Panel.TFrame", padding=14, width=280)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)
        panel.rowconfigure(2, weight=1)

        ttk.Label(panel, text="DETECTIONS", style="PanelTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.fps_label = ttk.Label(panel, text="FPS: --", style="Panel.TLabel")
        self.fps_label.grid(row=1, column=0, sticky="w", pady=(0, 6))

        # List of current frame detections
        self.detection_list = tk.Listbox(
            panel,
            bg=CARD,
            fg=TEXT,
            selectbackground="#2d4050",
            highlightthickness=0,
            borderwidth=0,
            font=("Segoe UI", 10),
            activestyle="none",
        )
        self.detection_list.grid(row=2, column=0, sticky="nsew")
        self.detection_list.insert("end", "No objects detected")

        # Distinct object counter
        self.type_count = ttk.Label(
            panel,
            text="Distinct types seen: 0 / 5",
            style="Panel.TLabel",
            font=("Segoe UI", 10, "bold"),
        )
        self.type_count.grid(row=3, column=0, sticky="w", pady=(12, 8))

        # Voice mute checkbox
        self.muted = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            panel,
            text="Mute voice announcements",
            variable=self.muted,
            command=lambda: self.voice.set_muted(self.muted.get()),
            style="App.TCheckbutton",
        ).grid(row=4, column=0, sticky="w", pady=(0, 10))

        # Instructions card for Android IP Webcam
        phone_card = tk.LabelFrame(
            panel,
            text=" ANDROID IP WEBCAM GUIDE ",
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=6,
        )
        phone_card.grid(row=5, column=0, sticky="ew")

        guide_text = (
            "1. Connect Android phone & laptop to the\n"
            "   same Wi-Fi (or phone mobile hotspot).\n"
            "2. Open IP Webcam on your Android phone.\n"
            "3. Scroll to the bottom & tap 'Start server'.\n"
            "4. Enter the shown URL in the box above\n"
            "   (e.g. http://192.168.1.X:8080/video).\n"
            "5. Click Connect."
        )
        tk.Label(
            phone_card,
            text=guide_text,
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 8),
            justify="left",
            anchor="w",
        ).pack(fill="x")

    def _load_settings(self) -> tuple[str, list[str]]:
        default_presets = [
            "http://100.94.215.244:8080/video",
            "http://192.168.1.100:8080/video",
            "http://172.20.10.8:8080/video",
        ]
        try:
            if self.settings_path.exists():
                data = json.loads(self.settings_path.read_text(encoding="utf-8"))
                current = str(data.get("camera_source", default_presets[0]))
                # Filter out old laptop indices like "0", "1", "2"
                if current.isdigit():
                    current = default_presets[0]
                history = data.get("history", [])
                if not isinstance(history, list):
                    history = []
                # Remove any numeric indices from history
                clean_history = [str(item).strip() for item in history if not str(item).strip().isdigit()]
                combined = []
                for item in [current] + clean_history + default_presets:
                    s = str(item).strip()
                    if s and s not in combined and not s.isdigit():
                        combined.append(s)
                return current, combined
        except Exception as exc:
            print(f"Could not read saved camera source: {exc}")
        return default_presets[0], default_presets

    def _save_settings(self, source: str) -> None:
        try:
            current_values = list(self.source["values"]) if self.source["values"] else []
            if source not in current_values:
                current_values.insert(0, source)
                self.source["values"] = current_values[:10]
            data = {
                "camera_source": source,
                "history": current_values[:10],
            }
            self.settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            self.message.configure(text=f"Could not save settings: {exc}")

    def toggle_camera(self):
        if self.closing or self.camera_state in ("connecting", "disconnecting"):
            return

        if self.camera_state == "connected":
            self.camera_state = "disconnecting"
            self._set_status("disconnecting", "Disconnecting phone camera…")
            self.detector.disconnect()
            return

        raw_source = self.source.get().strip()
        if not raw_source:
            self._set_status("disconnected", "Please enter your Android IP Webcam URL (e.g. http://192.168.1.X:8080/video).")
            return

        try:
            normalized = normalize_camera_source(raw_source)
        except ValueError as err:
            self._set_status("disconnected", str(err))
            return

        # Update entry if normalized
        if str(normalized) != raw_source:
            self.source.set(str(normalized))

        self._save_settings(str(normalized))
        self.camera_state = "connecting"
        self._set_status("connecting", f"Connecting to {normalized}…")
        self.detector.connect(normalized)

    def _set_status(self, state: str, message: str = ""):
        self.camera_state = state
        if state == "connected":
            label, color = "Phone Camera: Connected", GREEN
            button = "Disconnect"
        elif state in ("connecting", "disconnecting"):
            label = "Phone Camera: Connecting…" if state == "connecting" else "Phone Camera: Disconnecting…"
            color = AMBER
            button = "Connecting…" if state == "connecting" else "Disconnecting…"
        else:
            label, color = "Phone Camera: Disconnected", RED
            button = "Connect"

        self.status_button.configure(
            text=label,
            bg=color,
            activebackground=color,
            state="disabled" if state in ("connecting", "disconnecting") else "normal",
        )
        self.connect_button.configure(
            text=button,
            state="disabled" if state in ("connecting", "disconnecting") else "normal",
        )
        self.message.configure(text=message)

        if state == "disconnected":
            self._show_placeholder()
            self.detection_list.delete(0, "end")
            self.detection_list.insert("end", "No objects detected")
            self.fps_label.configure(text="FPS: --")

    def _show_placeholder(self):
        self.photo = None
        self._current_frame = None
        self.preview.configure(
            image="",
            text="Phone Camera disconnected.\n\nOpen IP Webcam on Android, tap 'Start server', and click here to connect.",
        )

    def _poll_workers(self):
        if self.closing:
            return

        while True:
            try:
                event, value = self.events.get_nowait()
            except queue.Empty:
                break

            if event == "model_ready":
                self.message.configure(text="YOLO model ready. Enter Android IP Webcam URL and click Connect.")
            elif event == "model_error":
                self._set_status("disconnected", f"Could not load YOLO model: {value}")
            elif event == "camera_connected":
                self._set_status("connected", f"Connected to {value}")
            elif event == "camera_disconnected":
                self._set_status("disconnected", value or "Phone Camera disconnected.")
            elif event == "detection_error":
                self.message.configure(text=f"Detection error: {value}")
            elif event == "worker_error":
                self._set_status("disconnected", f"Worker error: {value}")
            elif event == "announce":
                self.voice.announce(value)
            elif event == "voice_error":
                self.message.configure(text=value)

        try:
            update = self.updates.get_nowait()
        except queue.Empty:
            update = None

        if update is not None and self.camera_state == "connected":
            self._show_update(update)

        self.root.after(40, self._poll_workers)

    def _show_update(self, update):
        self._current_frame = update.frame
        self._render_frame()

        self.detection_list.delete(0, "end")
        if update.detections:
            for detection in update.detections:
                self.detection_list.insert(
                    "end",
                    f"{detection.label}  {detection.confidence * 100:.1f}%",
                )
        else:
            self.detection_list.insert("end", "No objects detected")

        fps = f"{update.fps:.1f}" if update.fps else "--"
        self.fps_label.configure(text=f"FPS: {fps}")

        distinct_text = f"Distinct types seen: {update.distinct_types} / 5"
        if update.distinct_types >= 5:
            distinct_text += " (Goal reached!)"
        self.type_count.configure(text=distinct_text)

    def _resize_current_frame(self, _event=None):
        if getattr(self, "_current_frame", None) is not None:
            self._render_frame()

    def _render_frame(self):
        frame = self._current_frame
        width = max(self.preview.winfo_width(), 2)
        height = max(self.preview.winfo_height(), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=image)
        self.preview.configure(image=self.photo, text="")

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.connect_button.configure(state="disabled")
        self.status_button.configure(state="disabled")
        self.message.configure(text="Closing phone stream and voice worker…")
        self.detector.request_stop()
        self.voice.request_stop()
        self.root.after(50, self._finish_close)

    def _finish_close(self):
        if self.detector.is_alive() or self.voice.is_alive():
            self.root.after(50, self._finish_close)
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    DetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
