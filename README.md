# YOLO Voice Detector (Android IP Webcam)

A compact, dark-themed Tkinter desktop application for real-time YOLO object detection using an **Android smartphone running IP Webcam** as the dedicated wireless camera source, with live telemetry and spoken voice announcements.

---

## Features

- **Dedicated to Android Phone Camera**: Built exclusively for wireless phone camera input via the popular free **IP Webcam** app. Laptop webcams are completely disabled.
- **Smart URL Normalization**: Automatically converts shorthand inputs (e.g. `192.168.1.15` or `192.168.1.15:8080`) into full stream endpoints (`http://192.168.1.15:8080/video`) so connections never fail from missing paths or port numbers.
- **Zero-Lag Wi-Fi Streaming**: Uses `CAP_PROP_BUFFERSIZE = 1` and dedicated background frame polling so video never buffers or lags behind real life.
- **Clickable Status Badge**: Displays `"Phone Camera: Connected"` (green) or `"Phone Camera: Disconnected"` (red). Clicking the badge, the Connect button, or the preview screen toggles the stream.
- **Live Telemetry & Object Counter**: Live bounding boxes, class names, confidence scores, real-time FPS counter, and distinct detected object counter (goal: at least 5).
- **Spoken Voice Announcements**:
  - Speaks `"<label> detected"` via a background thread with queue.
  - Announce-on-entry: speaks only once when an object enters view.
  - Anti-repeat: never repeats while the object remains visible.
  - Per-label cooldown: 4 seconds must elapse after the object leaves view before it can be announced again.
  - Mute checkbox to toggle voice anytime.

---

## Quick Setup

Install requirements in your virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

`yolo11n.pt` will load automatically on startup.

---

## How to Connect Your Android Phone (IP Webcam)

1. **Connect to Same Network**:
   - Connect both your Android phone and laptop to the **same Wi-Fi router**, **OR**
   - Turn on your Android phone's **Mobile Hotspot** and connect your laptop to it.

2. **Start Server in IP Webcam**:
   - Install and open **IP Webcam** (by Pavel Khlebovich) on your Android phone.
   - Scroll down to the bottom of the main menu and tap **Start server**.
   - Your phone will display camera video and show an address at the bottom of the screen, e.g.:
     `http://192.168.1.25:8080` (or `http://172.20.10.8:8080`)

3. **Connect in YOLO Voice Detector**:
   - In the **Android IP Webcam** box, enter the address shown on your phone:
     `http://<phone-ip>:8080/video` (or simply type `<phone-ip>:8080`).
   - Click **Connect** (or click the preview window).
   - The status badge turns green (`Phone Camera: Connected`), and the live detection feed appears immediately.

---

## Architecture Overview

- **`detector.py`**:
  - Independent of the GUI.
  - Normalizes Android IP Webcam stream endpoints.
  - Manages `DetectionWorker` background thread for network frame capture and timeout handling (2.5s).
  - Runs `ObjectDetector` (YOLO inference, 4-second presence/cooldown tracker).
- **`voice.py`**:
  - Independent background `VoiceWorker` thread with queue.
  - Handles Windows SAPI5 / pyttsx3 speech synthesis so voice never hangs or freezes after the first announcement.
- **`main.py`**:
  - Pure Tkinter / Pillow UI running strictly on the main thread.
  - Dispatches events from worker queues via `root.after()`.
  - Minimal dark palette (`#111820`).
