# YOLO Voice Detector

Android phone as camera -> Ultralytics YOLO on your PC -> spoken announcements.

## 1. Phone setup
Install **IP Webcam** (Play Store), tap **Start server**. Phone and PC must be on the same Wi-Fi.
The app shows an address like `http://192.168.1.23:8080`. Your stream URL is that address plus `/video`.

(Alternative: DroidCam, stream URL `http://<ip>:4747/video`.)

## 2. Install
```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```
Linux only, for voice: `sudo apt install espeak-ng`

## 3. Run
```
python main.py --source http://192.168.1.23:8080/video
```
Use `--source 0` to test with a laptop webcam first.

## Options
| Flag | Default | Purpose |
|---|---|---|
| `--model` | yolo11n.pt | Weights (downloads automatically). Use yolo11s.pt for better accuracy |
| `--conf` | 0.25 | Confidence threshold (lower values detect more items but may add false detections) |
| `--imgsz` | 640 | Inference size, lower is faster |

Keys: `q` quit, `m` mute/unmute.

## Troubleshooting
- Window never opens: open the stream URL in a PC browser to confirm it works; check the firewall.
- Lag: lower `--imgsz` (e.g. 416) or use the `n` model; lower the resolution in IP Webcam settings.
- The app speaks newly detected item types and additional items, then repeats the visible scene when the previous announcement finishes. Announcements are queued so new detections are not replaced by older ones. A detection that disappears for more than 0.75 seconds is treated as new if it reappears. The default `--conf` is 0.25 to help detect more items; lower thresholds can also add false detections.
- If you cannot hear speech, check Windows volume, the selected output device, and the app's Volume Mixer settings.
- No voice on Linux: install `espeak-ng`.
