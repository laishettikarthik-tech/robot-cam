"""
robot_cam.py - capture photos + sensor readings and save them for later.

Install:   pip3 install opencv-python pyserial
Run:       python3 robot_cam.py                 (manual: press S to save, Q to quit)
           python3 robot_cam.py --auto 2        (auto-save every 2 seconds)
           python3 robot_cam.py --cam 1         (use second camera)
           python3 robot_cam.py --serial /dev/tty.usbmodem1101   (sensors from Arduino/ESP32)

Output:    captures/YYYYMMDD_HHMMSS_mmm.jpg
           captures/YYYYMMDD_HHMMSS_mmm.json   (timestamp + sensor values)
           captures/log.csv                     (one row per capture)

Serial sensor format, one line per reading, e.g.:
    {"distance_cm": 42.1, "temp_c": 29.5, "gas": 310}
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime

import cv2

try:
    import serial  # pyserial
except ImportError:
    serial = None

OUT_DIR = "captures"


class Sensors:
    """Reads sensor data. Falls back to placeholder values if no serial port given."""

    def __init__(self, port=None, baud=115200):
        self.ser = None
        self.last = {}
        if port and serial:
            self.ser = serial.Serial(port, baud, timeout=0.05)
            time.sleep(2)  # let the board reset

    def read(self):
        if self.ser:
            try:
                while self.ser.in_waiting:  # drain buffer, keep newest line
                    line = self.ser.readline().decode(errors="ignore").strip()
                    if line:
                        self.last = json.loads(line)
            except Exception as e:
                self.last = {"sensor_error": str(e)}
            return dict(self.last)
        return {"distance_cm": None, "temp_c": None, "note": "no sensor connected"}


def save_capture(frame, sensors):
    os.makedirs(OUT_DIR, exist_ok=True)
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    img_path = os.path.join(OUT_DIR, stamp + ".jpg")
    meta_path = os.path.join(OUT_DIR, stamp + ".json")

    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    meta = {
        "timestamp": now.isoformat(),
        "image": os.path.basename(img_path),
        "resolution": [frame.shape[1], frame.shape[0]],
        "sensors": sensors,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log_path = os.path.join(OUT_DIR, "log.csv")
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "image", "sensors_json"])
        w.writerow([meta["timestamp"], meta["image"], json.dumps(sensors)])

    print(f"saved {img_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--auto", type=float, default=0, help="auto-save every N seconds")
    ap.add_argument("--serial", type=str, default=None, help="serial port for sensors")
    ap.add_argument("--headless", action="store_true", help="no preview window (for robots)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.cam}")

    sensors = Sensors(args.serial)
    last_save = 0.0
    print("Running. S = save, Q = quit" if not args.headless else "Running headless. Ctrl+C to stop")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed")
                break

            now = time.time()
            if args.auto and now - last_save >= args.auto:
                save_capture(frame, sensors.read())
                last_save = now

            if not args.headless:
                preview = frame.copy()
                cv2.putText(preview, time.strftime("%H:%M:%S"), (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("robot_cam", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    save_capture(frame, sensors.read())
                elif key == ord("q"):
                    break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
