"""
robot_cam.py - capture photos + sensor readings + face detection.

Run:  python robot_cam.py                  (S = save, Q = quit)
      python robot_cam.py --auto 2         (auto-save every 2 seconds)
      python robot_cam.py --only-faces     (auto-save only when a face is seen)
      python robot_cam.py --no-detect      (turn detection off)
      python robot_cam.py --serial /dev/tty.usbmodem1101

Output: captures/<stamp>.jpg, <stamp>.json (sensors + faces), log.csv
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
    def __init__(self, port=None, baud=115200):
        self.ser = None
        self.last = {}
        if port and serial:
            self.ser = serial.Serial(port, baud, timeout=0.05)
            time.sleep(2)

    def read(self):
        if self.ser:
            try:
                while self.ser.in_waiting:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    if line:
                        self.last = json.loads(line)
            except Exception as e:
                self.last = {"sensor_error": str(e)}
            return dict(self.last)
        return {"distance_cm": None, "temp_c": None, "note": "no sensor connected"}


class Detector:
    def __init__(self):
        base = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.eye = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
        if self.face.empty() or self.eye.empty():
            raise SystemExit("Could not load face/eye detection models")

    def run(self, frame):
        """Returns list of faces: {"box": [x,y,w,h], "eyes": n}"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        found = self.face.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        faces = []
        for (x, y, w, h) in found:
            roi = gray[y:y + h // 2, x:x + w]  # eyes are in the upper half
            eyes = self.eye.detectMultiScale(roi, 1.1, 6, minSize=(15, 15))
            faces.append({"box": [int(x), int(y), int(w), int(h)], "eyes": len(eyes)})
        return faces


def draw(frame, faces):
    out = frame.copy()
    for f in faces:
        x, y, w, h = f["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, "face", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    cv2.putText(out, f"faces: {len(faces)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def save_capture(frame, sensors, faces):
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
        "face_count": len(faces),
        "faces": faces,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log_path = os.path.join(OUT_DIR, "log.csv")
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "image", "face_count", "sensors_json"])
        w.writerow([meta["timestamp"], meta["image"], len(faces), json.dumps(sensors)])

    print(f"saved {img_path}  faces={len(faces)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--auto", type=float, default=0, help="auto-save every N seconds")
    ap.add_argument("--only-faces", action="store_true", help="auto-save only if a face is seen")
    ap.add_argument("--no-detect", action="store_true", help="disable detection")
    ap.add_argument("--serial", type=str, default=None)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.cam}")

    sensors = Sensors(args.serial)
    detector = None if args.no_detect else Detector()
    last_save = 0.0
    interval = args.auto if args.auto else (2 if args.only_faces else 0)
    print("Running. S = save, Q = quit" if not args.headless else "Headless. Ctrl+C to stop")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed")
                break

            faces = detector.run(frame) if detector else []

            now = time.time()
            if interval and now - last_save >= interval:
                if not args.only_faces or faces:
                    save_capture(frame, sensors.read(), faces)
                    last_save = now

            if not args.headless:
                cv2.imshow("robot_cam", draw(frame, faces))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    save_capture(frame, sensors.read(), faces)
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
