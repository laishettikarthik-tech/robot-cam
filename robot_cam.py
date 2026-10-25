"""
robot_cam.py - capture photos + sensors + face detection + shape detection.

Run:  python robot_cam.py                  (S = save, Q = quit)
      python robot_cam.py --auto 2
      python robot_cam.py --only-detect    (auto-save only when a face/shape is seen)
      python robot_cam.py --no-faces / --no-shapes
      python robot_cam.py --serial /dev/tty.usbmodem1101

Live keys (click the camera window first):
  S save | Q quit | D debug view | P person-zone ignore on/off
  [ ]  min shape size -/+
  - =  edge sensitivity (lower = more edges)
  , .  color-difference threshold -/+ (lower = detects fainter objects)

Output: captures/<stamp>.jpg, <stamp>.json (sensors + faces + shapes), log.csv
"""
import argparse
import csv
import json
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np

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


class FaceDetector:
    def __init__(self):
        base = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        if self.face.empty():
            raise SystemExit("Could not load face detection model")

    def run(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = self.face.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        return [{"box": [int(x), int(y), int(w), int(h)]} for (x, y, w, h) in found]


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0


class ShapeDetector:
    def __init__(self):
        self.min_area = 1200
        self.canny_low = 40
        self.color_thr = 20
        self.ignore_person = True
        self.debug = None
        self.zones = []

    def person_zones(self, faces, w, h):
        """Region around + below each face (head, shoulders, body) to ignore."""
        zones = []
        for f in faces:
            fx, fy, fw, fh = f["box"]
            x0 = max(0, int(fx - 0.7 * fw))
            x1 = min(w, int(fx + fw + 0.7 * fw))
            y0 = max(0, int(fy - 0.3 * fh))
            zones.append((x0, y0, x1, h))
        return zones

    def classify(self, c):
        area_raw = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        harea = cv2.contourArea(hull)
        if harea == 0 or area_raw / harea < 0.80:
            return None  # ragged / irregular blob
        peri = cv2.arcLength(hull, True)
        if peri == 0:
            return None
        approx = cv2.approxPolyDP(hull, 0.03 * peri, True)
        n = len(approx)
        circ = 4 * math.pi * harea / (peri * peri)
        (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
        if rw == 0 or rh == 0:
            return None
        fill = harea / (rw * rh)
        ratio = min(rw, rh) / max(rw, rh)

        if n == 3:
            return "triangle", approx
        if n >= 7 and circ > 0.88:
            return "circle", approx
        if fill >= 0.86 and n <= 8:  # includes rounded rectangles
            return ("square" if ratio >= 0.88 else "rectangle"), approx
        if n == 5:
            return "pentagon", approx
        if n == 6:
            return "hexagon", approx
        if circ > 0.75:
            return "oval", approx
        return None

    def run(self, frame, faces):
        h, w = frame.shape[:2]

        # 1) edge map
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_low * 3)
        k5 = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k5, iterations=2)

        # 2) color-difference map (finds low-contrast objects, e.g. white on beige)
        lab = cv2.cvtColor(cv2.GaussianBlur(frame, (9, 9), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        med = np.median(lab.reshape(-1, 3), axis=0)
        dist = np.linalg.norm(lab - med, axis=2)
        cmask = (dist > self.color_thr).astype(np.uint8) * 255
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_OPEN, k5)
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, k5, iterations=2)

        # 3) blank out person zones
        self.zones = self.person_zones(faces, w, h) if self.ignore_person else []
        for (x0, y0, x1, y1) in self.zones:
            edges[y0:y1, x0:x1] = 0
            cmask[y0:y1, x0:x1] = 0

        self.debug = cv2.bitwise_or(edges, cmask)

        # 4) find + classify candidates from both maps
        max_area = 0.25 * w * h
        cands = []
        for m in (edges, cmask):
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area or area > max_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                if x <= 4 or y <= 4 or x + bw >= w - 4 or y + bh >= h - 4:
                    continue  # touches frame edge = cut off, skip
                res = self.classify(c)
                if res is None:
                    continue
                name, approx = res
                cands.append({
                    "shape": name,
                    "box": [int(x), int(y), int(bw), int(bh)],
                    "area": int(area),
                    "corners": int(len(approx)),
                    "_pts": approx,
                })

        # 5) remove duplicates (same object found by both maps)
        cands.sort(key=lambda s: s["area"], reverse=True)
        shapes = []
        for s in cands:
            if all(iou(s["box"], t["box"]) < 0.5 for t in shapes):
                shapes.append(s)
        return shapes


def draw(frame, faces, shapes, det):
    out = frame.copy()
    if det:
        for (x0, y0, x1, y1) in det.zones:
            cv2.rectangle(out, (x0, y0), (x1 - 2, y1 - 2), (120, 120, 120), 1)
    for f in faces:
        x, y, w, h = f["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, "face", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for s in shapes:
        cv2.drawContours(out, [s["_pts"]], -1, (255, 128, 0), 3)
        x, y, w, h = s["box"]
        cv2.putText(out, s["shape"], (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    cv2.putText(out, f"faces:{len(faces)} shapes:{len(shapes)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if det:
        cv2.putText(out, f"min_area:{det.min_area} edge:{det.canny_low} color:{det.color_thr} "
                         f"ignore_person:{det.ignore_person}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def clean_shapes(shapes):
    return [{k: v for k, v in s.items() if k != "_pts"} for s in shapes]


def save_capture(frame, sensors, faces, shapes):
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
        "shape_count": len(shapes),
        "shapes": clean_shapes(shapes),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log_path = os.path.join(OUT_DIR, "log.csv")
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "image", "face_count", "shapes", "sensors_json"])
        names = ";".join(s["shape"] for s in shapes)
        w.writerow([meta["timestamp"], meta["image"], len(faces), names, json.dumps(sensors)])

    print(f"saved {img_path}  faces={len(faces)} shapes={[s['shape'] for s in shapes]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--auto", type=float, default=0, help="auto-save every N seconds")
    ap.add_argument("--only-detect", action="store_true")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-shapes", action="store_true")
    ap.add_argument("--serial", type=str, default=None)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.cam}")

    sensors = Sensors(args.serial)
    face_det = None if args.no_faces else FaceDetector()
    shape_det = None if args.no_shapes else ShapeDetector()
    last_save = 0.0
    show_debug = False
    interval = args.auto if args.auto else (2 if args.only_detect else 0)
    print("Running. S=save Q=quit D=debug P=person-zone [ ]=size - ==edges , .=color"
          if not args.headless else "Headless. Ctrl+C to stop")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed")
                break

            faces = face_det.run(frame) if face_det else []
            shapes = shape_det.run(frame, faces) if shape_det else []

            now = time.time()
            if interval and now - last_save >= interval:
                if not args.only_detect or faces or shapes:
                    save_capture(frame, sensors.read(), faces, shapes)
                    last_save = now

            if not args.headless:
                cv2.imshow("robot_cam", draw(frame, faces, shapes, shape_det))
                if show_debug and shape_det is not None and shape_det.debug is not None:
                    cv2.imshow("debug (what the detector sees)", shape_det.debug)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    save_capture(frame, sensors.read(), faces, shapes)
                elif key == ord("q"):
                    break
                elif key == ord("d"):
                    show_debug = not show_debug
                    if not show_debug:
                        cv2.destroyWindow("debug (what the detector sees)")
                elif shape_det is not None:
                    if key == ord("["):
                        shape_det.min_area = max(300, shape_det.min_area - 300)
                    elif key == ord("]"):
                        shape_det.min_area += 300
                    elif key == ord("-"):
                        shape_det.canny_low = max(5, shape_det.canny_low - 5)
                    elif key == ord("="):
                        shape_det.canny_low += 5
                    elif key == ord(","):
                        shape_det.color_thr = max(6, shape_det.color_thr - 2)
                    elif key == ord("."):
                        shape_det.color_thr += 2
                    elif key == ord("p"):
                        shape_det.ignore_person = not shape_det.ignore_person
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
