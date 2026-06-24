"""
GEREKSİNİMLER:
  pip install ultralytics opencv-python numpy
"""

import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "trafik isigi algilama.pt"
CAMERA_INDEX = 0          
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

CONF_THRESH = 0.7         
IOU_THRESH = 0.7
IMG_SIZE = 640
TARGET_FPS = 30           

HISTORY_LEN = 15           
STABLE_MIN_VOTES = 8       
STABLE_HOLD_SECONDS = 2.0  
MAX_DISPLAYED_LABELS = 10  

WINDOW_NAME = "q ile çık"


class ClassTracker:
    def __init__(self, history_len, min_votes, hold_seconds):
        self.history_len = history_len
        self.min_votes = min_votes
        self.hold_seconds = hold_seconds
        self.histories = {}
        self.last_seen = {}
        self.stable = {}

    def _ensure(self, class_id):
        if class_id not in self.histories:
            self.histories[class_id] = deque(maxlen=self.history_len)
            self.last_seen[class_id] = 0.0
            self.stable[class_id] = False

    def update(self, seen_class_ids, now):
        seen_set = set(seen_class_ids)

        for cid in seen_set:
            self._ensure(cid)
            self.histories[cid].append(1)
            self.last_seen[cid] = now

        for cid in list(self.histories.keys()):
            if cid not in seen_set:
                self.histories[cid].append(0)

        for cid, hist in self.histories.items():
            vote_count = sum(hist)
            if vote_count >= self.min_votes:
                self.stable[cid] = True
            elif (now - self.last_seen[cid]) > self.hold_seconds:
                self.stable[cid] = False

    def get_stable_classes(self):
        result = [
            (cid, sum(self.histories[cid]))
            for cid, is_stable in self.stable.items() if is_stable
        ]
        result.sort(key=lambda x: -x[1])
        return [cid for cid, _ in result]

    def cleanup(self, now, max_age=10.0):
        to_remove = [cid for cid, t in self.last_seen.items() if (now - t) > max_age]
        for cid in to_remove:
            self.histories.pop(cid, None)
            self.last_seen.pop(cid, None)
            self.stable.pop(cid, None)


def main():
    print(f"Model yükleniyor: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print(f"✓ Model yüklendi. {len(class_names)} sınıf tanımlı.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"HATA: Kamera açılamadı (index={CAMERA_INDEX}). "
              f"CAMERA_INDEX değerini değiştirip tekrar deneyin.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_interval = 1.0 / TARGET_FPS
    prev_time = time.time()
    fps_smooth = TARGET_FPS

    tracker = ClassTracker(HISTORY_LEN, STABLE_MIN_VOTES, STABLE_HOLD_SECONDS)

    print("Başladı. Çıkmak için pencere üzerinde 'q' tuşuna basın.\n")

    try:
        while True:
            loop_start = time.time()

            ret, frame_bgr = cap.read()
            if not ret:
                print("Uyarı: kameradan kare okunamadı, tekrar deneniyor...")
                continue

            results = model.predict(
                source=frame_bgr,
                imgsz=IMG_SIZE,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()

            now = time.time()

            seen_class_ids = [int(b.cls) for b in result.boxes]
            tracker.update(seen_class_ids, now)
            tracker.cleanup(now)

            stable_ids = tracker.get_stable_classes()[:MAX_DISPLAYED_LABELS]

            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps
            prev_time = now

            bar_h = 40 + 36 * max(len(stable_ids), 1)
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], bar_h), (30, 30, 30), -1)

            if stable_ids:
                for i, cid in enumerate(stable_ids):
                    label_text = f"TESPIT: {class_names[cid]}"
                    y = 40 + i * 36
                    cv2.putText(annotated, label_text, (15, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(annotated, "TESPIT: -", (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 2, cv2.LINE_AA)

            cv2.putText(annotated, f"FPS: {fps_smooth:.1f}",
                        (annotated.shape[1] - 160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            stable_names = [class_names[cid] for cid in stable_ids] if stable_ids else ["-"]
            print(f"\rStabil: {', '.join(stable_names):60s}", end="", flush=True)

            cv2.imshow(WINDOW_NAME, annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n\nKapatıldı.")


if __name__ == '__main__':
    main()