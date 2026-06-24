import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH_1 = "trafik isigi algilama.pt"
MODEL_PATH_2 = "trafik isareti algilama.pt"
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


def draw_panel(frame, stable_ids, class_names, panel_color, text_color, label_prefix):
    bar_h = 40 + 36 * max(len(stable_ids), 1)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_h), panel_color, -1)
    if stable_ids:
        for i, cid in enumerate(stable_ids):
            label_text = f"{label_prefix}: {class_names[cid]}"
            y = 40 + i * 36
            cv2.putText(frame, label_text, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"{label_prefix}: -", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 2, cv2.LINE_AA)


def main():
    print(f"Model 1 yükleniyor: {MODEL_PATH_1}")
    model1 = YOLO(MODEL_PATH_1)
    print(f"Model 2 yükleniyor: {MODEL_PATH_2}")
    model2 = YOLO(MODEL_PATH_2)

    names1 = model1.names
    names2 = model2.names
    print(f"Model 1: {len(names1)} sınıf | Model 2: {len(names2)} sınıf")

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

    tracker1 = ClassTracker(HISTORY_LEN, STABLE_MIN_VOTES, STABLE_HOLD_SECONDS)
    tracker2 = ClassTracker(HISTORY_LEN, STABLE_MIN_VOTES, STABLE_HOLD_SECONDS)

    print("Başladı. Çıkmak için pencere üzerinde 'q' tuşuna basın.\n")

    try:
        while True:
            loop_start = time.time()

            ret, frame_bgr = cap.read()
            if not ret:
                print("Uyarı: kameradan kare okunamadı, tekrar deneniyor...")
                continue

            res1 = model1.predict(source=frame_bgr, imgsz=IMG_SIZE,
                                  conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
            res2 = model2.predict(source=frame_bgr, imgsz=IMG_SIZE,
                                  conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]

            annotated1 = res1.plot()
            annotated2 = res2.plot()

            now = time.time()

            tracker1.update([int(b.cls) for b in res1.boxes], now)
            tracker1.cleanup(now)
            tracker2.update([int(b.cls) for b in res2.boxes], now)
            tracker2.cleanup(now)

            stable1 = tracker1.get_stable_classes()[:MAX_DISPLAYED_LABELS]
            stable2 = tracker2.get_stable_classes()[:MAX_DISPLAYED_LABELS]

            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps
            prev_time = now

            draw_panel(annotated1, stable1, names1, (30, 30, 30), (0, 220, 0), "MODEL1")
            draw_panel(annotated2, stable2, names2, (20, 20, 60), (0, 180, 255), "MODEL2")

            if annotated1.shape[0] != annotated2.shape[0]:
                h = max(annotated1.shape[0], annotated2.shape[0])
                annotated1 = cv2.resize(annotated1, (int(annotated1.shape[1] * h / annotated1.shape[0]), h))
                annotated2 = cv2.resize(annotated2, (int(annotated2.shape[1] * h / annotated2.shape[0]), h))

            combined = np.hstack([annotated1, annotated2])

            cv2.putText(combined, f"FPS: {fps_smooth:.1f}",
                        (combined.shape[1] - 160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            n1 = [names1[c] for c in stable1] if stable1 else ["-"]
            n2 = [names2[c] for c in stable2] if stable2 else ["-"]
            print(f"\rM1: {', '.join(n1):40s} | M2: {', '.join(n2):40s}", end="", flush=True)

            cv2.imshow(WINDOW_NAME, combined)

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