"""
TEKNOFEST Robotaksi - Gerçek Zamanlı Webcam YOLO Tespiti (Stabilize Edilmiş v2)
==================================================================================
NE YAPAR?
  Webcam görüntüsünü gerçek zamanlı yakalar, YOLO modeline (best.pt) verir
  ve tespit edilen trafik işaretlerini kutu + etiket + güven skoru ile
  gösterir.

  KARARLI ETİKET (Temporal Smoothing) - v2: ÇOK TABELA DESTEĞİ
  v1'de sadece "en yüksek güvenli TEK tabela" oy alıyordu. Bu, aynı anda
  birden fazla tabela görününce (örn. DUR + Park Yeri birlikte) iki
  sınıfın güven skorlarının kare kare yer değiştirmesi yüzünden kararsız
  davranışa yol açıyordu (bir kare DUR kazanıyor, sonraki kare Park
  kazanıyor, vs.).

  v2'de HER SINIFIN KENDİ BAĞIMSIZ OY GEÇMİŞİ var. Bir karede görünen
  TÜM tespitler (sadece en güvenlisi değil) kendi sınıflarının geçmişine
  oy olarak eklenir. Böylece "DUR" ve "Park Yeri" birbirini ezmez; ikisi
  de kendi başına stabil hale gelip AYNI ANDA ekranda gösterilebilir.

GEREKSİNİMLER:
  pip install ultralytics opencv-python numpy

KULLANIM:
  python webcam_yolo_stable.py
  (Çıkmak için pencere üzerinde 'q' tuşuna basın)
"""

import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# ── AYARLAR - kendi ortamınıza göre düzenleyin ────────────────────────
MODEL_PATH = "v5.pt"
CAMERA_INDEX = 0           # Birden fazla kameranız varsa 0,1,2... deneyin
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

CONF_THRESH = 0.7           # Bu güven skorunun altındaki ham tespitler dikkate alınmaz
IOU_THRESH = 0.7
IMG_SIZE = 640
TARGET_FPS = 24             # Hedef işleme hızı (donanıma göre ayarlanabilir)

# --- STABİLİZASYON (kararlı etiket) ayarları ---
HISTORY_LEN = 15            # Oy penceresi: son kaç kare dikkate alınsın (sınıf başına)
STABLE_MIN_VOTES = 8        # Bir sınıfın "stabil" sayılması için pencerede en az kaç oy alması gerekir
STABLE_HOLD_SECONDS = 2.0   # Sınıf görünmeyi kestiğinde, eski stabil etiket kaç saniye ekranda tutulsun
MAX_DISPLAYED_LABELS = 4    # Ekranda aynı anda en fazla kaç stabil etiket gösterilsin (kalabalık olmasın)

WINDOW_NAME = "YOLO Webcam - Stabilize Tespit (q: cikis)"


class ClassTracker:
    """
    Her sınıf (class_id) için bağımsız bir oy geçmişi tutar.
    Bu sayede aynı anda görünen farklı tabelalar birbirinin oyunu
    bozmaz - her biri kendi "stabil mi, değil mi" durumuna sahiptir.
    """

    def __init__(self, history_len, min_votes, hold_seconds):
        self.history_len = history_len
        self.min_votes = min_votes
        self.hold_seconds = hold_seconds
        # class_id -> deque(0/1) son N karede görülüp görülmediği
        self.histories = {}
        # class_id -> son görülme zamanı (stabil durumdaysa hold için)
        self.last_seen = {}
        # class_id -> şu an stabil mi
        self.stable = {}

    def _ensure(self, class_id):
        if class_id not in self.histories:
            self.histories[class_id] = deque(maxlen=self.history_len)
            self.last_seen[class_id] = 0.0
            self.stable[class_id] = False

    def update(self, seen_class_ids, now):
        """
        seen_class_ids: bu karede tespit edilen TÜM sınıfların id listesi
        (sadece en güvenlisi değil - hepsi).
        """
        seen_set = set(seen_class_ids)

        # Bu karede görülen her sınıf için "1" oyu ekle
        for cid in seen_set:
            self._ensure(cid)
            self.histories[cid].append(1)
            self.last_seen[cid] = now

        # Daha önce takip edilen ama bu karede görülmeyen sınıflara "0" oyu ekle
        for cid in list(self.histories.keys()):
            if cid not in seen_set:
                self.histories[cid].append(0)

        # Stabil durumunu güncelle
        for cid, hist in self.histories.items():
            vote_count = sum(hist)
            if vote_count >= self.min_votes:
                self.stable[cid] = True
            elif (now - self.last_seen[cid]) > self.hold_seconds:
                # Hem oy desteği yetersiz hem de hold süresi de doldu -> stabil değil
                self.stable[cid] = False
            # Not: oy desteği düştüyse ama hold süresi dolmadıysa stabil durum
            # olduğu gibi (son haliyle) korunur - ani titreme/kayıp karelere dayanıklı.

    def get_stable_classes(self):
        """Şu an stabil sayılan sınıf id'lerini, en güçlü oydan en zayıfa sıralı döndürür."""
        result = [
            (cid, sum(self.histories[cid]))
            for cid, is_stable in self.stable.items() if is_stable
        ]
        result.sort(key=lambda x: -x[1])
        return [cid for cid, _ in result]

    def cleanup(self, now, max_age=10.0):
        """Çok uzun süredir görülmeyen sınıfları hafızadan temizle (bellek şişmesin)."""
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

            # ── YOLO tespiti ──
            results = model.predict(
                source=frame_bgr,
                imgsz=IMG_SIZE,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()  # ham kutu+etiket+skor (referans için)

            now = time.time()

            # ── Bu karede görünen TÜM sınıfları (sadece en güvenlisi değil)
            #     ilgili sınıfların bağımsız geçmişine oy olarak ekle ──
            seen_class_ids = [int(b.cls) for b in result.boxes]
            tracker.update(seen_class_ids, now)
            tracker.cleanup(now)

            stable_ids = tracker.get_stable_classes()[:MAX_DISPLAYED_LABELS]

            # ── FPS hesapla ──
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps
            prev_time = now

            # ── Üst bilgi çubuğu (stabil etiket(ler) + FPS) ──
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

            # ── Konsola da yaz (opsiyonel takip için) ──
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