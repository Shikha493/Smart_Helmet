import os
import time
import queue
import random
import threading
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import pyttsx3
    _HAS_TTS = True
except ImportError:
    _HAS_TTS = False


# ─────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────
DISPLAY_W = 1280
DISPLAY_H = 720

# Model weights. "n" (nano) is fastest but least accurate — it's the main
# reason small/thin objects (fork, knife, spoon, scissors) were missed.
# "s" (small) is a big accuracy jump for a modest speed cost. Go to "m" if
# your machine can handle it and you want even better recognition.
MODEL_WEIGHTS = "yolov8s-seg.pt"

# Inference image size YOLO letterboxes internally (keeps aspect ratio,
# unlike the old manual cv2.resize to 640x360 which distorted every object).
IMGSZ = 640

# Detection thresholds
CONF_THRES = 0.35
IOU_THRES = 0.50

# Test-Time Augmentation ("augmentation" at inference time). When True,
# Ultralytics runs the frame through multiple augmented views (flips /
# scales) and merges the predictions — meaningfully more accurate,
# especially on objects it was previously missing, at ~2-3x slower
# inference. Turn off if your FPS drops too much on CPU.
USE_TTA = False

# Use GPU automatically if available
DEVICE = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"

# Ask Ultralytics for masks already aligned to the ORIGINAL frame
# resolution (instead of the low-res mask prototype), so we don't need to
# manually rescale/guess mask coordinates ourselves.
RETINA_MASKS = True

# ── Paths relative to this script's location ──────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# If you have a real photo for a class, drop it in /assets and point to it
# here — it will always be preferred. Anything NOT listed (or whose file is
# missing) automatically gets a generated, visually unique placeholder
# texture per label the first time it's seen (see _generate_placeholder),
# so every object gets its own distinct image instead of several objects
# sharing the same picture.
OBJECT_IMAGES = {
    # "bottle": os.path.join(ASSETS_DIR, "bottle.jpg"),
    # add your own real photos here as you collect them
}

CATEGORY_GRID_COLOR = {
    "PLASTIC": (30,  200,  80),
    "ORGANIC": (0,   200, 230),
    "GLASS":   (20,  120, 255),
    "E-WASTE": (220,  60,   0),
    "PAPER":   (180,  20, 220),
    "METAL":   (0,   220, 200),
    "TEXTILE": (220, 160,   0),
}

# ── Confidence smoothing ──────────────────────────
SMOOTH_FRAMES = 3

# ── Voice announcements ───────────────────────────
VOICE_ENABLED = True
VOICE_COOLDOWN_SEC = 6.0   # don't repeat the same object's name more often than this


# ─────────────────────────────────────────────────
#  WASTE MAP
# ─────────────────────────────────────────────────
WASTE_MAP = {
    "bottle":     "PLASTIC",
    "cup":        "PLASTIC",
    "bowl":       "PLASTIC",
    "fork":       "PLASTIC",
    "knife":      "PLASTIC",
    "spoon":      "PLASTIC",
    "banana":     "ORGANIC",
    "apple":      "ORGANIC",
    "orange":     "ORGANIC",
    "sandwich":   "ORGANIC",
    "hot dog":    "ORGANIC",
    "pizza":      "ORGANIC",
    "donut":      "ORGANIC",
    "cake":       "ORGANIC",
    "wine glass": "GLASS",
    "cell phone": "E-WASTE",
    "laptop":     "E-WASTE",
    "keyboard":   "E-WASTE",
    "mouse":      "E-WASTE",
    "remote":     "E-WASTE",
    "tv":         "E-WASTE",
    "book":       "PAPER",
    "scissors":   "METAL",
    "suitcase":   "TEXTILE",
    "handbag":    "TEXTILE",
    "backpack":   "TEXTILE",
    "umbrella":   "TEXTILE",
}


# ─────────────────────────────────────────────────
#  UNIQUE PLACEHOLDER TEXTURE GENERATOR
# ─────────────────────────────────────────────────
def _generate_placeholder(label, category):
    """
    Builds a unique, deterministic texture image for a given object label
    (not just its category) so every class gets its own look even if you
    haven't supplied a real photo yet. Saved to disk so it's reused next run.
    """
    color_bgr = CATEGORY_GRID_COLOR.get(category, (200, 200, 200))
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    size = 400
    img = Image.new("RGB", (size, size), color_rgb)
    draw = ImageDraw.Draw(img)

    # Deterministic per-label pattern so "apple" always looks different
    # from "orange" even though both are ORGANIC.
    rng = random.Random(label)
    for _ in range(50):
        x1, y1, x2, y2 = (rng.randint(0, size) for _ in range(4))
        shade = tuple(max(0, min(255, c + rng.randint(-50, 50))) for c in color_rgb)
        draw.line([x1, y1, x2, y2], fill=shade, width=rng.randint(2, 7))

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    text = label.upper()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, size - th - 24, size, size], fill=(0, 0, 0))
    draw.text(((size - tw) // 2, size - th - 14), text, fill=(255, 255, 255), font=font)

    out_path = os.path.join(ASSETS_DIR, f"placeholder_{label.replace(' ', '_')}.png")
    try:
        img.save(out_path)
    except Exception:
        pass

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────
#  IMAGE CACHE
# ─────────────────────────────────────────────────
_img_cache: dict = {}

def _load_object_image(label, category):
    if label in _img_cache:
        return _img_cache[label]

    path = OBJECT_IMAGES.get(label)
    img = None

    if path and os.path.isfile(path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None and _HAS_PIL:
            try:
                pil_img = Image.open(path).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[ERROR] Pillow also failed for '{label}' at {path}: {e}")

    if img is None and _HAS_PIL:
        # No real photo supplied (or it failed to load) — generate a
        # unique placeholder for this exact label instead of falling back
        # to a generic per-category grid.
        img = _generate_placeholder(label, category)

    if img is None:
        print(f"[WARN] No texture available for '{label}' — will draw grid fallback.")

    _img_cache[label] = img
    return img


# ─────────────────────────────────────────────────
#  VOICE ANNOUNCER
# ─────────────────────────────────────────────────
class Announcer:
    def __init__(self, cooldown=VOICE_COOLDOWN_SEC, enabled=VOICE_ENABLED):
        self.cooldown = cooldown
        self.enabled = enabled and _HAS_TTS
        self._last_said = {}
        self._q = queue.Queue()
        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        elif enabled and not _HAS_TTS:
            print("[WARN] pyttsx3 not installed — voice announcements disabled. "
                  "Install with: pip install pyttsx3")

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        while True:
            phrase = self._q.get()
            if phrase is None:
                break
            try:
                engine.say(phrase)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS ERROR] {e}")

    def announce(self, label, category):
        if not self.enabled:
            return
        now = time.time()
        if now - self._last_said.get(label, 0) < self.cooldown:
            return
        self._last_said[label] = now
        self._q.put(f"{label} detected. {category.title()} waste.")


# ─────────────────────────────────────────────────
#  LABEL SMOOTHER
# ─────────────────────────────────────────────────
class LabelSmoother:
    def __init__(self, window: int = SMOOTH_FRAMES):
        self.window = window
        self._history: dict = defaultdict(lambda: deque(maxlen=window))
        self._confirmed: dict = {}

    def _nearest_key(self, cx: int, cy: int, threshold: int = 80):
        best, best_d = None, threshold
        for (kx, ky) in list(self._history.keys()):
            d = abs(kx - cx) + abs(ky - cy)
            if d < best_d:
                best, best_d = (kx, ky), d
        return best

    def update(self, cx: int, cy: int, label: str, category: str):
        key = self._nearest_key(cx, cy) or (cx, cy)
        q = self._history[key]
        q.append((label, category))
        if len(q) == self.window and len(set(q)) == 1:
            self._confirmed[key] = (label, category)
        return self._confirmed.get(key)

    def purge_stale(self, active_keys):
        for k in list(self._history.keys()):
            kx, ky = k
            still_active = any(abs(kx - ax) + abs(ky - ay) < 80 for ax, ay in active_keys)
            if not still_active:
                self._history.pop(k, None)
                self._confirmed.pop(k, None)


# ─────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────
def _make_grid_layer(h, w, x1, y1, x2, y2, color, cell=14):
    layer = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(layer, (x1, y1), (x2, y2), tuple(c // 6 for c in color), -1)
    for gx in range(x1, x2, cell):
        cv2.line(layer, (gx, y1), (gx, y2), color, 1)
    for gy in range(y1, y2, cell):
        cv2.line(layer, (x1, gy), (x2, gy), color, 1)
    return layer


def _apply_image_to_mask(frame, mask_bool, tex):
    out = frame.copy()
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not rows.any() or not cols.any():
        return out

    y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    rh, rw = y2 - y1 + 1, x2 - x1 + 1

    tex_resized = cv2.resize(tex, (rw, rh), interpolation=cv2.INTER_AREA)
    region = out[y1:y2+1, x1:x2+1]
    mask_region = mask_bool[y1:y2+1, x1:x2+1, np.newaxis]
    out[y1:y2+1, x1:x2+1] = np.where(mask_region, tex_resized, region)
    return out


# ─────────────────────────────────────────────────
#  BUILD VIEW
# ─────────────────────────────────────────────────
def build_view(frame, detections, fps=None):
    h, w = frame.shape[:2]
    out = frame.copy()

    for det in detections:
        mask_bool = det["mask"] > 0.5
        category  = det["category"]
        label     = det["label"]
        conf      = det["conf"]
        x1, y1, x2, y2 = det["box"]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w - 1, x2), min(h - 1, y2)
        grid_color = CATEGORY_GRID_COLOR.get(category, (200, 200, 200))

        tex = _load_object_image(label, category)

        if tex is not None:
            out = _apply_image_to_mask(out, mask_bool, tex)
        else:
            grid = _make_grid_layer(h, w, x1c, y1c, x2c, y2c, grid_color)
            out[mask_bool] = np.where(grid[mask_bool] > 0, grid[mask_bool], out[mask_bool])

        outline_color = grid_color
        cv2.rectangle(out, (x1c, y1c), (x2c, y2c), outline_color, 2)

        bs = 14
        for (bx, by, ddx, ddy) in [(x1c, y1c, 1, 1), (x2c, y1c, -1, 1),
                                    (x1c, y2c, 1, -1), (x2c, y2c, -1, -1)]:
            cv2.line(out, (bx, by), (bx + ddx * bs, by), outline_color, 2)
            cv2.line(out, (bx, by), (bx, by + ddy * bs), outline_color, 2)

        mode_tag = "IMG" if tex is not None else "GRID"
        badge = f"{category}  {label}  {conf:.0%}  [{mode_tag}]"
        fs = 0.50
        (tw2, th2), bl = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        pad = 5
        lx2 = min(w - 1, x1c + tw2 + pad * 2)
        ly1 = max(0, y1c - th2 - bl - pad * 2)
        cv2.rectangle(out, (x1c, ly1), (lx2, y1c), (0, 0, 0), -1)
        cv2.rectangle(out, (x1c, ly1), (lx2, y1c), outline_color, 1)
        cv2.putText(out, badge, (x1c + pad, y1c - bl - pad // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, outline_color, 1, cv2.LINE_AA)

    out = cv2.resize(out, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
    if fps is not None:
        cv2.putText(out, f"FPS: {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
def main():
    print(f"Loading {MODEL_WEIGHTS} on device={DEVICE} (TTA={'on' if USE_TTA else 'off'})...")
    model = YOLO(MODEL_WEIGHTS)
    smoother = LabelSmoother(window=SMOOTH_FRAMES)
    announcer = Announcer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    win = "garbage"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISPLAY_W, DISPLAY_H)
    print("Running — press ESC to quit.")

    prev_t = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]

        # Feed the FULL, un-distorted frame straight to YOLO. Ultralytics
        # letterboxes internally (preserves aspect ratio) and automatically
        # rescales every box/mask back to this frame's own resolution — no
        # more manual scale_x/scale_y math and no more squashed small
        # objects like fork/knife/spoon/scissors.
        results = model(
            frame,
            imgsz=IMGSZ,
            conf=CONF_THRES,
            iou=IOU_THRES,
            augment=USE_TTA,
            retina_masks=RETINA_MASKS,
            device=DEVICE,
            verbose=False,
        )[0]

        raw_dets = []
        active_cxcy = []

        if results.boxes is not None:
            n_masks = len(results.masks.data) if results.masks is not None else 0

            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label not in WASTE_MAP:
                    continue

                category = WASTE_MAP[label]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                cx, cy = (x1i + x2i) // 2, (y1i + y2i) // 2
                active_cxcy.append((cx, cy))

                if i < n_masks:
                    raw_mask = results.masks.data[i].cpu().numpy()
                    if raw_mask.shape != (fh, fw):
                        raw_mask = cv2.resize(raw_mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
                else:
                    raw_mask = np.zeros((fh, fw), dtype=np.float32)

                raw_dets.append({
                    "label": label, "category": category,
                    "box": (x1i, y1i, x2i, y2i), "conf": conf,
                    "mask": raw_mask, "cx": cx, "cy": cy,
                })

        smoother.purge_stale(active_cxcy)

        detections = []
        for d in raw_dets:
            confirmed = smoother.update(d["cx"], d["cy"], d["label"], d["category"])
            if confirmed is not None:
                stable_label, stable_category = confirmed
                detections.append({**d, "label": stable_label, "category": stable_category})
                announcer.announce(stable_label, stable_category)

        now = time.time()
        fps = 1.0 / max(1e-6, now - prev_t)
        prev_t = now

        cv2.imshow(win, build_view(frame, detections, fps=fps))
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()