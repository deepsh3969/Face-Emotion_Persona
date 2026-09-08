import os
import time
import cv2
import numpy as np


def save_screenshot(img, save_dir="screenshots"):
    os.makedirs(save_dir, exist_ok=True)
    name = time.strftime("v2_%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}.png"
    path = os.path.join(save_dir, name)
    cv2.imwrite(path, img)
    try:
        import winsound
        winsound.Beep(1050, 55)
    except Exception:
        pass
    return path


def _box(img, x1, y1, x2, y2, color, alpha=0.78):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (7, 11, 19), -1)
    img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


def _text(img, text, xy, scale=0.55, color=(225, 235, 245), thick=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, width, height, value, color, bg=(38, 46, 57)):
    value = max(0.0, min(1.0, float(value)))
    cv2.rectangle(img, (x, y), (x + width, y + height), bg, -1)
    cv2.rectangle(img, (x, y), (x + int(width * value), y + height), color, -1)


def _fmt_time(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class HUD:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.fps = 0.0
        self._frames = 0
        self._fps_t = time.time()
        self.last_screenshot = 0.0
        self.pulse = 0.0

    def update_fps(self):
        self._frames += 1
        now = time.time()
        if now - self._fps_t >= 0.5:
            self.fps = self._frames / (now - self._fps_t)
            self._frames = 0
            self._fps_t = now

    def _draw_score_panel(self, img, stats, color):
        h, w = img.shape[:2]
        px = max(16, w - 342)
        py = 210
        pw = min(326, w - 32)
        ph = min(315, h - py - 105)
        _box(img, px, py, px + pw, py + ph, color)
        _text(img, "CUMULATIVE SCORE MATRIX", (px + 14, py + 23), .47, color, 2)
        _text(img, "AVERAGE OF ALL DETECTED FACE SAMPLES", (px + 14, py + 42), .31, (165, 180, 195), 1)

        overall = stats.get("overall", 50.0)
        _text(img, f"OVERALL EXPRESSION SCORE   {overall:05.1f}/100", (px + 14, py + 69), .48, (240, 245, 250), 2)
        _bar(img, px + 14, py + 78, pw - 28, 9, overall / 100.0, color)

        probs = stats.get("average_probabilities", {})
        labels = [
            ("HAPPY", "happy"), ("SAD", "sad"), ("ANGRY", "angry"),
            ("FEAR", "fear"), ("SURPRISE", "surprise"), ("DISGUST", "disgust"),
            ("NEUTRAL", "neutral"), ("UNCERTAIN", "uncertain")
        ]
        start_y = py + 106
        for i, (name, key) in enumerate(labels):
            col = 0 if i < 4 else 1
            row = i if i < 4 else i - 4
            x = px + 14 + col * (pw // 2)
            y = start_y + row * 38
            val = float(probs.get(key, 0.0))
            _text(img, f"{name:<9} {val * 100:5.1f}%", (x, y), .36, (220, 230, 240), 1)
            _bar(img, x, y + 6, (pw // 2) - 30, 5, val, color)

        bottom_y = py + ph - 28
        _text(img, f"SAMPLES {stats.get('samples', 0):,}   |   CONF {stats.get('avg_confidence', 0)*100:04.1f}%", (px + 14, bottom_y), .34, (170, 185, 200), 1)

    def _draw_milestones(self, img, stats, color):
        h, w = img.shape[:2]
        px = 16
        py = 14
        pw = min(355, w - 380)
        ph = 190
        _box(img, px, py, px + pw, py + ph, color)
        _text(img, "OBSERVATION MILESTONES", (px + 14, py + 23), .48, color, 2)
        _text(img, "CUMULATIVE • NOT SINGLE-FRAME", (px + 14, py + 42), .31, (165, 180, 195), 1)

        milestones = [(120.0, "2 MIN"), (300.0, "5 MIN"), (600.0, "10 MIN"), (6000.0, "100 MIN")]
        stored = stats.get("milestones", {})
        next_m = stats.get("next_milestone")
        elapsed = stats.get("elapsed", 0.0)
        for i, (seconds, label) in enumerate(milestones):
            y = py + 65 + i * 29
            if seconds in stored:
                item = stored[seconds]
                txt = f"{label:<6}  {item['overall']:05.1f}  {item['dominant'].upper():<9}  ✓"
                c = color
            else:
                remaining = max(0, seconds - elapsed)
                txt = f"{label:<6}  --.--  {(_fmt_time(remaining) if next_m == seconds else 'LOCKED'):<9}  ○"
                c = (150, 165, 180)
            _text(img, txt, (px + 14, y), .39, c, 1)

        if next_m:
            remaining = max(0, next_m - elapsed)
            _text(img, f"NEXT REPORT IN  {_fmt_time(remaining)}", (px + 14, py + ph - 12), .38, color, 2)
        else:
            _text(img, "ALL 4 LONG-TERM REPORTS COMPLETE", (px + 14, py + ph - 12), .34, color, 2)

    def draw(self, img, result, stats, response, backend="HEURISTIC", smoothing=0.72, breathing=None):
        self.update_fps()
        h, w = img.shape[:2]
        self.width, self.height = w, h
        color = tuple(map(int, result.get("color", (210, 220, 230))))
        bbox = result.get("bbox")
        emotion = result.get("emotion", "neutral").upper()
        persona = result.get("persona", "OBSERVER")
        conf = result.get("confidence", 0.0)

        # Face tracking brackets
        if bbox:
            x, y, bw, bh = bbox
            x = max(4, x); y = max(28, y); x2 = min(w - 4, x + bw); y2 = min(h - 4, y + bh); L = 22
            corners = [
                ((x, y), (x + L, y)), ((x, y), (x, y + L)),
                ((x2, y), (x2 - L, y)), ((x2, y), (x2, y + L)),
                ((x, y2), (x + L, y2)), ((x, y2), (x, y2 - L)),
                ((x2, y2), (x2 - L, y2)), ((x2, y2), (x2, y2 - L)),
            ]
            for p, q in corners:
                cv2.line(img, p, q, color, 2, cv2.LINE_AA)
            _text(img, f"EXPRESSION // {emotion}", (x, max(20, y - 10)), .62, color, 2)
            _text(img, f"PERSONA // {persona}", (x, min(h - 8, y2 + 22)), .45, color, 1)

        # Top-right live status
        px = max(390, w - 300)
        _box(img, px, 14, w - 16, 98, color)
        _text(img, "LIVE ANALYSIS", (px + 14, 37), .50, color, 2)
        _text(img, f"● {self.fps:04.1f} FPS   {backend}", (px + 14, 61), .39, (210, 220, 230), 1)
        _text(img, f"SESSION { _fmt_time(stats.get('elapsed', 0)) }   FACE SAMPLES {stats.get('samples', 0):,}", (px + 14, 82), .34, (170, 185, 200), 1)

        # Milestones + cumulative score matrix
        self._draw_milestones(img, stats, color)
        self._draw_score_panel(img, stats, color)

        # Live signal strip
        live_y = max(108, h - 170)
        _box(img, 16, live_y, min(w - 360, 430), live_y + 66, color)
        _text(img, f"LIVE SIGNAL   {emotion}", (30, live_y + 23), .44, color, 2)
        _text(img, f"FRAME CONFIDENCE   {conf * 100:05.1f}%", (30, live_y + 45), .36, (215, 225, 235), 1)
        _bar(img, 30, live_y + 51, 280, 6, conf, color)

        # Support message
        msg_y = h - 92
        _box(img, 16, msg_y, min(w - 16, max(440, w - 350)), h - 16, color)
        _text(img, "AI SUPPORT", (30, msg_y + 23), .43, color, 2)
        text = response or "Keep going — the system is building your cumulative expression profile."
        max_chars = 66
        if len(text) > max_chars:
            text = text[:max_chars - 1] + "…"
        _text(img, text, (30, msg_y + 50), .46, (235, 240, 245), 1)
        _text(img, "S screenshot   B breathing   R reset   H support   +/- smoothing   ESC exit", (30, h - 28), .35, (150, 165, 180), 1)

        # Subtle scanline
        sy = int((time.time() * 55) % h)
        cv2.line(img, (0, sy), (w, sy), (80, 210, 220), 1, cv2.LINE_AA)

        if breathing and breathing.get("active"):
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (4, 8, 14), -1)
            img[:] = cv2.addWeighted(overlay, .56, img, .44, 0)
            _text(img, "RESET PROTOCOL", (w // 2 - 120, h // 2 - 50), .8, color, 2)
            _text(img, breathing.get("phase", "BREATHE"), (w // 2 - 80, h // 2 + 5), .75, (240, 245, 250), 2)
            _text(img, breathing.get("instruction", "slow down"), (w // 2 - 120, h // 2 + 40), .5, color, 1)
            cv2.circle(img, (w // 2, h // 2 + 110), int(38 + 20 * breathing.get("progress", 0)), color, 2, cv2.LINE_AA)
        return img
