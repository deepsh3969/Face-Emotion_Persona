import math
from collections import deque
import numpy as np


LABELS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral", "uncertain"]
COLORS = {
    "happy": (60, 230, 120),
    "sad": (220, 120, 255),
    "angry": (80, 90, 255),
    "fear": (190, 90, 255),
    "surprise": (80, 230, 255),
    "disgust": (100, 190, 110),
    "neutral": (210, 220, 230),
    "uncertain": (150, 170, 190),
}
PERSONAS = {
    "happy": "OPTIMIST", "sad": "DREAMER", "angry": "WARRIOR",
    "fear": "COURAGE SEEKER", "surprise": "EXPLORER",
    "disgust": "CRITIC", "neutral": "OBSERVER", "uncertain": "MYSTERY",
}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _safe(landmarks, i):
    return landmarks[i] if 0 <= i < len(landmarks) else (0, 0)


def extract_features(lm):
    """Scale-invariant facial geometry features using MediaPipe landmarks."""
    if len(lm) < 468:
        return None

    # Face scale: cheek-to-cheek. All ratios below are dimensionless.
    face_w = max(_dist(_safe(lm, 234), _safe(lm, 454)), 1.0)
    face_h = max(_dist(_safe(lm, 10), _safe(lm, 152)), 1.0)

    left_eye_h = (_dist(_safe(lm, 159), _safe(lm, 145)) + _dist(_safe(lm, 158), _safe(lm, 153))) / 2
    right_eye_h = (_dist(_safe(lm, 386), _safe(lm, 374)) + _dist(_safe(lm, 385), _safe(lm, 380))) / 2
    eye_open = ((left_eye_h + right_eye_h) / 2) / face_w

    mouth_w = _dist(_safe(lm, 61), _safe(lm, 291)) / face_w
    mouth_h = _dist(_safe(lm, 13), _safe(lm, 14)) / face_h
    mouth_open = mouth_h / max(mouth_w, 0.001)

    left_brow = _dist(_safe(lm, 105), _safe(lm, 159)) / face_h
    right_brow = _dist(_safe(lm, 334), _safe(lm, 386)) / face_h
    brow_eye = (left_brow + right_brow) / 2

    # Smile/corner lift: positive when mouth corners sit above the mouth center.
    center_y = (_safe(lm, 13)[1] + _safe(lm, 14)[1]) / 2
    corner_y = (_safe(lm, 61)[1] + _safe(lm, 291)[1]) / 2
    smile_lift = (center_y - corner_y) / face_h

    # Brow compression proxy for concentrated/angry appearance.
    brow_mid_y = (_safe(lm, 70)[1] + _safe(lm, 300)[1]) / 2
    eye_mid_y = (_safe(lm, 159)[1] + _safe(lm, 386)[1]) / 2
    brow_compress = (eye_mid_y - brow_mid_y) / face_h

    return {
        "eye_open": float(eye_open),
        "mouth_w": float(mouth_w),
        "mouth_open": float(mouth_open),
        "brow_eye": float(brow_eye),
        "smile_lift": float(smile_lift),
        "brow_compress": float(brow_compress),
    }


def _score(f):
    # These are expression estimates, not measurements of internal emotion.
    s = {k: 0.05 for k in LABELS}
    eye = f["eye_open"]
    mouth = f["mouth_w"]
    open_ratio = f["mouth_open"]
    lift = f["smile_lift"]
    brow = f["brow_compress"]
    brow_eye = f["brow_eye"]

    s["happy"] += 3.0 * max(0, lift - 0.015) + 1.4 * max(0, mouth - 0.34)
    s["sad"] += 2.0 * max(0, -lift - 0.005) + 0.8 * max(0, 0.28 - eye)
    s["angry"] += 2.4 * max(0, brow - 0.018) + 0.9 * max(0, 0.18 - brow_eye)
    s["surprise"] += 2.5 * max(0, eye - 0.12) + 2.0 * max(0, open_ratio - 0.18)
    s["fear"] += 1.4 * max(0, eye - 0.105) + 1.0 * max(0, open_ratio - 0.12) + 0.6 * max(0, -lift)
    s["disgust"] += 1.6 * max(0, 0.34 - mouth) + 1.0 * max(0, brow - 0.01) + 0.5 * max(0, -lift)

    # Neutral gets stronger when no feature is strongly expressive.
    expression_energy = max(s[k] for k in LABELS if k != "neutral")
    s["neutral"] += max(0, 1.15 - expression_energy)
    return s


def _softmax(scores):
    vals = np.array([scores[k] for k in LABELS], dtype=np.float32)
    vals -= np.max(vals)
    e = np.exp(vals * 1.5)
    p = e / max(float(e.sum()), 1e-8)
    return {k: float(v) for k, v in zip(LABELS, p)}


class EmotionModel:
    """Fast, dependency-light facial-expression estimator with temporal stability."""

    def __init__(self, mode="image", use_onnx=False, smoothing=0.72, history_size=12):
        self.mode = mode
        self.use_onnx = use_onnx
        self.smoothing_weight = float(np.clip(smoothing, 0.0, 0.95))
        self.history = deque(maxlen=history_size)
        self.prob_history = deque(maxlen=history_size)
        self.last = self._empty()
        self.frame_count = 0

    def _empty(self):
        return {"emotion": "neutral", "persona": "OBSERVER", "color": COLORS["neutral"],
                "confidence": 0.0, "bbox": None, "probabilities": {},
                "stable": False, "expression_energy": 0.0, "backend": "HEURISTIC"}

    def set_smoothing(self, weight):
        self.smoothing_weight = float(np.clip(weight, 0.0, 0.95))

    def reset(self):
        self.history.clear(); self.prob_history.clear(); self.last = self._empty(); self.frame_count = 0

    def update(self, faces, frame):
        self.frame_count += 1
        if not faces:
            self.last = self._empty()
            return self.last

        face = faces[0]
        feats = extract_features(face["landmarks"])
        if feats is None:
            return self.last

        probs = _softmax(_score(feats))
        if self.prob_history:
            prev = self.prob_history[-1]
            a = 1.0 - self.smoothing_weight
            probs = {k: a * probs[k] + self.smoothing_weight * prev.get(k, 0.0) for k in LABELS}
            total = sum(probs.values()) or 1.0
            probs = {k: v / total for k, v in probs.items()}
        self.prob_history.append(probs)

        ordered = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        label, top = ordered[0]
        second = ordered[1][1]
        confidence = float(np.clip(0.55 * top + 0.45 * (top - second) * 2.0, 0.0, 1.0))

        # Avoid overconfident labels when the classifier is ambiguous.
        if top < 0.31 or (top - second) < 0.055:
            label = "uncertain"
            confidence = min(confidence, 0.52)

        self.history.append(label)
        stable = len(self.history) >= 5 and len(set(list(self.history)[-5:])) == 1
        if len(self.history) >= 3 and not stable:
            # weighted majority over recent predictions
            recent = list(self.history)[-5:]
            counts = {k: recent.count(k) for k in set(recent)}
            label = max(counts, key=counts.get)

        energy = float(np.clip(1.0 - probs.get("neutral", 0.0), 0.0, 1.0))
        result = {
            "emotion": label,
            "persona": PERSONAS.get(label, "OBSERVER"),
            "color": COLORS.get(label, COLORS["neutral"]),
            "confidence": confidence,
            "bbox": face["bbox"],
            "probabilities": probs,
            "stable": stable,
            "expression_energy": energy,
            "backend": "HEURISTIC",
            "features": feats,
        }
        self.last = result
        return result
