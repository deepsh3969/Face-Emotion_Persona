import time
from collections import Counter, defaultdict


MILESTONES = [120.0, 300.0, 600.0, 6000.0]  # 2, 5, 10, 100 minutes


class SessionTracker:
    """Accumulates every detected-face probability sample and creates cumulative milestone reports."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.started = time.time()
        self.last_time = self.started
        self.last_detected_time = self.started
        self.counts = Counter()
        self.duration = defaultdict(float)
        self.prob_sum = defaultdict(float)
        self.conf_sum = 0.0
        self.samples = 0
        self.detected_seconds = 0.0
        self.timeline = []
        self.milestones = {}

    @staticmethod
    def _minute_label(seconds):
        return {120.0: "2 MIN", 300.0: "5 MIN", 600.0: "10 MIN", 6000.0: "100 MIN"}.get(seconds, "MILESTONE")

    def update(self, result):
        now = time.time()
        dt = min(max(now - self.last_time, 0.0), 0.25)
        self.last_time = now

        # Only facial detections contribute to the scores.
        bbox = result.get("bbox")
        probs = result.get("probabilities") or {}
        if bbox is None or not probs:
            return

        emotion = result.get("emotion", "neutral")
        conf = float(result.get("confidence", 0.0))
        self.counts[emotion] += 1
        self.duration[emotion] += dt
        self.detected_seconds += dt
        self.conf_sum += conf
        self.samples += 1

        for label, probability in probs.items():
            self.prob_sum[label] += float(probability)

        if not self.timeline or self.timeline[-1][0] != emotion:
            self.timeline.append((emotion, now - self.started, conf))
            self.timeline = self.timeline[-30:]

        elapsed = now - self.started
        for milestone in MILESTONES:
            if elapsed >= milestone and milestone not in self.milestones:
                self.milestones[milestone] = self._make_milestone(milestone, elapsed)

    def _average_probabilities(self):
        if self.samples <= 0:
            return {}
        return {k: v / self.samples for k, v in self.prob_sum.items()}

    def _make_milestone(self, milestone_seconds, elapsed):
        probs = self._average_probabilities()
        labels = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral", "uncertain"]
        probs = {k: float(probs.get(k, 0.0)) for k in labels}

        # The overall score is derived from the average facial-expression signal,
        # not from one frame. Happy/surprise add positive signal; challenging
        # expressions subtract it; neutral/uncertain stay near the midpoint.
        positive = probs["happy"] + 0.65 * probs["surprise"]
        challenging = probs["sad"] + probs["fear"] + probs["angry"] + 0.55 * probs["disgust"]
        overall = 50.0 + 50.0 * (positive - challenging)
        overall = max(0.0, min(100.0, overall))

        dominant = max(probs, key=probs.get) if probs else "neutral"
        return {
            "label": self._minute_label(milestone_seconds),
            "minutes": milestone_seconds / 60.0,
            "elapsed": elapsed,
            "samples": self.samples,
            "overall": overall,
            "dominant": dominant,
            "avg_confidence": self.conf_sum / max(self.samples, 1),
            "probabilities": probs,
            "positive_ratio": probs["happy"],
        }

    def _current_report(self):
        # Current score is also cumulative average of all detected-face samples.
        probs = self._average_probabilities()
        labels = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral", "uncertain"]
        probs = {k: float(probs.get(k, 0.0)) for k in labels}
        positive = probs["happy"] + 0.65 * probs["surprise"]
        challenging = probs["sad"] + probs["fear"] + probs["angry"] + 0.55 * probs["disgust"]
        overall = max(0.0, min(100.0, 50.0 + 50.0 * (positive - challenging)))
        dominant = max(probs, key=probs.get) if probs else "neutral"
        return {
            "overall": overall,
            "dominant": dominant,
            "avg_confidence": self.conf_sum / max(self.samples, 1),
            "probabilities": probs,
            "samples": self.samples,
        }

    def stats(self):
        elapsed = time.time() - self.started
        positive = self.duration["happy"]
        challenging = self.duration["sad"] + self.duration["fear"] + self.duration["angry"]
        vibe = 50 + 50 * ((positive - challenging) / max(self.detected_seconds, 0.01))
        dominant = self.counts.most_common(1)[0][0] if self.counts else "neutral"
        current = self._current_report()

        next_milestone = next((m for m in MILESTONES if m not in self.milestones), None)
        return {
            "elapsed": elapsed,
            "dominant": dominant,
            "avg_confidence": self.conf_sum / max(self.samples, 1),
            "vibe": max(0.0, min(100.0, vibe)),
            "overall": current["overall"],
            "positive_ratio": current["probabilities"].get("happy", 0.0),
            "counts": dict(self.counts),
            "duration": dict(self.duration),
            "timeline": list(self.timeline),
            "samples": self.samples,
            "detected_seconds": self.detected_seconds,
            "average_probabilities": current["probabilities"],
            "milestones": dict(self.milestones),
            "next_milestone": next_milestone,
            "next_milestone_remaining": max(0.0, next_milestone - elapsed) if next_milestone else 0.0,
        }
