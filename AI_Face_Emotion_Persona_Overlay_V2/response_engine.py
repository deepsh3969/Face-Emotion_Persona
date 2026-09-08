import random
import time


MESSAGES = {
    "happy": ["That energy is showing. Keep it going!", "You're bringing great energy today.", "That smile just upgraded the room."],
    "sad": ["Take it one step at a time. You've got this.", "A slower moment is still progress.", "Be kind to yourself and keep moving forward."],
    "angry": ["Pause. Breathe. Then choose your next move.", "Use that energy as fuel, not friction.", "Reset for a second — you've got control of the next step."],
    "fear": ["Courage can look like taking one small step.", "Breathe slowly. You don't need to solve everything at once.", "Keep going at your own pace."],
    "surprise": ["Plot twist detected. Stay curious!", "Something caught your attention — explore it.", "Unexpected moment. Make it useful."],
    "disgust": ["Not your favorite moment. Reset and refocus.", "You can change direction. Keep your standards high.", "Shake it off and focus on what matters."],
    "neutral": ["Steady mode. Keep building.", "Calm focus detected. Nice.", "One step at a time — stay in the flow."],
    "uncertain": ["Signal is unclear. Let's take a breath and continue.", "Reading the room... stay relaxed.", "Expression signal is mixed — no pressure to label it."],
}


class ResponseEngine:
    def __init__(self, cooldown=5.0, min_confidence=0.48):
        self.cooldown = cooldown
        self.min_confidence = min_confidence
        self.last_emotion = None
        self.last_time = 0.0
        self.message = "SYSTEM READY — I'm watching the expression signal."
        self.message_until = 0.0

    def reset(self):
        self.last_emotion = None; self.last_time = 0.0
        self.message = "SESSION RESET — Fresh start."; self.message_until = time.time() + 3

    def update(self, result, force=False):
        now = time.time()
        emotion = result.get("emotion", "neutral")
        conf = result.get("confidence", 0.0)
        stable = result.get("stable", False)
        changed = emotion != self.last_emotion
        if force or (changed and stable and conf >= self.min_confidence) or (self.last_emotion is None and conf >= self.min_confidence):
            if force or now - self.last_time >= self.cooldown:
                self.message = random.choice(MESSAGES.get(emotion, MESSAGES["neutral"]))
                self.message_until = now + 5.0
                self.last_emotion = emotion
                self.last_time = now
        return self.message

    def active_message(self):
        if time.time() <= self.message_until:
            return self.message
        return "AI SUPPORT // " + self.message
