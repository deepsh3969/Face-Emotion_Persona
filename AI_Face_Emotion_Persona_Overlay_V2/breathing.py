import time


class Breathing:
    def __init__(self):
        self.active = False; self.started = 0.0; self.duration = 18.0

    def start(self):
        self.active = True; self.started = time.time()

    def update(self):
        if not self.active: return {"active": False}
        elapsed = time.time() - self.started
        if elapsed >= self.duration:
            self.active = False; return {"active": False}
        phase_t = elapsed % 6.0
        if phase_t < 2.5: phase, instruction, progress = "INHALE", "Breathe in slowly", phase_t/2.5
        elif phase_t < 3.5: phase, instruction, progress = "HOLD", "Stay comfortable", 1.0
        else: phase, instruction, progress = "EXHALE", "Breathe out slowly", 1-(phase_t-3.5)/2.5
        return {"active": True, "phase": phase, "instruction": instruction, "progress": max(0,min(1,progress))}
