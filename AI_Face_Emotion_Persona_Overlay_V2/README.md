# AI Face Emotion & Persona Overlay — V2

A real-time webcam HUD that estimates **apparent facial expressions** locally and builds a cumulative expression profile over time.

## Long-term cumulative scoring

The screen now keeps a running average of **all detected-face probability samples**. It creates reports at:

- **2 minutes** — first report
- **5 minutes** — deeper average
- **10 minutes** — extended average
- **100 minutes** — long-session average

Each milestone is cumulative. For example, the 10-minute report averages the facial-detection probability data collected throughout the session up to 10 minutes; it is not based on the last frame.

The HUD shows:
- Overall cumulative expression score /100
- Average percentage for Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral and Uncertain
- Number of face samples used
- Average detection confidence
- Dominant expression
- 2/5/10/100-minute milestone status
- Live expression and persona
- Support message and breathing reset

If no face is detected, that frame is **not added as an emotion sample**, so the long-term score is based only on actual facial detections.

## Controls

- `S` — screenshot
- `B` — breathing reset
- `R` — reset the entire session and clear cumulative scores
- `H` — force a support message
- `+ / =` — increase smoothing
- `-` — decrease smoothing
- `D` — backend display
- `ESC` — exit

## Run

```powershell
python main.py --camera-scale 0.70
```

## Install

Python 3.11 is recommended on Windows.

```powershell
python -m pip install -r requirements.txt
```

## Important interpretation note

This system estimates **visible facial expression patterns**. It does not scientifically determine a person's private/internal emotional state.
