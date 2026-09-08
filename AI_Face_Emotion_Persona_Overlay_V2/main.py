import argparse
import time
import cv2

from face_detector import FaceDetector
from emotion_model import EmotionModel
from response_engine import ResponseEngine
from session_tracker import SessionTracker
from overlay_utils import HUD, save_screenshot
from breathing import Breathing


def args():
    p = argparse.ArgumentParser(description="AI Face Emotion & Persona Overlay — Hackathon V2")
    p.add_argument("--webcam-id", type=int, default=0)
    p.add_argument("--camera-scale", type=float, default=0.70)
    p.add_argument("--smoothing", type=float, default=0.72)
    return p.parse_args()


def _print_milestone(item):
    print(f"\n========== {item['label']} REPORT ==========")
    print(f"Overall expression score: {item['overall']:.1f}/100")
    print(f"Dominant expression:      {item['dominant']}")
    print(f"Average confidence:       {item['avg_confidence'] * 100:.1f}%")
    print(f"Face samples averaged:    {item['samples']:,}")
    print("Expression averages:")
    for label, value in sorted(item["probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:<10} {value * 100:5.1f}%")
    print("===========================================\n")


def main():
    a = args()
    cap = cv2.VideoCapture(a.webcam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(a.webcam_id)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Try --webcam-id 1")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = FaceDetector(max_faces=1)
    model = EmotionModel(smoothing=a.smoothing)
    responder = ResponseEngine(cooldown=5.0)
    tracker = SessionTracker()
    hud = HUD(1280, 720)
    breath = Breathing()
    backend = "HEURISTIC"
    announced = set()

    print("\n===============================================")
    print(" AI FACE EMOTION & PERSONA OVERLAY — V2")
    print("===============================================")
    print("LIVE cumulative scoring:")
    print("  2 min  -> first long-term report")
    print("  5 min  -> deeper average")
    print(" 10 min  -> extended average")
    print("100 min  -> long-session average")
    print("Every score is the average of ALL detected-face probability samples up to that milestone.")
    print("S screenshot | B breathing | R reset | H support | +/- smoothing | ESC exit")
    print("Privacy: processing is local. Expression estimate only.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: camera frame unavailable")
                break

            frame = cv2.flip(frame, 1)
            scale = max(0.35, min(1.0, a.camera_scale))
            if scale != 1.0:
                small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                faces_small = detector.detect(small)
                faces = []
                for f in faces_small:
                    x, y, w, h = f["bbox"]
                    lm = [(int(px / scale), int(py / scale)) for px, py in f["landmarks"]]
                    faces.append({"bbox": (int(x / scale), int(y / scale), int(w / scale), int(h / scale)), "landmarks": lm})
            else:
                faces = detector.detect(frame)

            result = model.update(faces, frame)
            tracker.update(result)
            msg = responder.update(result)
            stats = tracker.stats()

            # Announce each milestone once when it is reached.
            for seconds, item in stats.get("milestones", {}).items():
                if seconds not in announced:
                    _print_milestone(item)
                    announced.add(seconds)

            display = hud.draw(
                frame,
                result,
                stats,
                responder.active_message(),
                backend,
                a.smoothing,
                breath.update(),
            )

            cv2.imshow("AI Face Emotion & Persona Overlay — V2", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord('+'), ord('=')):
                a.smoothing = min(.95, a.smoothing + .05)
                model.set_smoothing(a.smoothing)
            elif key == ord('-'):
                a.smoothing = max(0.0, a.smoothing - .05)
                model.set_smoothing(a.smoothing)
            elif key in (ord('s'), ord('S')) and time.time() - hud.last_screenshot > .7:
                path = save_screenshot(display)
                hud.last_screenshot = time.time()
                print("Screenshot:", path)
            elif key in (ord('h'), ord('H')):
                responder.update(result, force=True)
            elif key in (ord('b'), ord('B')):
                breath.start()
            elif key in (ord('r'), ord('R')):
                model.reset()
                tracker.reset()
                responder.reset()
                announced.clear()
                print("Session reset — all cumulative scores cleared.")
            elif key in (ord('d'), ord('D')):
                backend = "HEURISTIC" if backend != "HEURISTIC" else "ONNX READY"
                print("Backend display:", backend, "(ONNX is not activated without a compatible model file.)")

    finally:
        detector.release()
        cap.release()
        cv2.destroyAllWindows()
        stats = tracker.stats()
        print("\n========== SESSION SUMMARY ==========")
        print(f"Duration: {stats['elapsed']:.1f}s")
        print(f"Face samples: {stats['samples']:,}")
        print(f"Dominant expression: {stats['dominant']}")
        print(f"Average confidence: {stats['avg_confidence'] * 100:.1f}%")
        print(f"Current cumulative score: {stats['overall']:.1f}/100")
        print("Milestones reached:", ", ".join(v["label"] for v in stats["milestones"].values()) or "none")
        print("=====================================\n")


if __name__ == "__main__":
    main()
