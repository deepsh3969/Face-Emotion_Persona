import cv2
import mediapipe as mp


class FaceDetector:
    """Stable MediaPipe Face Mesh wrapper for MediaPipe 0.10.x."""

    def __init__(self, max_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.55, min_tracking_confidence=0.55):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        faces = []
        if not results.multi_face_landmarks:
            return faces

        h, w = frame.shape[:2]
        for face in results.multi_face_landmarks:
            pts = [(int(max(0, min(w - 1, p.x * w))),
                    int(max(0, min(h - 1, p.y * h))))
                   for p in face.landmark]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
            y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
            faces.append({"bbox": (x1, y1, x2 - x1, y2 - y1), "landmarks": pts})

        # Primary face first, largest face wins.
        faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
        return faces

    def release(self):
        self.face_mesh.close()
