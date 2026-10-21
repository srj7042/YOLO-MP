import os
import cv2
import pickle
import numpy as np
from datetime import datetime

# Path constants
ENCODINGS_DIR = "encodings"
ENCODINGS_FILE = os.path.join(ENCODINGS_DIR, "faces.pkl")
os.makedirs(ENCODINGS_DIR, exist_ok=True)

# Lazy-loaded MTCNN detector and FaceNet embedder
_mtcnn_model = None
_facenet_model = None
_torch = None


def get_models():
    global _mtcnn_model, _facenet_model, _torch
    if _facenet_model is None or _mtcnn_model is None:
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
            _torch = torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _mtcnn_model = MTCNN(keep_all=True, device=device)
            _facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
            print(f"[INFO] FaceNet & MTCNN loaded on device: {device}")
        except Exception as e:
            print(f"[WARN] Could not load FaceNet/MTCNN: {e}")
    return _mtcnn_model, _facenet_model


def load_known_faces():
    """Load and format known faces from pickle file safely."""
    if not os.path.exists(ENCODINGS_FILE):
        return {}, [], np.array([])

    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            return {}, [], np.array([])

        names = []
        vectors = []
        for name, embeds in data.items():
            for emb in embeds:
                emb = np.array(emb, dtype=np.float32).flatten()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                names.append(name)
                vectors.append(emb)

        vectors = np.array(vectors) if vectors else np.array([])
        return data, names, vectors
    except Exception as e:
        print(f"[ERROR] Failed to load faces: {e}")
        return {}, [], np.array([])


def save_known_faces(data):
    """Save embeddings dict to pickle file."""
    os.makedirs(ENCODINGS_DIR, exist_ok=True)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)


def detect_faces(frame):
    """
    Detect faces using MTCNN deep neural network.
    Returns list of (x, y, w, h) bounding boxes.
    """
    mtcnn, _ = get_models()
    if mtcnn is None or frame is None or frame.size == 0:
        return []

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_frame)

        if boxes is None or len(boxes) == 0:
            return []

        clean_boxes = []
        H, W = frame.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(W, int(x2))
            y2 = min(H, int(y2))
            w = x2 - x1
            h = y2 - y1
            if w > 20 and h > 20:
                clean_boxes.append((x1, y1, w, h))
        return clean_boxes
    except Exception as e:
        print(f"[ERROR] Detection error: {e}")
        return []


def extract_face_embedding(face_img):
    """Extract normalized 512-d FaceNet embedding from cropped face."""
    _, model = get_models()
    if model is None or face_img is None or face_img.size == 0:
        return None

    try:
        face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160, 160))
        face = (face.astype(np.float32) - 127.5) / 128.0

        device = next(model.parameters()).device
        tensor = _torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with _torch.no_grad():
            emb = model(tensor).cpu().numpy().flatten()

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception as e:
        print(f"[ERROR] FaceNet embedding error: {e}")
        return None


def register_new_face(name, image_path_or_array):
    """Detects face from image and adds embedding for name."""
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
    else:
        image = image_path_or_array

    if image is None:
        return False, "Could not read image"

    faces = detect_faces(image)
    if len(faces) == 0:
        return False, "No face detected in image"

    # Select the largest face box
    largest = max(faces, key=lambda b: b[2] * b[3])
    x, y, w, h = largest
    face_crop = image[y:y+h, x:x+w]

    emb = extract_face_embedding(face_crop)
    if emb is None:
        return False, "Could not generate face embedding"

    data, _, _ = load_known_faces()
    if name not in data:
        data[name] = []
    data[name].append(emb)
    save_known_faces(data)
    return True, f"Face registered for {name}"


def match_face(face_crop, threshold=0.85):
    """
    Compare cropped face with known vectors.
    Returns (matched_name, confidence_str, min_distance).
    """
    emb = extract_face_embedding(face_crop)
    if emb is None:
        return "Unknown", 0.0, 999.0

    _, known_names, known_vectors = load_known_faces()
    if len(known_vectors) == 0:
        return "Unknown", 0.0, 999.0

    distances = np.linalg.norm(known_vectors - emb, axis=1)
    min_idx = np.argmin(distances)
    min_dist = float(distances[min_idx])

    if min_dist < threshold:
        matched_name = known_names[min_idx]
        confidence = max(0.0, round((1.0 - (min_dist / threshold)) * 100, 1))
        return matched_name, confidence, min_dist

    return "Unknown", 0.0, min_dist


def process_image_and_mark(image_path_or_array, mark_in_db=True):
    """
    Detect all faces in an image, match each against database,
    optionally mark attendance, and return annotated image + detections list.
    """
    import database

    if isinstance(image_path_or_array, str):
        frame = cv2.imread(image_path_or_array)
    else:
        frame = image_path_or_array.copy()

    if frame is None:
        return None, []

    faces = detect_faces(frame)
    detected_results = []

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        name, confidence, dist = match_face(face_crop)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        student_info = None
        status_msg = "Unknown"

        if name != "Unknown":
            student_info = database.get_student_by_name(name)
            if student_info and mark_in_db:
                success, msg = database.mark_attendance(student_info["id"])
                status_msg = "Marked" if success else "Already Marked"
            elif student_info:
                status_msg = "Identified"

        detected_results.append({
            "name": name,
            "confidence": confidence,
            "distance": round(dist, 3),
            "status": status_msg,
            "student_info": student_info,
            "box": (int(x), int(y), int(w), int(h))
        })

        # Annotate image
        label = f"{name} ({confidence}%)" if name != "Unknown" else "Unknown"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, detected_results
