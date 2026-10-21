import os
import cv2
import database
import face_engine

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(face_engine.ENCODINGS_DIR, exist_ok=True)

print("[INFO] Checking dataset directory...")
people = [p for p in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, p))]

if not people:
    print(f"⚠️ No person folders found in '{DATASET_DIR}/'.")
    print(f"👉 Example structure: {DATASET_DIR}/Suraj_Jaiswal/photo1.jpg")
    print("👉 You can also register students and their photos directly via the Web UI at /students!")
    exit(0)

database.init_db()
embeddings_data, _, _ = face_engine.load_known_faces()
total_images = 0

for person in people:
    person_path = os.path.join(DATASET_DIR, person)
    clean_name = person.replace("_", " ").strip()
    print(f"\n[INFO] Processing faces for: {clean_name}")

    if clean_name not in embeddings_data:
        embeddings_data[clean_name] = []

    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = face_engine.detect_faces(img)
        if len(faces) == 0:
            print(f"  ⚠️ No face found in {img_name}")
            continue

        largest_face = max(faces, key=lambda b: b[2] * b[3])
        x, y, w, h = largest_face
        face_crop = img[y:y+h, x:x+w]

        emb = face_engine.extract_face_embedding(face_crop)
        if emb is not None:
            embeddings_data[clean_name].append(emb)
            total_images += 1
            print(f"  ✅ Encoded {img_name}")

face_engine.save_known_faces(embeddings_data)
print(f"\n🎉 Training complete! {total_images} face embeddings saved to {face_engine.ENCODINGS_FILE}")
