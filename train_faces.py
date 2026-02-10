import face_recognition
import os
import pickle

dataset_path = "dataset"
known_encodings = []
known_names = []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    for img in os.listdir(person_path):
        image = face_recognition.load_image_file(os.path.join(person_path, img))
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person)

data = {"encodings": known_encodings, "names": known_names}

with open("encodings/faces.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Face training completed")
