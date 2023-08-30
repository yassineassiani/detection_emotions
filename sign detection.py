import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Chemin vers les dossiers contenant les images
folder_a = "y/a"
folder_b = "y/b"
folder_c = "y/c"

# Liste des chemins d'accès aux images et étiquettes correspondantes
image_paths = []
labels = []

for filename in os.listdir(folder_a):
    if filename.endswith(".jpg"):
        image_paths.append(os.path.join(folder_a, filename))
        labels.append("a")

for filename in os.listdir(folder_b):
    if filename.endswith(".jpg"):
        image_paths.append(os.path.join(folder_b, filename))
        labels.append("b")

for filename in os.listdir(folder_c):
    if filename.endswith(".jpg"):
        image_paths.append(os.path.join(folder_c, filename))
        labels.append("c")

# Création du DataFrame avec les chemins d'accès aux images et les étiquettes
data = pd.DataFrame({"image_paths": image_paths, "labels": labels})

# Paramètres du modèle
image_size = (300, 300)
batch_size = 32
num_classes = 3

# Création des générateurs d'images
train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_dataframe(
    data,
    x_col="image_paths",
    y_col="labels",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

test_generator = test_datagen.flow_from_dataframe(
    data,
    x_col="image_paths",
    y_col="labels",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)

# Création du modèle CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(300, 300, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entraînement du modèle
model.fit(train_generator, epochs=10, validation_data=test_generator)


# Évaluation du modèle sur
# Paramètres du détecteur de main
detector = HandDetector(detectionCon=0.8)


# Capture vidéo
cap = cv2.VideoCapture(0)


while True:
    # Lecture de la frame vidéo
    success, frame = cap.read()

    if not success:
        break

    # Détection des mains dans la frame
    im = frame.copy()
    hands, frame = detector.findHands(frame)

    for hand in hands:
        x, y, w, h = hand["bbox"]
        imgC = frame[y - 20 : y + h + 20, x - 20 : x + w + 20]
        imgw = np.ones((300, 300, 3), np.uint8) * 255
        a = h / w
        if a > 1:
            k = 300 / h
            wcal = math.ceil(k * w)
            imgR = cv2.resize(imgC, (wcal, 300))
            wg = math.ceil((300 - wcal) / 2)
            imgw[:, wg : wcal + wg] = imgR

        else:
            k = 300 / w
            hcal = math.ceil(k * h)
            imgR = cv2.resize(imgC, (300, hcal))
            hg = math.ceil((300 - hcal) / 2)
            imgw[hg : hcal + hg, :] = imgR
        img = cv2.cvtColor(imgw, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300, 300))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        label = np.argmax(predictions)
        print(label)
        label_text = chr(
            label + ord("A")
        )  # Convertir l'indice de classe en lettre majuscule
        cv2.putText(
            im, label_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2
        )
        cv2.rectangle(im, (x - 20, y - 20), (x + h + 20, y + w + 20), (255, 0, 255), 4)

    # Affichage de la frame résultante
    cv2.imshow("Hand Detection", im)

    # Sortie de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()
