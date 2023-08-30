import os
import cv2
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (
    Dense,
    Input,
    Dropout,
    GlobalAveragePooling2D,
    Flatten,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
)
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop

# Chemin vers les dossiers contenant les images d'émotions
emotion_folders = ["happy", "neutral", "sad"]
# Taille d'entrée du modèle
input_size = (48, 48)

# Prétraitement des images
images = []
labels = []
for label, folder in enumerate(emotion_folders):
    folder_path = os.path.join("y/train", folder)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris
        image = cv2.resize(image, input_size)  # Redimensionner l'image
        image = image.astype("float32") / 255.0  # Normaliser les valeurs des pixels
        images.append(image)
        labels.append(label)

# Conversion des listes en tableaux numpy
images = np.array(images)
labels = np.array(labels)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Convertir les étiquettes en one-hot encoded
y_train_encoded = to_categorical(y_train, num_classes=len(emotion_folders))
y_test_encoded = to_categorical(y_test, num_classes=len(emotion_folders))

# Créer un modèle
model = keras.Sequential()
model.add(
    keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(input_size[0], input_size[1], 1)
    )
)
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Dense(len(emotion_folders), activation="softmax"))

# Compilation du modèle
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Entraînement du modèle
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Évaluation du modèle
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
emotion_classes = ["happy", "neutral", "sad"]

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

# Chargement du classificateur de visage pré-entraîné
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    # Lire une frame de la capture vidéo
    ret, frame = cap.read()

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Pour chaque visage détecté
    for x, y, w, h in faces:
        # Extraire le visage de la frame
        face = gray[y : y + h, x : x + w]

        # Redimensionner le visage à la taille requise pour le modèle
        face = cv2.resize(face, (48, 48))

        # Normaliser les valeurs des pixels
        face = face.astype("float32") / 255.0

        # Effectuer la prédiction sur le visage
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        emotion_prediction = model.predict(face)[0]
        emotion_label = emotion_classes[np.argmax(emotion_prediction)]

        # Dessiner un rectangle vert autour du visage détecté
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Afficher l'étiquette d'émotion prédite à côté du visage
        cv2.putText(
            frame,
            emotion_label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    # Afficher la frame résultante
    cv2.imshow("Emotion Detection", frame)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
