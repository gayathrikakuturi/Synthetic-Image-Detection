import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# paths
test_dir = "datasets/processed/test"
model_path = "models/mobilenetv2/mobilenet_best.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# load model
model = tf.keras.models.load_model(model_path)

# generator
datagen = ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# prediction
y_prob = model.predict(test_gen, verbose=1).ravel()
y_pred = (y_prob > 0.5).astype(int)
y_true = test_gen.classes

# confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Real","Fake"])
plt.yticks(tick_marks, ["Real","Fake"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.ylabel("True")
plt.xlabel("Predicted")

plt.savefig("confusion_matrix.png")
plt.show()