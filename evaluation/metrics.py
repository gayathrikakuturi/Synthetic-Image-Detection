import numpy as np
from sklearn.metrics import classification_report
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

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))