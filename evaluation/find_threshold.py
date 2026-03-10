import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model("models/mobilenetv2/mobilenet_best.keras")

test_dir = "datasets/processed/test"

gen = ImageDataGenerator(rescale=1./255)

test_gen = gen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

y_true = test_gen.classes
y_prob = model.predict(test_gen).ravel()

fpr, tpr, thresholds = roc_curve(y_true, y_prob)

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print("Optimal threshold:", optimal_threshold)