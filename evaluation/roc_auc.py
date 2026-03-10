import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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
y_true = test_gen.classes

# ROC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.show()

print("AUC Score:", roc_auc)