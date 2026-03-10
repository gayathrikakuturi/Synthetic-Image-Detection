import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# PATHS
# =========================
test_dir = "datasets/processed/test"
model_path = "models/mobilenetv2/mobilenet_best.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(model_path)

# =========================
# TEST GENERATOR
# =========================
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# =========================
# PREDICTIONS  ⭐ FIXED
# =========================
y_prob = model.predict(test_gen, verbose=1).ravel()
y_pred = (y_prob > 0.5).astype(int)
y_true = test_gen.classes

# =========================
# CONFUSION MATRIX
# =========================
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

plt.ylabel("True label")
plt.xlabel("Predicted label")

plt.savefig("confusion_matrix.png")
plt.show()

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# =========================
# ROC + AUC ⭐ FIXED
# =========================
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