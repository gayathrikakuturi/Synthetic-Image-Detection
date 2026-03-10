import cv2
import numpy as np
import tensorflow as tf

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/mobilenetv2/mobilenet_best.keras"
IMG_PATH = "test.png"   # change if needed
IMG_SIZE = 224
LAST_CONV_LAYER = "Conv_1"   # if error try: block_16_project

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# GRADCAM FUNCTION
# =========================
def make_gradcam(img, model, layer_name):
    processed = preprocess(img)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(processed)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(pooled_grads * conv_output, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.reduce_max(heatmap)

    heatmap = heatmap.numpy()   # convert ONCE here

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return overlay

# =========================
# RUN
# =========================
if __name__ == "__main__":
    img = cv2.imread(IMG_PATH)

    overlay = make_gradcam(img, model, LAST_CONV_LAYER)

    cv2.imwrite("gradcam_output.jpg", overlay)
    print("Grad-CAM saved as gradcam_output.jpg")