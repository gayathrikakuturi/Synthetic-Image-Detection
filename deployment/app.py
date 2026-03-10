import cv2
import numpy as np
import tensorflow as tf
import base64
from flask import Flask, render_template, request

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

IMG_SIZE = 224
MODEL_PATH = "models/mobilenetv2/mobilenet_best.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# =========================
# IMAGE PREPROCESS
# =========================

def preprocess(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img,axis=0)

    return img


# =========================
# GRADCAM
# =========================

def generate_gradcam(img):

    processed = preprocess(img)

    last_conv_layer = None

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(processed)

        loss = predictions[:,0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads,axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap,0)

    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))

    heatmap_color = np.uint8(255*heatmap)

    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img,0.6,heatmap_color,0.4,0)

    return overlay,heatmap


# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(img, heatmap):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -----------------
    # Texture Consistency
    # -----------------
    texture = np.var(cv2.Laplacian(gray, cv2.CV_64F))
    texture_score = min(100, (texture / 500) * 100)

    # -----------------
    # Edge Artifacts
    # -----------------
    edges = cv2.Canny(gray,100,200)

    edge_pixels = np.sum(edges > 0)
    edge_ratio = edge_pixels / edges.size

    edge_score = min(100, edge_ratio * 100)

    # -----------------
    # Lighting Consistency
    # -----------------
    lighting = np.std(gray)
    lighting_score = min(100, (lighting / 128) * 100)

    # -----------------
    # Background Influence
    # -----------------
    h,w = heatmap.shape

    top = heatmap[:int(h*0.2),:]
    bottom = heatmap[int(h*0.8):,:]

    heatmap_sum = np.sum(heatmap)

    if heatmap_sum == 0:
        background_score = 0
    else:
        background_score = ((np.sum(top) + np.sum(bottom)) / heatmap_sum) * 100

    # clamp values
    texture_score = max(0,min(texture_score,100))
    lighting_score = max(0,min(lighting_score,100))
    edge_score = max(0,min(edge_score,100))
    background_score = max(0,min(background_score,100))

    return (
        round(texture_score,2),
        round(lighting_score,2),
        round(edge_score,2),
        round(background_score,2)
    )


# =========================
# EXPLANATION
# =========================

def generate_explanation(texture,background,edge,lighting):

    explanation=""

    if texture>60:
        explanation+="Irregular texture patterns detected. "

    if edge>60:
        explanation+="Edge artifacts detected. "

    if lighting<40:
        explanation+="Lighting inconsistency observed. "

    if background>60:
        explanation+="Model attention focused on background regions. "

    if explanation=="":
        explanation="Image exhibits natural structural patterns typical of real photographs."

    return explanation


# =========================
# MAIN PROCESS
# =========================

def process(img):

    processed = preprocess(img)

    prob = model.predict(processed,verbose=0)[0][0]

    if prob > 0.5:
        label = "Real"
        confidence = prob
    else:
        label = "Fake"
        confidence = 1 - prob

    overlay,heatmap = generate_gradcam(img)

    texture,lighting,edges,background = extract_features(img,heatmap)

    explanation = generate_explanation(texture,background,edges,lighting)

    # encode original image
    _, buffer1 = cv2.imencode(".jpg", img)
    original_base64 = base64.b64encode(buffer1).decode()

    # encode heatmap
    _, buffer2 = cv2.imencode(".jpg", overlay)
    heatmap_base64 = base64.b64encode(buffer2).decode()

    return {

        "label":label,
        "confidence":round(confidence*100,2),

        "original":original_base64,
        "heatmap":heatmap_base64,

        "texture":texture,
        "lighting":lighting,
        "edges":edges,
        "background":background,

        "real_prob":round(prob*100,2),
        "fake_prob":round((1-prob)*100,2),

        "explanation":explanation
    }


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/learnmore")
def learnmore():
    return render_template("learnmore.html")


@app.route("/analyze",methods=["POST"])
def analyze():

    file = request.files.get("image")

    if file:

        file_bytes = np.frombuffer(file.read(),np.uint8)
        img = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)

    else:

        webcam_data = request.form["webcam"]
        img_bytes = base64.b64decode(webcam_data.split(",")[1])
        np_arr = np.frombuffer(img_bytes,np.uint8)
        img = cv2.imdecode(np_arr,cv2.IMREAD_COLOR)

    result = process(img)

    return render_template("result.html",data=result)


# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True)