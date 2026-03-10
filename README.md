# 🖼️ AI Synthetic Image Detection System

An **AI-powered web application** that detects whether an image is **Real or AI-generated (Synthetic)** using deep learning and explainable AI techniques.

The system uses **MobileNetV2 for classification** and **GradCAM for visual explanation**, allowing users to understand which parts of an image influenced the model's prediction.

Users can upload images through **file upload 📁 or webcam 📷**, and the system provides **prediction results, feature analysis, and explainable heatmaps**.

---

# 📌 Problem Statement

With the rapid growth of **Generative AI models (GANs, diffusion models, etc.)**, synthetic images are becoming increasingly realistic.

This creates several challenges:

❌ Spread of misinformation through fake images  
❌ Difficulty verifying authenticity of digital media  
❌ Lack of transparency in deep learning predictions  
❌ Limited tools for explainable AI detection

This project aims to build a system capable of:

✔ Detecting AI-generated images  
✔ Providing visual explanation for predictions  
✔ Analyzing structural artifacts in images  
✔ Delivering results through an interactive web interface

---

# 🎯 Solution Overview

The **AI Synthetic Image Detection System** works as follows:

1️⃣ User uploads an image or captures it via webcam  
2️⃣ Image is preprocessed and resized to **224×224**  
3️⃣ **MobileNetV2 CNN** predicts Real or Fake  
4️⃣ **GradCAM** generates attention heatmaps  
5️⃣ Feature analysis extracts patterns such as:
- Texture consistency
- Lighting patterns
- Edge artifacts
- Background influence  
6️⃣ Results are displayed in a **Flask web application**

---

# ✨ Key Features

🚀 Deep learning based **Real vs Synthetic image detection**

🧠 **MobileNetV2 CNN model**

🔥 **GradCAM Explainable AI visualization**

📊 Feature analysis including:

- Texture consistency
- Lighting consistency
- Edge artifact detection
- Background influence

📷 Image upload + webcam support

📈 Confidence score and probability distribution

🌐 Interactive web interface built with **Flask**

---

# 🧠 Machine Learning Approach

### 📌 Model Used

**MobileNetV2 Convolutional Neural Network**

### Why MobileNetV2?

⚡ Lightweight architecture  
⚡ Fast inference speed  
⚡ Efficient for image classification  
⚡ Suitable for real-time applications

---

### 🔍 Explainable AI

The system integrates **GradCAM (Gradient-weighted Class Activation Mapping)**.

GradCAM highlights:

🔥 Image regions influencing predictions  
🔎 Structural patterns and artifacts  
📊 Model attention areas

This makes the model **interpretable and transparent**.

---

# 📊 Model Evaluation

The model was evaluated using:

📊 Accuracy  
📉 Confusion Matrix  
📈 ROC Curve  
📊 AUC Score  
📋 Classification Report  

Typical performance:

```
Accuracy: ~85%
AUC Score: ~0.88
Balanced Precision and Recall
```

---

# 📂 Datasets Used

The model was trained using **multiple real and synthetic image datasets**.

### Real Image Datasets

#### CIFAR-10
Natural images of animals, vehicles, and objects.

🔗 https://www.kaggle.com/datasets/ayush1220/cifar10

---

#### StyleGAN Real Faces Dataset
Contains **70k real faces and 70k GAN-generated faces**.

Only the **real faces** were used as authentic samples.

🔗 https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

---

#### MU-CIFAR10
Modified CIFAR-10 dataset used for machine learning benchmarking.

🔗 https://www.kaggle.com/competitions/mu-cifar10

---

### Synthetic Image Datasets

#### SFHQ (Synthetic Faces High Quality)

🔗 https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1

---

#### Synthetic Objects Dataset

🔗 https://www.kaggle.com/datasets/zarkonium/synthetic-image-dataset-cats-dogs-bikes-cars

---

#### SuSy Dataset

🔗 https://huggingface.co/datasets/HPAI-BSC/SuSy-Dataset

---
## 📦 Dataset Download

Due to GitHub file size limits, the processed dataset is hosted externally.

Download the datasets from Google Drive:

https://drive.google.com/drive/folders/1_mR-Pe0wZ6ODpzbnR0wJ6DVejqtzTRNq?usp=sharing

After downloading, extract them inside:

datasets/processed/

### Dataset Structure

```
datasets
│
├── raw
│   ├── cifar10
│   ├── sfhq
│   ├── stylegan_real
│   ├── SuSy
│   └── synthetic_objects
│
└── processed
    ├── train
    │   ├── fake
    │   └── real
    │
    ├── val
    │   ├── fake
    │   └── real
    │
    └── test
        ├── fake
        └── real
```

---

# ⚙️ Data Preprocessing

Preprocessing includes:

✔ Image resizing to **224×224**  
✔ Dataset balancing  
✔ Train/validation/test split  
✔ Data normalization  

Scripts available in:

```
preprocessing/
```

---

# 🛠️ Tech Stack

| Category | Technology |
|--------|-----------|
| Language | Python |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV |
| Data Processing | NumPy |
| Explainable AI | GradCAM |
| Version Control | Git & GitHub |

---
## 📦 Pretrained Model Weights

The trained model weights are not included in this repository due to GitHub file size limitations.

Download the pretrained MobileNetV2 model from the link below:

Google Drive Link:  
https://drive.google.com/drive/folders/1_mR-Pe0wZ6ODpzbnR0wJ6DVejqtzTRNq?usp=sharing

After downloading, place the file inside:

models/mobilenetv2/

Example:

models/
└── mobilenetv2/
    └── mobilenet_best.keras

# 🗂️ Project Structure

```
## 🗂️ Project Structure

SYNIMGDET
│
├── datasets
│   │
│   ├── raw
│   │   ├── cifar10
│   │   ├── sfhq
│   │   ├── stylegan_real
│   │   ├── SuSy
│   │   └── synthetic_objects
│   │
│   └── processed
│       │
│       ├── train
│       │   ├── fake
│       │   └── real
│       │
│       ├── val
│       │   ├── fake
│       │   └── real
│       │
│       └── test
│           ├── fake
│           └── real
│
├── preprocessing
│   ├── count_images.py
│   ├── count_processed.py
│   ├── rebalance_processed.py
│   ├── reduce_dataset.py
│   └── split_dataset.py
│
├── models
│   └── mobilenetv2
│       ├── mobilenet_best.keras
│       ├── mobilenet_best.h5
│       ├── mobilenet_model.py
│       └── train_mobilenet.py
│
├── evaluation
│   ├── confusion_matrix.py
│   ├── eval_model.py
│   ├── find_threshold.py
│   ├── metrics.py
│   └── roc_auc.py
│
├── explainability
│   └── gradcam.py
│
├── frontend
│   ├── static
│   │
│   └── templates
│       ├── index.html
│       ├── learnmore.html
│       └── result.html
│
├── deployment
│   └── app.py
│
├── requirements.txt
│
└── README.md
```

---

# ▶️ How to Run the Project Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/gayathrikakuturi/Synthetic-Image-Detection.git
```

### 2️⃣ Navigate to project folder

```bash
cd YOUR_REPO_NAME
```

### 3️⃣ Create virtual environment

```bash
python -m venv venv
```

### 4️⃣ Activate environment

Windows

```bash
venv\Scripts\activate
```

### 5️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 6️⃣ Run the application

```bash
python deployment/app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

# ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.

- The model may not detect all AI-generated images.
- Predictions should not be considered definitive proof of authenticity.
- Always verify digital media from trusted sources.

---

# 🚀 Future Enhancements

🔮 Increase dataset diversity  
🔮 Integrate transformer-based detection models  
🔮 Improve GradCAM visualization  
🔮 Add face artifact detection  
🔮 Deploy system to cloud platforms  
🔮 Add user authentication

---

# 👩‍💻 Author

**Gayathri Kakuturi**  
AI / Machine Learning Enthusiast  

🔗 GitHub: https://github.com/gayathrikakuturi
---

# ⭐ Support

If you found this project useful:

⭐ Star the repository  
🍴 Fork the project  
💡 Share suggestions or improvements