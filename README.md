# 🧬 Skin Cancer Classifier

This project is a web-based application that uses Artificial Intelligence (AI) to classify skin lesions as **Benign** or **Malignant**. It uses deep learning models such as **ResNet50** and **MobileNetV2**, combined through an ensemble approach to improve classification accuracy.

> ⚠️ **Disclaimer**: This application is for educational and demonstration purposes only. It is not intended for real-world medical diagnosis. Always consult a certified medical professional.

---

## 🔍 Features

- 🖼️ **Image Classification**: Upload an image of a skin lesion (`.jpg`, `.png`, etc.), and the model predicts if it's benign or malignant.
- 🧠 **Ensemble Learning**: Combines predictions from ResNet50 and MobileNetV2 for better accuracy.
- 📊 **Performance Metrics**: Displays accuracy, precision, recall, F1-score, and ROC AUC.
- 🌐 **Interactive Web UI**: Built with Flask, HTML, CSS (Bootstrap), and JavaScript.
- 🧾 **Dataset**: Trained on the [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) dataset — a large collection of dermatoscopic images for skin cancer classification.

---

## 📊 Model Performance

### ✅ ResNet50 Evaluation

- **Accuracy**: 84.97%  
- **Precision**: 65.52%  
- **Recall**: 48.59%  
- **F1 Score**: 55.80%  
- **ROC AUC**: 88.10%

> You can find full metrics for each model in the `saved_models/` folder.

---

## 📁 Project Structure
── app.py # Main Flask server
├── templates/ # HTML templates
├── static/ # CSS, JS, and image files
├── uploads/ # Uploaded test images
├── saved_models/ # Pretrained models and metrics
│ ├── resnet50_model.h5
│ ├── mobilenetv2_model.h5
│ ├── ensemble_metrics.json
│ └── ...
├── README.md # Project documentation

---

## 🚀 How to Run Locally

```bash
# Step 1: Clone the repository
git clone https://github.com/your-username/skin-cancer-classifier.git
cd skin-cancer-classifier

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the Flask server
python app.py

