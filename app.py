import os
import sys
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from werkzeug.utils import secure_filename

# Initialize Flask app with custom templates folder
app = Flask(__name__, template_folder=r'C:\Users\HP\Desktop\scan\templates')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join('saved_models', 'resnet50_skin_cancer.keras')
METRICS_PATH = os.path.join('saved_models', 'resnet50_metrics.json')
CLASS_INDICES_PATH = os.path.join('saved_models', 'class_indices.json')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and metrics
print("🔄 Loading model and metrics...")
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

def load_json_safely(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
        return {}

metrics = load_json_safely(METRICS_PATH)
class_indices = load_json_safely(CLASS_INDICES_PATH)
class_labels = {v: k for k, v in class_indices.items()}
print("✅ Model and metrics loaded!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html',
                           accuracy=f"{metrics.get('accuracy', 0)*100:.2f}%",
                           precision=f"{metrics.get('precision', 0)*100:.2f}%",
                           recall=f"{metrics.get('recall', 0)*100:.2f}%",
                           f1_score=f"{metrics.get('f1_score', 0)*100:.2f}%")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)
        predicted_class = class_labels[int(prediction[0][0] > 0.5)]
        confidence = float(prediction[0][0] if predicted_class == 'malignant' else 1 - prediction[0][0])

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence*100:.2f}%",
            'image_url': f"/uploads/{filename}"
        })
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        # Keep the file temporarily for display (optional: remove after serving)
        pass

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/metrics')
def get_metrics():
    return jsonify(metrics)

@app.route('/download-model')
def download_model():
    keras_model_path = os.path.join('saved_models', 'resnet50_skin_cancer.keras')
    if os.path.exists(keras_model_path):
        return send_file(keras_model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'})

if __name__ == '__main__':
    # Check if running in Jupyter Notebook
    is_jupyter = 'ipykernel' in sys.modules
    try:
        # Disable reloader in Jupyter to avoid SystemExit
        if is_jupyter:
            print("📓 Detected Jupyter Notebook environment. Disabling Flask reloader.")
            os.environ['FLASK_ENV'] = 'development'  # Ensure development mode
            app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
        else:
            print("🖥️ Running in standard Python environment.")
            app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"⚠️ Error starting Flask server: {e}")
        sys.exit(1)