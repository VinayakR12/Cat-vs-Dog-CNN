from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle
import os
import io

app = Flask(__name__)

MODEL_PATH = "CNNModel.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    img_array = preprocess_image(file_path)

    prediction = model.predict(img_array)
    result = "Dog" if prediction[0][0] > 0.5 else "Cat"

    return jsonify({"prediction": result, "image_url": file_path})

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)