import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
import joblib
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xcep_preprocess
from tensorflow.keras.applications import VGG16, Xception
from tensorflow.keras.models import Model
import cv2
from PIL import Image
import uuid

IMG_SIZE = 200
loading_str = f'_size_{IMG_SIZE}'

app = Flask(__name__)

# Load the trained model and class map
model = joblib.load(f"xgb_model{loading_str}.pkl")
class_map = np.load(f"features{loading_str}/class_map.npy", allow_pickle=True).item()
id_to_class = {v: k for k, v in class_map.items()}

# Load the pre-trained models for feature extraction
vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
vgg_model = Model(inputs=vgg_model.input, outputs=vgg_model.output)

xception_model = Xception(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
xception_model = Model(inputs=xception_model.input, outputs=xception_model.output)

def remove_background(image):
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(img_cv, img_cv, mask=mask)
    result[mask == 0] = [255, 255, 255]
    return result

def preprocess_image(image_path):
    try:
        # Open the image and convert to RGB
        image = Image.open(image_path).convert("RGB")
        
        # Remove background
        img = remove_background(image)
        
        # Resize the image to the target size
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize the image
        return resized / 255.0
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE, 3))

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = []
    file_paths_for_display = []

    if request.method == "POST":
        if "images" not in request.files:
            return render_template("index.html", error="No files uploaded.")
        
        files = request.files.getlist("images")
        if not files or all(file.filename == "" for file in files):
            return render_template("index.html", error="No files selected.")

        # Ensure the 'uploads' directory exists
        os.makedirs("uploads", exist_ok=True)

        for file in files:
            # Generate a unique filename
            filename = f"{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
            file_path = os.path.join("uploads", filename)
            file.save(file_path)

            # Update file_path for the template
            file_path_for_display = f"/uploads/{filename}"
            file_paths_for_display.append(file_path_for_display)

            # Preprocess the image
            image = preprocess_image(file_path)
            image = np.expand_dims(image, axis=0)

            # Extract features for the uploaded image
            vgg_features = vgg_model.predict(image)
            xception_features = xception_model.predict(image)

            # Flatten and combine features
            combined_features = np.concatenate([vgg_features.flatten(), xception_features.flatten()]).reshape(1, -1)
            print("Shape of combined features:", combined_features.shape)  # For debugging

            # Predict the class
            try:
                prediction = model.predict(combined_features)[0]
                predicted_class = id_to_class[prediction]
                predictions.append(predicted_class)
            except ValueError as e:
                predictions.append(f"Error: {e}")

        return render_template("index.html", predictions=zip(file_paths_for_display, predictions))

    return render_template("index.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory("uploads", filename)


if __name__ == "__main__":
    app.run(debug=True)