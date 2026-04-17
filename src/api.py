import io
import configparser
import numpy as np
import joblib
from flask import Flask, request, jsonify
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from logger import get_logger

app = Flask(__name__)
logger = get_logger("api")

config = configparser.ConfigParser()
config.read("config.ini")

IMAGE_SIZE  = int(config["DATA"]["image_size"])
HOG_ORIENT  = int(config["FEATURES"]["hog_orientations"])
HOG_PPC     = int(config["FEATURES"]["hog_pixels_per_cell"])
HOG_CPB     = int(config["FEATURES"]["hog_cells_per_block"])
MODEL_PATH  = config["PATHS"]["model_path"]
SCALER_PATH = config["PATHS"]["scaler_path"]
HOST        = config["API"]["host"]
PORT        = int(config["API"]["port"])

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def extract_hog_features(image_path):
    img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_gray = rgb2gray(np.array(img))
    return hog(
        img_gray,
        orientations=HOG_ORIENT,
        pixels_per_cell=(HOG_PPC, HOG_PPC),
        cells_per_block=(HOG_CPB, HOG_CPB),
        block_norm="L2-Hys"
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_gray = rgb2gray(np.array(img))
    features = hog(
        img_gray,
        orientations=HOG_ORIENT,
        pixels_per_cell=(HOG_PPC, HOG_PPC),
        cells_per_block=(HOG_CPB, HOG_CPB),
        block_norm="L2-Hys"
    )

    features_scaled = scaler.transform([features])
    pred  = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    label = "dog" if pred == 1 else "cat"
    logger.info(f"Предсказание: {label}, confidence: {max(proba):.3f}")

    return jsonify({
        "label": label,
        "confidence": round(float(max(proba)), 3)
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
