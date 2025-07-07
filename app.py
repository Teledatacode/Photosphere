from flask import Flask, request, jsonify 
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
from flask_cors import CORS  # <-- Importa flask_cors

app = Flask(__name__)
CORS(app)  # <-- Habilita CORS para todas las rutas y orígenes

def base64_to_cv2image(b64):
    try:
        header, encoded = b64.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("Error decoding base64:", e)
        return None

def cv2image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return "data:image/jpeg;base64," + b64

@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    if not data or "photos" not in data:
        return jsonify({"error": "No photos provided"}), 400

    images = []
    for item in data["photos"]:
        img = base64_to_cv2image(item["photo"])
        if img is not None:
            images.append(img)

    if len(images) < 2:
        return jsonify({"error": "Need at least 2 valid images for stitching"}), 400

    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        return jsonify({"error": f"Stitching failed (code {status})"}), 500

    result_b64 = cv2image_to_base64(stitched)
    return jsonify({ "image": result_b64 })

@app.route("/", methods=["GET"])
def index():
    return "Stitching backend is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
