from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CLOUDFLARE_UPLOAD_URL = "https://traveler-publications-worker.legendary24000.workers.dev/upload"

def base64_to_cv2image(b64):
    try:
        header, encoded = b64.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("Error decoding base64:", e)
        return None

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def is_uniform(image, threshold=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray) < threshold

@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    if not data or "photos" not in data:
        return jsonify({"error": "No photos provided"}), 400

    images = []
    for i, item in enumerate(data["photos"]):
        img = base64_to_cv2image(item["photo"])
        if img is None:
            print(f"⚠️ Imagen {i} inválida, ignorada")
            continue
        if is_blurry(img):
            print(f"⚠️ Imagen {i} borrosa, ignorada")
            continue
        if is_uniform(img):
            print(f"⚠️ Imagen {i} muy uniforme, ignorada")
            continue
        images.append(img)

    if len(images) < 2:
        return jsonify({"error": "Need at least 2 valid images for stitching"}), 400

    # Ajustes para reducir fallos por solapamiento insuficiente
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    stitcher.setPanoConfidenceThresh(0.3)  # Default 1.0 → más tolerante

    max_retries = 2
    for attempt in range(max_retries):
        status, stitched = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            break
        else:
            print(f"⚠️ Stitching failed (code {status}), intento {attempt+1}/{max_retries}")
            # Reintentar quitando la última imagen
            if len(images) > 2:
                images.pop(-1)
            else:
                return jsonify({"error": f"Stitching failed after {max_retries} attempts", "code": status}), 500

    print("✅ Stitching completo, preparando envío a Cloudflare...")

    _, buffer = cv2.imencode('.jpg', stitched)
    image_bytes = BytesIO(buffer.tobytes())

    try:
        token = data.get("token", "")
        description = data.get("description", "")
        user_name = data.get("userName", "Traveler")
        user_photo = data.get("userPhoto", "default-avatar.png")
        user_social = data.get("userSocial", "")

        files = {"image": ("photo360.jpg", image_bytes, "image/jpeg")}
        payload = {
            "token": token,
            "description": description,
            "userName": user_name,
            "userPhoto": user_photo,
            "userSocial": user_social,
        }

        # Retry simple para subir a Cloudflare
        response = None
        for _ in range(3):
            try:
                response = requests.post(
                    CLOUDFLARE_UPLOAD_URL,
                    files=files,
                    data=payload,
                    timeout=60
                )
                if response.status_code == 200:
                    break
            except Exception as e:
                print("Retry upload due to exception:", e)

        if response is None or response.status_code != 200:
            return jsonify({"error": "Upload failed", "details": response.text if response else "No response"}), 500

        return jsonify({"ok": True, "cloudflare": response.json()})

    except Exception as e:
        return jsonify({"error": "Exception during upload", "details": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "Stitching backend is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
