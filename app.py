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

# --- Helpers ---
def base64_to_cv2image(b64):
    try:
        header, encoded = b64.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("Error decoding base64:", e)
        return None

def is_image_valid(img, threshold=10):
    """Descarta imágenes casi blancas o sin textura"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = gray.var()
    return variance > threshold

def preprocess_image(img):
    """Mejora contraste y textura para mejor stitching"""
    try:
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    except Exception as e:
        print("⚠️ Error en detailEnhance:", e)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# --- Routes ---
@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    if not data or "photos" not in data:
        return jsonify({"error": "No photos provided"}), 400

    images = []
    for item in data["photos"]:
        img = base64_to_cv2image(item["photo"])
        if img is not None and is_image_valid(img):
            img = preprocess_image(img)
            images.append(img)

    if len(images) < 2:
        return jsonify({"error": "Not enough valid images for stitching"}), 400

    print(f"🧵 Procesando stitching con {len(images)} imágenes...")

    # --- Stitching ---
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, stitched = stitcher.stitch(images)

    # Fallback SCANS si PANORAMA falla
    if status != cv2.Stitcher_OK:
        print("⚠️ PANORAMA failed, intentando SCANS...")
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        status, stitched = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        print("❌ Falló el stitching:", status)
        return jsonify({"error": f"Stitching failed (code {status})"}), 500

    print("✅ Stitching completo, preparando envío a Cloudflare...")

    _, buffer = cv2.imencode('.jpg', stitched)
    image_bytes = BytesIO(buffer.tobytes())

    # --- Enviar a Cloudflare ---
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

        response = requests.post(
            CLOUDFLARE_UPLOAD_URL,
            files=files,
            data=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Imagen enviada a Cloudflare R2:", result)
            return jsonify({"ok": True, "cloudflare": result})
        else:
            print("⚠️ Error desde Cloudflare:", response.text)
            return jsonify({"error": "Upload failed", "details": response.text}), 500

    except Exception as e:
        print("⚠️ Excepción al subir a Cloudflare:", e)
        return jsonify({"error": "Exception during upload", "details": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return "Stitching backend is running."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
