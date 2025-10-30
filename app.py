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

# ----------------- UTILIDADES -----------------
def base64_to_cv2image(b64):
    try:
        header, encoded = b64.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("⚠️ Error decoding base64:", e)
        return None

def safe_resize_to_2_1(img, width=4096):
    h = int(width / 2)
    return cv2.resize(img, (width, h))

def safe_stitch(images):
    print(f"🧩 Stitching {len(images)} images en grupos...")
    results = []
    group_size = 5
    for i in range(0, len(images), group_size):
        chunk = images[i:i + group_size]
        print(f"📸 Grupo {i // group_size + 1}: {len(chunk)} fotos")

        stitcher = cv2.Stitcher_create()
        status, pano = stitcher.stitch(chunk)

        if status != cv2.Stitcher_OK or pano is None:
            print(f"⚠️ Stitch parcial falló (code {status}), aplicando fallback horizontal")
            pano = fallback_concat(chunk)
        results.append(pano)

    if len(results) == 1:
        return results[0]

    print("🔁 Stitching entre resultados parciales...")
    stitcher = cv2.Stitcher_create()
    status, final_pano = stitcher.stitch(results)
    if status != cv2.Stitcher_OK or final_pano is None:
        print("⚠️ Stitch final falló, aplicando fallback global")
        final_pano = fallback_concat(results)

    return safe_resize_to_2_1(final_pano)

def fallback_concat(images):
    # Mezcla horizontal simple, asegurando igual altura
    min_height = min(img.shape[0] for img in images)
    resized = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) for img in images]
    pano = cv2.hconcat(resized)

    # Suaviza bordes
    pano = cv2.GaussianBlur(pano, (5, 5), 0)
    return pano

# ----------------- RUTA PRINCIPAL -----------------
@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    if not data or "photos" not in data:
        return jsonify({"error": "No photos provided"}), 400

    images = [base64_to_cv2image(item["photo"]) for item in data["photos"] if base64_to_cv2image(item["photo"]) is not None]
    if len(images) < 2:
        return jsonify({"error": "Need at least 2 valid images for stitching"}), 400

    print("🧵 Procesando stitching robusto...")
    pano = safe_stitch(images)
    print("✅ Panorámica generada con éxito")

    # Convertir resultado a JPEG
    _, buffer = cv2.imencode('.jpg', pano)
    image_bytes = BytesIO(buffer.tobytes())

    # Datos opcionales
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

    print("☁️ Enviando panorámica a Cloudflare...")
    try:
        response = requests.post(CLOUDFLARE_UPLOAD_URL, files=files, data=payload, timeout=60)
        if response.status_code == 200:
            print("✅ Imagen subida correctamente a Cloudflare R2")
            return jsonify({"ok": True, "cloudflare": response.json()})
        else:
            print("⚠️ Error en subida:", response.text)
            return jsonify({"error": "Upload failed", "details": response.text}), 500
    except Exception as e:
        print("❌ Excepción durante subida:", e)
        return jsonify({"error": "Exception during upload", "details": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "Robust stitching backend is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
