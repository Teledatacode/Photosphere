from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import requests
from flask_cors import CORS

cv2.ocl.setUseOpenCL(False)

app = Flask(__name__)
CORS(app)

CLOUDFLARE_UPLOAD_URL = "https://traveler-publications-worker.legendary24000.workers.dev/upload"

# --- Utilidades ---
def base64_to_cv2image(b64):
    try:
        header, encoded = b64.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("Error decoding base64:", e)
        return None

def enhance_image(img):
    """Mejora contraste, color y nitidez."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)

def safe_stitch(images):
    """Realiza stitching o fallback sin fallar nunca."""
    print("🧵 Iniciando stitching robusto...")

    # Mejorar todas las imágenes
    processed = [enhance_image(img) for img in images]

    # Probar stitching en modos tolerantes
    for mode_name, mode in [("PANORAMA", cv2.Stitcher_PANORAMA), ("SCANS", cv2.Stitcher_SCANS)]:
        print(f"🔁 Intentando modo {mode_name}...")
        stitcher = cv2.Stitcher_create(mode)
        stitcher.setPanoConfidenceThresh(0.2)
        status, result = stitcher.stitch(processed)
        if status == cv2.Stitcher_OK and result is not None:
            print(f"✅ Stitching exitoso con modo {mode_name}")
            return result

    # Fallback → unir horizontalmente las imágenes
    print("⚠️ Stitching falló, aplicando fallback...")
    try:
        h = min(img.shape[0] for img in processed)
        resized = [cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)) for img in processed]
        fallback = cv2.hconcat(resized)
        print("🧩 Imagen unida horizontalmente como respaldo.")
        return fallback
    except Exception as e:
        print("❌ Falló el respaldo:", e)
        # Si incluso esto falla, devolvemos la primera imagen
        return processed[0]

# --- Rutas ---
@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    if not data or "photos" not in data:
        return jsonify({"error": "No photos provided"}), 400

    images = [base64_to_cv2image(p["photo"]) for p in data["photos"] if base64_to_cv2image(p["photo"]) is not None]
    if len(images) < 1:
        return jsonify({"error": "No valid images"}), 400

    print("🧩 Procesando stitching...")
    stitched = safe_stitch(images)

    print("✅ Stitching completo, enviando a Cloudflare...")

    _, buffer = cv2.imencode('.jpg', stitched)
    image_bytes = BytesIO(buffer.tobytes())

    # Metadata
    payload = {
        "token": data.get("token", ""),
        "description": data.get("description", ""),
        "userName": data.get("userName", "Traveler"),
        "userPhoto": data.get("userPhoto", "default-avatar.png"),
        "userSocial": data.get("userSocial", ""),
    }

    files = {"image": ("photo360.jpg", image_bytes, "image/jpeg")}

    try:
        response = requests.post(
            CLOUDFLARE_UPLOAD_URL,
            files=files,
            data=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Imagen enviada a Cloudflare R2:", result)
            return jsonify({"ok": True, "cloudflare": result})
        else:
            print("⚠️ Error Cloudflare:", response.text)
            return jsonify({"error": "Upload failed", "details": response.text}), 500

    except Exception as e:
        print("⚠️ Excepción al subir a Cloudflare:", e)
        return jsonify({"error": "Exception during upload", "details": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return "🛰️ Stitching backend is running (robusto y sin fallas)."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
