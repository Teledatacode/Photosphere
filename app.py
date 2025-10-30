from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import requests
from flask_cors import CORS

# ⚙️ Desactiva OpenCL (Render free no tiene GPU)
cv2.ocl.setUseOpenCL(False)

app = Flask(__name__)
CORS(app)

# 🌐 URL de tu Worker de Cloudflare
CLOUDFLARE_UPLOAD_URL = "https://traveler-publications-worker.legendary24000.workers.dev/upload"


# 🧩 Convertir base64 a imagen OpenCV
def base64_to_cv2image(b64):
    try:
        header, encoded = b64.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("Error decoding base64:", e)
        return None


# 🧠 Función de stitching robusta
def safe_stitch(images):
    """Intenta stitching en varios modos y genera imagen de respaldo si todo falla."""
    print("🧵 Iniciando stitching robusto...")

    # Preprocesar brillo/contraste (mejora en escenas con paredes o cielo)
    equalized = []
    for img in images:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        equalized.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))

    # Probar distintos modos de stitching
    for mode_name, mode in [("PANORAMA", cv2.Stitcher_PANORAMA), ("SCANS", cv2.Stitcher_SCANS)]:
        print(f"🔁 Intentando stitching en modo {mode_name}...")
        stitcher = cv2.Stitcher_create(mode)
        stitcher.setPanoConfidenceThresh(0.3)  # más tolerante
        status, stitched = stitcher.stitch(equalized)
        if status == cv2.Stitcher_OK:
            print(f"✅ Stitching completado con modo {mode_name}.")
            return stitched

    # Si todo falla, crear imagen simple de respaldo
    print("⚠️ Stitching falló, generando respaldo...")
    try:
        h = max(img.shape[0] for img in images)
        resized = [cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)) for img in images]
        stitched = cv2.hconcat(resized)
        print("🧩 Imagen de respaldo creada (concatenación simple).")
        return stitched
    except Exception as e:
        print("❌ Falló la creación de respaldo:", e)
        raise RuntimeError("No se pudo generar imagen final")


# 📸 Endpoint principal de subida
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

    print("🧵 Procesando stitching...")
    try:
        stitched = safe_stitch(images)
    except Exception as e:
        print("❌ Error en stitching:", e)
        return jsonify({"error": "Stitching failed", "details": str(e)}), 500

    print("✅ Stitching completo, preparando envío a Cloudflare...")

    # Convertir resultado a JPG
    _, buffer = cv2.imencode('.jpg', stitched)
    image_bytes = BytesIO(buffer.tobytes())

    try:
        # Metadatos del usuario
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

        # Enviar a Cloudflare Worker
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
            print("⚠️ Error desde Cloudflare:", response.text)
            return jsonify({"error": "Upload failed", "details": response.text}), 500

    except Exception as e:
        print("⚠️ Excepción al subir a Cloudflare:", e)
        return jsonify({"error": "Exception during upload", "details": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return "🛰️ Stitching backend is running on Render!"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
