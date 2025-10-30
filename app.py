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

# ======= Utilidades =======
def base64_to_cv2image(b64):
    try:
        if ',' in b64:
            header, encoded = b64.split(',', 1)
        else:
            encoded = b64
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("❌ Error decoding base64:", e)
        return None

def preprocess_image(img, scale=0.5):
    # Reducir tamaño
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    # Mejorar contraste
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    img[:, :, 0] = gray
    img[:, :, 1] = gray
    img[:, :, 2] = gray
    return img

def stitch_incremental(images):
    if len(images) < 2:
        return None, "Need at least 2 images"

    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    stitcher.setPanoConfidenceThresh(0.6)

    pano = images[0]
    for i in range(1, len(images)):
        status, pano_new = stitcher.stitch([pano, images[i]])
        if status != cv2.Stitcher_OK:
            print(f"⚠️ Stitching failed at image {i} (code {status})")
            # Omitimos la imagen y continuamos
            continue
        pano = pano_new
    return pano, None

# ======= Rutas =======
@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    if not data or "photos" not in data:
        return jsonify({"error": "No photos provided"}), 400

    images = []
    for item in data["photos"]:
        img = base64_to_cv2image(item["photo"])
        if img is not None:
            img = preprocess_image(img, scale=0.5)
            images.append(img)

    if len(images) < 2:
        return jsonify({"error": "Need at least 2 valid images"}), 400

    print("🧵 Procesando stitching incremental...")
    stitched, err = stitch_incremental(images)
    if stitched is None:
        return jsonify({"error": "Stitching failed", "details": err}), 500

    print("✅ Stitching completo, preparando envío a Cloudflare...")
    _, buffer = cv2.imencode('.jpg', stitched)
    image_bytes = BytesIO(buffer.tobytes())
    image_bytes.seek(0)

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
    return "Stitching backend is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
