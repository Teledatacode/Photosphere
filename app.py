from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permite CORS para todas las rutas

# ---------------------------
# Funciones auxiliares
# ---------------------------
def base64_to_cv2image(b64):
    try:
        header, encoded = b64.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("❌ Error decoding base64:", e)
        return None

def cv2image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return "data:image/jpeg;base64," + b64

# Stitching incremental por bloques
def stitch_images_incremental(images, block_size=6):
    # Paso 1: dividir en bloques
    partials = []
    for i in range(0, len(images), block_size):
        block = images[i:i+block_size]
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        stitcher.setPanoConfidenceThresh(0.5)
        status, stitched = stitcher.stitch(block)
        if status != cv2.Stitcher_OK:
            print(f"⚠️ Stitching de bloque {i//block_size} falló con código {status}")
            continue
        partials.append(stitched)

    if len(partials) == 0:
        return None  # Ningún bloque funcionó

    # Paso 2: unir los panoramas parciales
    final = partials[0]
    for part in partials[1:]:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        stitcher.setPanoConfidenceThresh(0.5)
        status, stitched = stitcher.stitch([final, part])
        if status != cv2.Stitcher_OK:
            print(f"⚠️ Stitching parcial falló con código {status}, usando parcial previo")
            continue
        final = stitched

    return final

# ---------------------------
# Ruta principal de stitching
# ---------------------------
@app.route("/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json()
        if not data or "photos" not in data:
            return jsonify({"error": "No photos provided"}), 400

        images = []
        for i, item in enumerate(data["photos"]):
            img = base64_to_cv2image(item["photo"])
            if img is not None:
                images.append(img)
            else:
                print(f"⚠️ Imagen {i} inválida")

        if len(images) < 2:
            return jsonify({"error": "Need at least 2 valid images"}), 400

        print(f"🧵 Procesando {len(images)} imágenes con stitching incremental...")
        stitched_final = stitch_images_incremental(images, block_size=6)

        if stitched_final is None:
            return jsonify({"error": "Stitching incremental falló"}), 500

        print("✅ Stitching completado correctamente")
        result_b64 = cv2image_to_base64(stitched_final)
        return jsonify({"image": result_b64})

    except Exception as e:
        print("🔥 Error general:", e)
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Ruta de prueba
# ---------------------------
@app.route("/", methods=["GET"])
def index():
    return "Stitching backend incremental running."

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
