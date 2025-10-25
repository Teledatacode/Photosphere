from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)

# Habilita CORS completamente (esto evita "failed to fetch" por origen)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


# ------------------------------
# 🔹 Funciones auxiliares
# ------------------------------
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


# ------------------------------
# 🔹 Ruta principal de stitching
# ------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    print("📥 Petición recibida en /upload")

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
            return jsonify({"error": "Need at least 2 valid images for stitching"}), 400

        # 🔧 Crear stitcher (modo panorámico, mejor compatibilidad)
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        stitcher.setPanoConfidenceThresh(0.5)

        print("🧵 Iniciando stitching con", len(images), "imágenes...")
        status, stitched = stitcher.stitch(images)

        if status != cv2.Stitcher_OK:
            print(f"❌ Error en stitching: código {status}")
            return jsonify({"error": f"Stitching failed (code {status})"}), 500

        print("✅ Stitching completado correctamente")

        # 🔄 Corregir posible hueco entre bordes horizontales (wrap-around)
        height, width = stitched.shape[:2]
        border = int(width * 0.002)
        stitched[:, :border] = stitched[:, -border:]

        result_b64 = cv2image_to_base64(stitched)
        return jsonify({"image": result_b64})

    except Exception as e:
        print("🔥 Error general:", e)
        return jsonify({"error": str(e)}), 500


# ------------------------------
# 🔹 Ruta base (ping)
# ------------------------------
@app.route("/", methods=["GET"])
def index():
    return "Stitching backend is running."


# ------------------------------
# 🔹 Main
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
