from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas y orígenes


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


def close_panorama(image, blend_ratio=0.02):
    """Une suavemente los bordes izquierdo y derecho de una imagen panorámica equirectangular."""
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    blend_width = max(10, int(w * blend_ratio))

    left = image[:, :blend_width].astype(np.float32)
    right = image[:, -blend_width:].astype(np.float32)

    alpha = np.linspace(0, 1, blend_width).reshape(1, -1, 1)
    blended_left = left * (1 - alpha) + right * alpha
    blended_right = right * (1 - alpha) + left * alpha

    result = image.copy()
    result[:, :blend_width] = blended_left
    result[:, -blend_width:] = blended_right

    return result.astype(np.uint8)


def preprocess_image(img, max_size=1600):
    """
    Mejora la estabilidad del stitching:
    - Reduce resolución si es muy grande.
    - Aplica equalización de histograma para mejorar contraste.
    - Descarta imágenes con poca textura (muy planas).
    """
    if img is None:
        return None

    # Reducción adaptativa si la imagen es muy grande
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Convertir a gris para análisis de textura
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Medir nivel de detalle
    contrast = np.std(gray)
    if contrast < 10:  # umbral bajo → imagen muy plana
        print("Descartando imagen por baja textura (σ =", contrast, ")")
        return None

    # Equalización de histograma (solo para detección de features, no afecta color)
    gray_eq = cv2.equalizeHist(gray)
    img_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    return img_eq


@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    if not data or "photos" not in data:
        return jsonify({"error": "No photos provided"}), 400

    images = []
    for item in data["photos"]:
        img = base64_to_cv2image(item["photo"])
        processed = preprocess_image(img)
        if processed is not None:
            images.append(processed)

    if len(images) < 2:
        return jsonify({"error": "Need at least 2 valid images for stitching"}), 400

    # Usa el modo PANORAMA explícitamente (más tolerante)
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, stitched = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        print("Error en el stitching:", status)
        return jsonify({"error": f"Stitching failed (code {status})"}), 500

    stitched = close_panorama(stitched)
    result_b64 = cv2image_to_base64(stitched)

    return jsonify({"image": result_b64})


@app.route("/", methods=["GET"])
def index():
    return "Stitching backend is running (optimized version)."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
