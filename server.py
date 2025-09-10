import os
import tempfile
from typing import List, Dict, Any

import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf


# --- App Setup ---
app = Flask(__name__)
CORS(app)


# --- Model Loading ---
_model = None
_reverse_mapping: Dict[int, str] = {}
_input_size = (224, 224)


def _load_model_and_mapping(model_name: str = "MobileNetV2"):
    global _model, _reverse_mapping
    if _model is not None and _reverse_mapping:
        return

    # Candidate filenames in priority order
    candidates = [
        (
            f"isl_transfer_{model_name.lower()}_final.keras",
            f"class_mapping_transfer_{model_name.lower()}.pkl",
        ),
        ("isl_model.h5", "class_mapping.pkl"),
    ]

    # Final fallback: try to auto-discover any .keras/.h5 with a mapping .pkl
    try:
        import glob
        keras_files = glob.glob("*.keras") + glob.glob("*.h5")
        pkl_files = glob.glob("*.pkl")
        for kf in keras_files:
            # Prefer a pkl that starts with 'class_mapping'
            mapping = None
            for pf in pkl_files:
                if os.path.basename(pf).startswith("class_mapping"):
                    mapping = pf
                    break
            if mapping:
                candidates.append((kf, mapping))
    except Exception:
        pass

    model_path = None
    mapping_path = None
    for m, c in candidates:
        if os.path.exists(m) and os.path.exists(c):
            model_path, mapping_path = m, c
            break

    if not model_path or not mapping_path:
        raise FileNotFoundError(
            "Model or mapping files not found. Place your model (.keras/.h5) and "
            "class mapping (.pkl) in the project root."
        )

    _model = tf.keras.models.load_model(model_path)

    import pickle
    with open(mapping_path, "rb") as f:
        class_mapping = pickle.load(f)
    _reverse_mapping = {v: k for k, v in class_mapping.items()}

    # Determine input size from model
    global _input_size
    try:
        ishape = _model.input_shape
        if isinstance(ishape, list):
            ishape = ishape[0]
        h, w = int(ishape[1]), int(ishape[2])
        if h > 0 and w > 0:
            _input_size = (w, h) if False else (h, w)  # keep (H,W)
            _input_size = (h, w)
    except Exception:
        _input_size = (224, 224)

    # Validate mapping size vs model output units
    try:
        out_shape = _model.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[-1]
        num_units = int(out_shape[-1])
        mapping_count = len(_reverse_mapping)
        if mapping_count != num_units:
            raise ValueError(
                f"Class mapping size ({mapping_count}) does not match model output units ({num_units}). "
                "Ensure the mapping .pkl corresponds to the loaded model."
            )
    except Exception as e:
        # Surface clear error so frontend can display it
        raise e


# --- Preprocessing ---
def _preprocess_image_from_bgr(image_bgr: np.ndarray, target_size=None) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("Invalid image provided")
    if target_size is None:
        target_size = _input_size
    img_bgr = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img = np.stack([img_gray, img_gray, img_gray], axis=-1)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def _predict_single_image(image_bgr: np.ndarray) -> Dict[str, Any]:
    _load_model_and_mapping()
    input_tensor = _preprocess_image_from_bgr(image_bgr)
    preds = _model.predict(input_tensor, verbose=0)
    predicted_class_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    if predicted_class_idx not in _reverse_mapping:
        raise ValueError(
            f"Predicted class index {predicted_class_idx} not found in class mapping. "
            "Mapping/model pair likely mismatched."
        )
    letter = _reverse_mapping[predicted_class_idx]
    return {"letter": str(letter), "confidence": confidence}


def _analyze_video(file_path: str, smoothing_window: int = 6, consecutive_frames: int = 2) -> List[Dict[str, Any]]:
    from collections import deque

    _load_model_and_mapping()
    if not os.path.exists(file_path):
        raise FileNotFoundError("Video file not found")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    prediction_history = deque(maxlen=smoothing_window)
    last_detected_char = None
    frame_counter = 0
    timeline: List[Dict[str, Any]] = []
    frame_index = 0

    chunk_size = 12  # frames per chunk for fallback summarization
    chunk_predictions: List[int] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = _preprocess_image_from_bgr(frame)
        prediction = _model.predict(input_tensor, verbose=0)
        predicted_class_idx = int(np.argmax(prediction[0]))

        prediction_history.append(predicted_class_idx)
        most_common_pred = int(np.bincount(list(prediction_history)).argmax())
        predicted_char = _reverse_mapping.get(most_common_pred, "?")

        if predicted_char == last_detected_char:
            frame_counter += 1
        else:
            frame_counter = 1

        if frame_counter >= consecutive_frames:
            if predicted_char != last_detected_char:
                timestamp_s = frame_index / fps
                timeline.append(
                    {
                        "letter": predicted_char,
                        "confidence": float(np.max(prediction[0])),
                        "timestamp": float(timestamp_s),
                    }
                )
                last_detected_char = predicted_char

        # Collect for fallback summarization
        chunk_predictions.append(most_common_pred)
        if len(chunk_predictions) >= chunk_size:
            # summarize this chunk even if not stable
            try:
                fallback_idx = int(np.bincount(chunk_predictions).argmax())
                fallback_char = _reverse_mapping.get(fallback_idx, "?")
                if not timeline or timeline[-1]["letter"] != fallback_char:
                    timestamp_s = frame_index / fps
                    timeline.append({
                        "letter": fallback_char,
                        "confidence": 0.0,
                        "timestamp": float(timestamp_s),
                    })
            except Exception:
                pass
            chunk_predictions = []

        frame_index += 1

    cap.release()
    return timeline


# --- Routes ---
@app.route("/api/health", methods=["GET"])
def health():
    try:
        _load_model_and_mapping()
        # Gather diagnostics
        try:
            ishape = _model.input_shape
            if isinstance(ishape, list):
                ishape = ishape[0]
            out_units = _model.output_shape
            if isinstance(out_units, list):
                out_units = out_units[-1]
            out_units = int(out_units[-1])
        except Exception:
            ishape = None
            out_units = None
        return jsonify({
            "status": "ok",
            "classes": len(_reverse_mapping),
            "input_shape": ishape,
            "output_units": out_units,
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict-image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Field name must be 'image'."}), 400

    file = request.files["image"]
    data = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Invalid image data"}), 400

    try:
        result = _predict_single_image(img_bgr)
        return jsonify({
            "letter": result["letter"],
            "confidence": result["confidence"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict-video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided. Field name must be 'video'."}), 400

    file = request.files["video"]
    tmp_dir = tempfile.mkdtemp(prefix="isl_vid_")
    tmp_path = os.path.join(tmp_dir, file.filename or "video.mp4")
    file.save(tmp_path)

    try:
        sequence = _analyze_video(tmp_path)
        return jsonify({"sequence": sequence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(tmp_dir):
                os.rmdir(tmp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    _load_model_and_mapping()
    app.run(host="0.0.0.0", port=port, debug=False)

