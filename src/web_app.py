"""Flask web UI backend for KUMPAS Dual-Mode FSL Two-Way Communication System."""

from __future__ import annotations

import base64
import csv
import json
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.serving import WSGIRequestHandler

from src.predictor import ModelPredictor
from src.config import (
    CLASSES,
    CONFIDENCE_THRESHOLD,
    DATA_DIR,
    FSL105_DIR,
    FSL105_LABEL_MAP_PATH,
    FSL105_MODEL_PATH,
    FSL105_SVM_PATH,
    FSL105_THUMBS_DIR,
    PHRASE_CONFIDENCE_THRESHOLD,
    SPACE_PAUSE_SECONDS,
)
from src.features import (
    HAND_CONNECTIONS,
    HolisticLandmarkExtractor,
    HandLandmarkExtractor,
    UPPER_BODY_CONNECTIONS,
)
from src.phrase_predictor import PhrasePredictor
from src.postprocess import PhraseSignProcessor, WordFormationProcessor
from src.translator import (
    autocomplete_word,
    clean_fsl_label,
    compose_sentence,
    get_lexicon_entry,
    match_listener_signs,
)

BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True

_lock = threading.Lock()
_predictor = ModelPredictor()
_extractor = HandLandmarkExtractor(running_mode="IMAGE")
_processor = WordFormationProcessor()
_last_hand_time: Optional[float] = None
_auto_spaced = False

# Lazy-loaded FSL-105 phrase communication components
_phrase_predictor: Optional[PhrasePredictor] = None
_holistic_extractor: Optional[HolisticLandmarkExtractor] = None
_phrase_processor = PhraseSignProcessor()
_last_top3_phrase: List[Dict[str, Any]] = []
_conversation: List[Dict[str, Any]] = []
_msg_counter = 0


def _get_phrase_predictor() -> Optional[PhrasePredictor]:
    global _phrase_predictor
    if _phrase_predictor is None and FSL105_MODEL_PATH.exists() and FSL105_LABEL_MAP_PATH.exists():
        _phrase_predictor = PhrasePredictor()
    return _phrase_predictor


def _get_holistic_extractor() -> HolisticLandmarkExtractor:
    global _holistic_extractor
    if _holistic_extractor is None:
        _holistic_extractor = HolisticLandmarkExtractor(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )
    return _holistic_extractor


def _decode_frame(data_url: str) -> Optional[np.ndarray]:
    if "," in data_url:
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _mirror_points(pts: List[List[float]]) -> List[List[float]]:
    return [[round(1.0 - float(pt[0]), 4), float(pt[1])] for pt in pts]


def _composer_payload(mode: str = "phrase") -> Dict[str, Any]:
    tokens = _phrase_processor.get_tokens()
    text_buf = _processor.get_text()
    if mode == "alphabet" and text_buf.strip():
        active_items: List[Any] = list(tokens) + [text_buf.strip()]
    elif tokens:
        active_items = list(tokens)
    elif text_buf.strip():
        active_items = [text_buf.strip()]
    else:
        active_items = []

    return {
        "tokens": tokens,
        "text_buffer": text_buf,
        "formatted_en": compose_sentence(active_items, lang="en") if active_items else "",
        "formatted_fil": compose_sentence(active_items, lang="fil") if active_items else "",
    }


def _load_test_demo_samples() -> List[Dict[str, Any]]:
    """Load curated held-out test.csv clips across categories for instant evaluation/demo."""
    test_csv = FSL105_DIR / "test.csv"
    if not test_csv.exists():
        return []

    priority_labels = {
        "GOOD MORNING", "HOW ARE YOU", "THANK YOU", "HELLO",
        "DON'T UNDERSTAND", "UNDERSTAND", "SLOW", "YES", "NO", "CORRECT",
        "DEAF", "HARD OF HEARING", "MOTHER", "FATHER",
        "COFFEE", "NO SUGAR", "HOT", "COLD", "CHICKEN", "RICE",
        "TOMORROW", "TODAY", "MONDAY", "BLUE", "RED", "ONE", "FIVE",
    }
    chosen: Dict[str, Dict[str, Any]] = {}
    fallback: List[Dict[str, Any]] = []

    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = clean_fsl_label(row["label"])
            item = {
                "vid_path": row["vid_path"].replace("\\", "/"),
                "id": int(row["id_label"]),
                "label": lbl,
                "category": row["category"].strip().upper(),
            }
            if lbl in priority_labels and lbl not in chosen:
                chosen[lbl] = item
            elif len(fallback) < 20:
                fallback.append(item)

    samples = sorted(chosen.values(), key=lambda x: (x["category"], x["id"]))
    if not samples:
        samples = fallback
    return samples[:24]


@app.route("/")
def index():
    phrase_ready = FSL105_MODEL_PATH.exists() and FSL105_LABEL_MAP_PATH.exists()
    return render_template(
        "index.html",
        classes=CLASSES,
        has_svm=_predictor.svm is not None,
        has_phrase_model=phrase_ready,
        has_phrase_svm=FSL105_SVM_PATH.exists(),
        confidence_threshold=int(CONFIDENCE_THRESHOLD * 100),
        phrase_threshold=int(PHRASE_CONFIDENCE_THRESHOLD * 100),
        hand_connections=[list(pair) for pair in HAND_CONNECTIONS],
        upper_body_connections=[list(pair) for pair in UPPER_BODY_CONNECTIONS],
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """A-Z Fingerspelling frame recognition with auto-CLAHE fallback & bilingual autocomplete."""
    global _last_hand_time, _auto_spaced
    payload = request.get_json(silent=True) or {}
    frame_b64 = payload.get("image")
    model_mode = payload.get("model", "Deep MLP")
    use_clahe = bool(payload.get("clahe", False))
    auto_clahe = bool(payload.get("auto_clahe", True))
    auto_space = bool(payload.get("auto_space", True))

    if not frame_b64:
        return jsonify({"error": "Missing image payload"}), 400

    frame = _decode_frame(frame_b64)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    t0 = time.perf_counter()
    now_mono = time.monotonic()
    with _lock:
        vec, landmarks, clahe_assisted = _extractor.extract_realtime(
            frame, use_clahe=use_clahe, auto_clahe_fallback=auto_clahe
        )
        top3 = []
        raw_label = None
        raw_conf = 0.0
        lm_points: List[List[float]] = []

        if vec is not None and landmarks is not None:
            _last_hand_time = now_mono
            _auto_spaced = False
            top3 = [
                {"label": lbl, "prob": round(prob, 4)}
                for lbl, prob in _predictor.predict_top_k(vec, mode=model_mode, k=3)
            ]
            raw_label = top3[0]["label"]
            raw_conf = top3[0]["prob"]
            lm_points = [[round(float(lm.x), 4), round(float(lm.y), 4)] for lm in landmarks]
        elif (
            auto_space
            and _last_hand_time is not None
            and not _auto_spaced
            and (now_mono - _last_hand_time) >= SPACE_PAUSE_SECONDS
        ):
            _processor.insert_space()
            _auto_spaced = True

        state = _processor.update(raw_label, raw_conf, now_mono)
        partial_word = _processor.get_current_partial_word()
        suggestions = autocomplete_word(partial_word, limit=6)
        comp = _composer_payload(mode="alphabet")

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return jsonify({
        "hand_detected": vec is not None,
        "clahe_assisted": clahe_assisted,
        "landmarks": lm_points,
        "top3": top3,
        "raw_label": state.raw_label,
        "raw_confidence": round(state.raw_confidence, 4),
        "smoothed_label": state.smoothed_label,
        "dwell_progress": round(state.dwell_progress, 4),
        "just_confirmed": state.just_confirmed,
        "text_buffer": state.text_buffer,
        "partial_word": partial_word,
        "suggestions": suggestions,
        "formatted_en": comp["formatted_en"],
        "formatted_fil": comp["formatted_fil"],
        "latency_ms": latency_ms,
    })


@app.route("/api/predict_phrase_frame", methods=["POST"])
def api_predict_phrase_frame():
    """Live FSL-105 Holistic phrase segmentation & recognition."""
    global _last_top3_phrase
    pred = _get_phrase_predictor()
    if pred is None:
        return jsonify({"error": "FSL-105 phrase model is not trained yet."}), 503

    payload = request.get_json(silent=True) or {}
    frame_b64 = payload.get("image")
    model_mode = payload.get("model", "FSLPhraseNet")
    use_clahe = bool(payload.get("clahe", False))
    auto_clahe = bool(payload.get("auto_clahe", True))
    mirrored = bool(payload.get("mirrored", True))
    manual_capture = bool(payload.get("manual_capture", False))

    if not frame_b64:
        return jsonify({"error": "Missing image payload"}), 400

    frame = _decode_frame(frame_b64)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    t0 = time.perf_counter()
    now_mono = time.monotonic()

    with _lock:
        ext = _get_holistic_extractor()
        vec_293, overlay, clahe_assisted = ext.extract(
            frame, use_clahe=use_clahe, auto_clahe_fallback=auto_clahe
        )
        state = _phrase_processor.update(
            vec_293,
            hands_raised=bool(overlay["hands_raised"]),
            timestamp_sec=now_mono,
            force_finalize=manual_capture,
        )

        just_confirmed = None
        top3 = list(_last_top3_phrase)

        if state.ready_sequence is not None:
            top3 = pred.predict_sequence_top_k(state.ready_sequence, mode=model_mode, k=3)
            _last_top3_phrase = list(top3)
            if state.capture_state == "finalized" and top3:
                just_confirmed = _phrase_processor.try_commit_prediction(
                    top3[0], now_mono, bypass_threshold=True
                )
        elif state.capture_state == "idle" and not overlay["has_hands"]:
            top3 = []
            _last_top3_phrase = []

        pose_pts = _mirror_points(overlay["pose"]) if mirrored else overlay["pose"]
        lh_pts = _mirror_points(overlay["left_hand"]) if mirrored else overlay["left_hand"]
        rh_pts = _mirror_points(overlay["right_hand"]) if mirrored else overlay["right_hand"]
        comp = _composer_payload(mode="phrase")

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return jsonify({
        "has_hands": overlay["has_hands"],
        "hands_raised": overlay["hands_raised"],
        "clahe_assisted": clahe_assisted,
        "pose": pose_pts,
        "left_hand": lh_pts,
        "right_hand": rh_pts,
        "capture_state": state.capture_state,
        "active_frames": state.active_frames,
        "progress": round(state.progress, 4),
        "top3": top3,
        "just_confirmed": just_confirmed,
        "tokens": comp["tokens"],
        "formatted_en": comp["formatted_en"],
        "formatted_fil": comp["formatted_fil"],
        "latency_ms": latency_ms,
    })


@app.route("/api/predict_phrase_video", methods=["POST"])
def api_predict_phrase_video():
    """Classify an uploaded video clip or a curated held-out test.csv clip."""
    pred = _get_phrase_predictor()
    if pred is None:
        return jsonify({"error": "FSL-105 phrase model is not trained yet."}), 503

    t0 = time.perf_counter()
    model_mode = "FSLPhraseNet"
    append_token = True
    ground_truth: Optional[str] = None
    tmp_path: Optional[Path] = None

    if "video" in request.files:
        file_obj = request.files["video"]
        if not file_obj or not file_obj.filename:
            return jsonify({"error": "Empty video upload"}), 400
        model_mode = request.form.get("model", "FSLPhraseNet")
        append_token = request.form.get("append", "true").lower() != "false"
        suffix = Path(file_obj.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file_obj.save(tmp.name)
            tmp_path = Path(tmp.name)
        video_path = tmp_path
    else:
        payload = request.get_json(silent=True) or {}
        sample_rel = str(payload.get("sample_path", "")).strip()
        model_mode = payload.get("model", "FSLPhraseNet")
        append_token = bool(payload.get("append", True))
        if not sample_rel:
            return jsonify({"error": "Provide a video upload or sample_path"}), 400

        clips_root = (FSL105_DIR / "clips").resolve()
        candidate = (FSL105_DIR / sample_rel).resolve()
        try:
            candidate.relative_to(clips_root)
        except ValueError:
            return jsonify({"error": "Invalid sample_path"}), 400
        if not candidate.is_file():
            return jsonify({"error": f"Sample video not found: {sample_rel}"}), 404
        video_path = candidate

        # Look up ground truth from class folder ID
        try:
            cid = int(candidate.parent.name)
            if 0 <= cid < len(pred.label_entries):
                ground_truth = pred.label_entries[cid]["label"]
        except ValueError:
            pass

    try:
        with _lock:
            ext = _get_holistic_extractor()
            top5, stats = pred.predict_video_top_k(video_path, extractor=ext, mode=model_mode, k=5)
            top1 = top5[0] if top5 else None
            if append_token and top1 is not None:
                _phrase_processor.append_manual_token(
                    label=top1["label"],
                    class_id=top1["id"],
                    category=top1["category"],
                    confidence=top1["prob"],
                )
            comp = _composer_payload(mode="phrase")
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return jsonify({
        "top5": top5,
        "top1": top1,
        "ground_truth": ground_truth,
        "extraction_stats": stats,
        "tokens": comp["tokens"],
        "formatted_en": comp["formatted_en"],
        "formatted_fil": comp["formatted_fil"],
        "latency_ms": latency_ms,
    })


@app.route("/api/buffer/<action>", methods=["POST"])
def api_buffer(action: str):
    with _lock:
        if action == "space":
            text = _processor.insert_space()
        elif action == "backspace":
            text = _processor.backspace()
        elif action == "clear":
            text = _processor.clear()
        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400
        comp = _composer_payload(mode="alphabet")
        partial_word = _processor.get_current_partial_word()
        suggestions = autocomplete_word(partial_word, limit=6)
    return jsonify({
        "text_buffer": text,
        "partial_word": partial_word,
        "suggestions": suggestions,
        "formatted_en": comp["formatted_en"],
        "formatted_fil": comp["formatted_fil"],
    })


@app.route("/api/composer/<action>", methods=["POST"])
def api_composer(action: str):
    """Manage active phrase/word composer and two-way conversation history."""
    global _msg_counter
    payload = request.get_json(silent=True) or {}

    with _lock:
        if action == "pick_phrase":
            label = clean_fsl_label(str(payload.get("label", "")))
            if not label:
                return jsonify({"error": "Missing label"}), 400
            entry = get_lexicon_entry(label)
            cid = int(payload.get("id", entry["id"]))
            category = str(payload.get("category", entry["category"]))
            prob = float(payload.get("prob", 1.0))
            _phrase_processor.append_manual_token(
                label=label, class_id=cid, category=category, confidence=prob
            )
            comp = _composer_payload(mode="phrase")
            return jsonify(comp)

        if action == "complete_word":
            word = str(payload.get("word", "")).strip()
            _processor.complete_current_word(word)
            comp = _composer_payload(mode="alphabet")
            return jsonify({**comp, "suggestions": []})

        if action == "undo":
            mode = payload.get("mode", "phrase")
            if mode == "alphabet":
                _processor.backspace()
            else:
                _phrase_processor.undo_token()
            return jsonify(_composer_payload(mode=mode))

        if action == "clear":
            _phrase_processor.clear_tokens()
            _processor.clear()
            return jsonify(_composer_payload(mode="phrase"))

        if action == "reset_phrase_capture":
            _phrase_processor.reset_capture()
            return jsonify({"ok": True})

        if action == "send":
            mode = payload.get("mode", "phrase")
            comp = _composer_payload(mode=mode)
            if not comp["formatted_en"]:
                return jsonify({"error": "Composer is empty"}), 400
            _msg_counter += 1
            msg = {
                "id": _msg_counter,
                "sender": "signer",
                "text_en": comp["formatted_en"],
                "text_fil": comp["formatted_fil"],
                "tokens": comp["tokens"] if comp["tokens"] else [{"type": "text", "label": comp["text_buffer"].strip()}],
                "timestamp": datetime.now().strftime("%H:%M"),
            }
            _conversation.append(msg)
            _phrase_processor.clear_tokens()
            _processor.clear()
            return jsonify({
                "message": msg,
                "conversation": list(_conversation),
                **_composer_payload(mode=mode),
            })

        if action == "listener_reply":
            text = str(payload.get("text", "")).strip()
            if not text:
                return jsonify({"error": "Reply text cannot be empty"}), 400
            _msg_counter += 1
            matched_signs = match_listener_signs(text)
            msg = {
                "id": _msg_counter,
                "sender": "listener",
                "text": text,
                "matched_signs": matched_signs,
                "timestamp": datetime.now().strftime("%H:%M"),
            }
            _conversation.append(msg)
            return jsonify({
                "message": msg,
                "conversation": list(_conversation),
            })

        if action == "reset_chat":
            _conversation.clear()
            return jsonify({"conversation": []})

    return jsonify({"error": f"Unknown composer action: {action}"}), 400


@app.route("/api/conversation", methods=["GET"])
def api_conversation():
    with _lock:
        comp = _composer_payload(mode="phrase")
        return jsonify({
            "conversation": list(_conversation),
            **comp,
        })


@app.route("/api/fsl105/catalog", methods=["GET"])
def api_fsl105_catalog():
    signs: List[Dict[str, Any]] = []
    if FSL105_LABEL_MAP_PATH.exists():
        raw_map = json.loads(FSL105_LABEL_MAP_PATH.read_text(encoding="utf-8"))
        for i in range(len(raw_map)):
            item = raw_map[str(i)]
            lex = get_lexicon_entry(item["label"])
            signs.append({
                "id": int(item["id"]),
                "label": item["label"],
                "category": item["category"],
                "en": lex["en_sentence"],
                "fil": lex["fil_sentence"],
                "has_thumb": (FSL105_THUMBS_DIR / f"{int(item['id'])}.jpg").exists(),
                "thumb_url": f"/api/fsl105/thumb/{int(item['id'])}",
                "video_url": f"/api/fsl105/video/{int(item['id'])}",
            })
    categories = list(dict.fromkeys(s["category"] for s in signs))
    return jsonify({
        "signs": signs,
        "categories": categories,
        "test_samples": _load_test_demo_samples(),
    })


@app.route("/api/fsl105/thumb/<int:class_id>")
def api_fsl105_thumb(class_id: int):
    thumb = FSL105_THUMBS_DIR / f"{class_id}.jpg"
    if not thumb.exists():
        return jsonify({"error": "Thumbnail not found"}), 404
    return send_file(thumb, mimetype="image/jpeg")


@app.route("/api/fsl105/video/<int:class_id>")
def api_fsl105_video(class_id: int):
    clip_dir = FSL105_DIR / "clips" / str(class_id)
    if not clip_dir.is_dir():
        return jsonify({"error": "Video directory not found"}), 404
    clips = sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in {".mov", ".mp4"})
    if not clips:
        return jsonify({"error": "No clip found"}), 404
    mime = "video/mp4" if clips[0].suffix.lower() in {".mp4", ".mov"} else "application/octet-stream"
    return send_file(clips[0], mimetype=mime, conditional=True)


@app.route("/api/reference/<letter>")
def api_reference(letter: str):
    letter = letter.upper()
    if letter not in CLASSES:
        return jsonify({"error": "Invalid letter"}), 404

    idx = max(0, request.args.get("idx", default=0, type=int)) % 4
    ref_img = DATA_DIR / "reference" / letter / f"{idx}.jpg"
    if not ref_img.exists():
        return jsonify({"error": "Reference image not found"}), 404
    return send_file(ref_img, mimetype="image/jpeg")


class KeepAliveRequestHandler(WSGIRequestHandler):
    """HTTP/1.1 keep-alive handler that prevents per-frame socket exhaustion and log floods."""

    protocol_version = "HTTP/1.1"
    timeout = 30

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        # Silence noisy 200 logs on high-frequency webcam inference endpoints
        path = getattr(self, "path", "")
        if str(code) == "200" and path in ("/api/predict", "/api/predict_phrase_frame"):
            return
        super().log_request(code, size)


def main() -> None:
    try:
        import uvicorn
        from uvicorn.middleware.wsgi import WSGIMiddleware

        print("KUMPAS Communication Server running at http://127.0.0.1:5000 (HTTP/1.1 Keep-Alive)")
        uvicorn.run(
            WSGIMiddleware(app),
            host="127.0.0.1",
            port=5000,
            timeout_keep_alive=30,
            access_log=False,
            log_level="warning",
        )
    except ImportError:
        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            threaded=True,
            request_handler=KeepAliveRequestHandler,
        )


if __name__ == "__main__":
    main()
