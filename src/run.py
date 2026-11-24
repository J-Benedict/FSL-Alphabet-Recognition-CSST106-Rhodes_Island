"""Unified single-command FSL pipeline supporting A-Z fingerspelling and FSL-105 phrases."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.predictor import ModelPredictor
from src.config import CAMERA_HEIGHT, CAMERA_WIDTH, CONFIDENCE_THRESHOLD, TARGET_FPS
from src.features import (
    HandLandmarkExtractor,
    HolisticLandmarkExtractor,
    apply_clahe,
    draw_hand_skeleton,
    draw_holistic_skeleton,
)
from src.phrase_predictor import PhrasePredictor
from src.postprocess import PhraseSignProcessor, ProcessorState, WordFormationProcessor
from src.translator import compose_sentence

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def preprocess(
    frame: np.ndarray,
    extractor: HandLandmarkExtractor,
    use_clahe: bool = False,
    allow_pad_fallback: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[Sequence], np.ndarray]:
    """Stage 1: Optional CLAHE + MediaPipe 21-keypoint extraction & 63-dim normalization."""
    if allow_pad_fallback:
        return extractor.extract_with_fallback(frame, use_clahe=use_clahe)
    proc = apply_clahe(frame) if use_clahe else frame
    vec, landmarks, _ = extractor.extract_realtime(proc, use_clahe=False, auto_clahe_fallback=True)
    return vec, landmarks, proc


def infer(
    vec: Optional[np.ndarray],
    predictor: ModelPredictor,
    model_mode: str = "Deep MLP",
) -> Tuple[Optional[str], float]:
    """Stage 2: Classify 63-dim vector into A-Z label and probability."""
    if vec is None:
        return None, 0.0
    return predictor.predict(vec, mode=model_mode)


def postprocess(
    label: Optional[str],
    confidence: float,
    processor: Optional[WordFormationProcessor],
    timestamp_sec: float,
) -> ProcessorState:
    """Stage 3: Apply confidence thresholding, temporal smoothing, and dwell confirmation."""
    if processor is None:
        return ProcessorState(
            raw_label=label,
            raw_confidence=confidence,
            smoothed_label=label,
            dwell_progress=1.0 if label else 0.0,
            just_confirmed=None,
            text_buffer=label or "",
        )
    return processor.update(label, confidence, timestamp_sec)


def render(
    frame: np.ndarray,
    landmarks: Optional[Sequence],
    state: ProcessorState,
    show_text_bar: bool = True,
) -> np.ndarray:
    """Stage 4: Draw hand skeleton overlay, predicted label, confidence, and buffer."""
    out = draw_hand_skeleton(frame, landmarks) if landmarks is not None else frame.copy()
    h, w = out.shape[:2]

    cv2.rectangle(out, (0, 0), (w, 36), (15, 20, 27), -1)
    if state.raw_label is not None:
        active = state.smoothed_label or state.raw_label
        msg = f"Pred: {active} ({state.raw_confidence * 100:.1f}%)"
        color = (56, 189, 248) if state.raw_confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    else:
        msg = "No hand detected"
        color = (0, 165, 255)

    cv2.putText(out, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    if show_text_bar and state.text_buffer:
        cv2.rectangle(out, (0, h - 38), (w, h), (15, 20, 27), -1)
        cv2.putText(
            out,
            f"Text: {state.text_buffer[-36:]}",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 235, 242),
            2,
            cv2.LINE_AA,
        )
    return out


def run_images(
    inputs: list[Path],
    output_dir: Path,
    model_mode: str,
    use_clahe: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = ModelPredictor()
    csv_path = output_dir / "results.csv"

    with HandLandmarkExtractor(running_mode="IMAGE", min_detection_confidence=0.3) as ext, open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["file", "hand_detected", "predicted_label", "confidence", "latency_ms"])

        for img_path in inputs:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Skipping unreadable file: {img_path}")
                continue
            t0 = time.perf_counter()
            vec, landmarks, proc = preprocess(
                frame, ext, use_clahe=use_clahe, allow_pad_fallback=True
            )
            label, conf = infer(vec, predictor, model_mode=model_mode)
            state = postprocess(label, conf, processor=None, timestamp_sec=0.0)
            annotated = render(proc, landmarks, state, show_text_bar=False)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            out_img = output_dir / f"out_{img_path.stem}.png"
            cv2.imwrite(str(out_img), annotated)
            writer.writerow([img_path.name, vec is not None, label or "NONE", f"{conf:.4f}", f"{latency_ms:.2f}"])
            print(f"{img_path.name} -> {label or 'NONE'} ({conf * 100:.1f}%, {latency_ms:.1f} ms) -> {out_img.name}")

    print(f"Saved CSV summary: {csv_path}")


def run_video(
    video_path: Path,
    output_path: Path,
    model_mode: str,
    use_clahe: bool,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    predictor = ModelPredictor()
    processor = WordFormationProcessor()
    frame_idx = 0

    with HandLandmarkExtractor(running_mode="IMAGE") as ext:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts_sec = frame_idx / fps
            vec, landmarks, proc = preprocess(frame, ext, use_clahe=use_clahe)
            label, conf = infer(vec, predictor, model_mode=model_mode)
            state = postprocess(label, conf, processor=processor, timestamp_sec=ts_sec)
            annotated = render(proc, landmarks, state, show_text_bar=True)
            writer.write(annotated)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"Processed {frame_idx} frames -> {output_path} | Transcribed text: {processor.get_text()!r}")


def run_webcam(cam_index: int, model_mode: str, use_clahe: bool) -> None:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {cam_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    predictor = ModelPredictor()
    processor = WordFormationProcessor()
    start = time.monotonic()

    with HandLandmarkExtractor(running_mode="IMAGE") as ext:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            vec, landmarks, proc = preprocess(frame, ext, use_clahe=use_clahe)
            label, conf = infer(vec, predictor, model_mode=model_mode)
            state = postprocess(label, conf, processor=processor, timestamp_sec=time.monotonic() - start)
            annotated = render(proc, landmarks, state, show_text_bar=True)
            cv2.imshow("FSL Alphabet Pipeline (Press Q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


def run_phrase_video(
    video_path: Path,
    output_path: Path,
    model_mode: str,
    use_clahe: bool,
) -> None:
    """Classify an FSL-105 phrase video and write an annotated video with EN/FIL output."""
    predictor = PhrasePredictor()
    with HolisticLandmarkExtractor(static_image_mode=False, model_complexity=1) as ext:
        top3, stats = predictor.predict_video_top_k(video_path, extractor=ext, mode=model_mode, k=3)

    top1 = top3[0]
    en_sentence = compose_sentence([{"type": "phrase", "label": top1["label"]}], lang="en")
    fil_sentence = compose_sentence([{"type": "phrase", "label": top1["label"]}], lang="fil")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    with HolisticLandmarkExtractor(static_image_mode=False, model_complexity=1) as ext:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, overlay, _ = ext.extract(frame, use_clahe=use_clahe, auto_clahe_fallback=True)
            annotated = draw_holistic_skeleton(frame, overlay)
            cv2.rectangle(annotated, (0, 0), (w, 36), (15, 20, 27), -1)
            cv2.putText(
                annotated,
                f"FSL-105: {top1['label']} ({top1['prob'] * 100:.1f}%) [{top1['category']}]",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 174, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(annotated, (0, h - 38), (w, h), (15, 20, 27), -1)
            cv2.putText(
                annotated,
                f"EN: {en_sentence} | FIL: {fil_sentence}",
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 240, 245),
                1,
                cv2.LINE_AA,
            )
            writer.write(annotated)

    cap.release()
    writer.release()

    print(f"Video: {video_path.name} ({stats['active_frames']}/{stats['sampled_frames']} active frames)")
    print("Top-3 Predictions:")
    for rank, item in enumerate(top3, start=1):
        print(f"  {rank}. {item['label']} ({item['prob'] * 100:.1f}%) [{item['category']}]")
    print(f"Natural EN : {en_sentence}")
    print(f"Natural FIL: {fil_sentence}")
    print(f"Saved annotated video -> {output_path}")


def run_phrase_webcam(cam_index: int, model_mode: str, use_clahe: bool) -> None:
    """Live webcam FSL-105 phrase recognition & sentence builder."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {cam_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    predictor = PhrasePredictor()
    processor = PhraseSignProcessor()
    start = time.monotonic()
    last_pred_label = "-"

    with HolisticLandmarkExtractor(static_image_mode=False, model_complexity=1) as ext:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now_sec = time.monotonic() - start
            vec, overlay, _ = ext.extract(frame, use_clahe=use_clahe, auto_clahe_fallback=True)
            state = processor.update(vec, overlay["hands_raised"], now_sec)
            if state.ready_sequence is not None:
                top3 = predictor.predict_sequence_top_k(state.ready_sequence, mode=model_mode, k=3)
                if top3:
                    last_pred_label = f"{top3[0]['label']} ({top3[0]['prob']*100:.0f}%)"
                    if state.capture_state == "finalized":
                        processor.try_commit_prediction(top3[0], now_sec)

            annotated = draw_holistic_skeleton(cv2.flip(frame, 1), {
                **overlay,
                "pose": [[1.0 - p[0], p[1]] for p in overlay["pose"]],
                "left_hand": [[1.0 - p[0], p[1]] for p in overlay["left_hand"]],
                "right_hand": [[1.0 - p[0], p[1]] for p in overlay["right_hand"]],
            })
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w, 36), (15, 20, 27), -1)
            cv2.putText(
                annotated,
                f"State: {state.capture_state.upper()} ({state.active_frames}f) | {last_pred_label}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 174, 255),
                2,
                cv2.LINE_AA,
            )
            sentence = compose_sentence(processor.get_tokens(), lang="en")
            if sentence:
                cv2.rectangle(annotated, (0, h - 38), (w, h), (15, 20, 27), -1)
                cv2.putText(
                    annotated,
                    sentence[-48:],
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (240, 244, 248),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("KUMPAS FSL-105 Phrase Pipeline (Q=quit, C=clear)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                processor.clear_tokens()

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="FSL End-to-End Pipeline Runner")
    parser.add_argument("--mode", default="alphabet", choices=["alphabet", "phrase"], help="Recognition mode")
    parser.add_argument("--input", type=Path, help="Image file, image folder, or video file")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory or .mp4 file")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam index (e.g. 0)")
    parser.add_argument("--model", default=None, help="Classifier (Deep MLP / Baseline SVM / FSLPhraseNet / FSL-105 SVM)")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE lighting enhancement")
    args = parser.parse_args()

    if args.mode == "phrase":
        model_mode = args.model or "FSLPhraseNet"
        if args.webcam is not None:
            run_phrase_webcam(args.webcam, model_mode, args.clahe)
            return
        if args.input is None or args.input.suffix.lower() not in VIDEO_EXTS:
            parser.error("Phrase mode requires --input <video_file> or --webcam <index>.")
        out_vid = args.output if args.output.suffix.lower() in VIDEO_EXTS else args.output / f"out_{args.input.name}"
        run_phrase_video(args.input, out_vid, model_mode, args.clahe)
        return

    model_mode = args.model or "Deep MLP"
    if args.webcam is not None:
        run_webcam(args.webcam, model_mode, args.clahe)
        return

    if args.input is None:
        parser.error("Specify either --input <path> or --webcam <index>.")

    if args.input.is_dir():
        imgs = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        run_images(imgs, args.output, model_mode, args.clahe)
    elif args.input.suffix.lower() in VIDEO_EXTS:
        out_vid = args.output if args.output.suffix.lower() in VIDEO_EXTS else args.output / f"out_{args.input.name}"
        run_video(args.input, out_vid, model_mode, args.clahe)
    elif args.input.suffix.lower() in IMAGE_EXTS:
        run_images([args.input], args.output, model_mode, args.clahe)
    else:
        parser.error(f"Unsupported input: {args.input}")


if __name__ == "__main__":
    main()
