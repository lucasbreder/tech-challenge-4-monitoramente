"""
Modelo YOLOv8 para análise de vídeos cirúrgicos de saúde da mulher.

Critérios do edital atendidos:
- Detecção de instrumentos cirúrgicos ginecológicos
- Detecção de sangramento anômalo durante procedimentos
- Identificação de áreas críticas em cirurgias

O modelo é treinado no dataset cirúrgico do Roboflow Universe
(1748 imagens, 21 classes de instrumentos).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

from src.config import resolve_device, settings

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


@dataclass
class DetectionResult:
    class_name: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]
    frame_number: int
    timestamp_seconds: float

    @property
    def is_anomaly(self) -> bool:
        return self.confidence > 0.5


@dataclass
class VideoAnalysisReport:
    file_path: Path
    video_duration_seconds: float
    total_frames: int
    frames_analyzed: int
    detections: list[DetectionResult] = field(default_factory=list)

    @property
    def anomaly_count(self) -> int:
        return sum(1 for d in self.detections if d.is_anomaly)

    @property
    def anomaly_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for d in self.detections:
            if d.is_anomaly:
                summary[d.class_name] = summary.get(d.class_name, 0) + 1
        return summary


class _SuppressStderr:
    def __enter__(self):
        self._fd = os.dup(2)
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 2)

    def __exit__(self, *_):
        if self._fd is not None:
            os.dup2(self._fd, 2)
            os.close(self._null)
            self._fd = None

_suppress = _SuppressStderr()


class YOLODetector:
    """
    Detector YOLOv8 para vídeos cirúrgicos de saúde da mulher.

    Usa modelo treinado no dataset cirúrgico do Roboflow (21 instrumentos).
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
    ):
        self.model_path = model_path or settings.model.yolo_model_path
        self.confidence = confidence_threshold or settings.model.yolo_confidence_threshold
        self.iou = iou_threshold or settings.model.yolo_iou_threshold
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        custom_path = Path(self.model_path)
        if custom_path.exists():
            logger.info(f"Carregando modelo cirúrgico: {self.model_path}")
            self._model = YOLO(str(custom_path))
        else:
            logger.warning(f"Modelo não encontrado: {self.model_path}")
            logger.info("Treine o modelo primeiro:")
            logger.info("  python convert_obb_dataset.py")
            logger.info("  python train.py --config data/dataset/yolo/data.yaml --epochs 30 --batch 8")
            logger.info("Usando YOLOv8n COCO como fallback...")
            self._model = YOLO("yolov8n.pt")
        device = resolve_device(settings.device)
        self._model.to(device)
        logger.info(f"Modelo carregado no dispositivo: {device}")

    def process_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> list[DetectionResult]:
        results = self.model.predict(
            source=frame, conf=self.confidence, iou=self.iou, verbose=False,
        )
        detections: list[DetectionResult] = []
        timestamp = frame_number / fps if fps > 0 else 0.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                confidence = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                class_name = self.model.names.get(cls_id, f"cls_{cls_id}")

                detections.append(DetectionResult(
                    class_name=class_name, class_id=cls_id,
                    confidence=confidence, bbox=bbox,
                    frame_number=frame_number, timestamp_seconds=timestamp,
                ))

        return detections

    def analyze_video(
        self,
        video_path: str | Path,
        sample_every_n_frames: int = 10,
        max_frames: int | None = None,
        progress_callback: callable = None,
    ) -> VideoAnalysisReport:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        report = VideoAnalysisReport(
            file_path=video_path, video_duration_seconds=duration,
            total_frames=total_frames, frames_analyzed=0,
        )

        logger.info(f"Analisando: {video_path.name} ({duration:.1f}s, {total_frames} frames)")

        frame_idx = 0
        frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

        from tqdm import tqdm
        with tqdm(total=frames_to_process, desc="Processando") as pbar:
            while frame_idx < frames_to_process:
                with _suppress:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_every_n_frames == 0:
                    detections = self.process_frame(frame, frame_idx, fps)
                    report.detections.extend(detections)
                    report.frames_analyzed += 1

                    for d in detections:
                        if d.is_anomaly:
                            logger.info(
                                f"[{d.timestamp_seconds:.1f}s] {d.class_name} "
                                f"(conf: {d.confidence:.2f})"
                            )

                frame_idx += 1
                pbar.update(1)
                if progress_callback:
                    progress_callback(frame_idx, frames_to_process)

        cap.release()
        logger.info(
            f"Concluído: {report.frames_analyzed} frames, "
            f"{len(report.detections)} detecções, {report.anomaly_count} anomalias"
        )
        return report

    def annotate_frame(self, frame: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 0, 255) if det.is_anomaly else (0, 255, 0)
            if x2 > 0 and y2 > 0:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(annotated, label, (x1 + 2, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated

    def live_detect(
        self, video_path: str | Path, sample_every_n_frames: int = 3,
        window_name: str = "Deteccao Cirurgica",
    ) -> None:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(w, 960), min(h, 720))

        frame_idx, paused = 0, False
        last_detections: list[DetectionResult] = []
        delay_ms = max(1, int(1000 / fps))

        logger.info(f"Live: {video_path.name} | ESPAÇO=pausar Q=sair →=avançar")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_every_n_frames == 0:
                    last_detections = self.process_frame(frame, frame_idx, fps)

                annotated = self.annotate_frame(frame, last_detections)
                info = f"Frame {frame_idx}/{total_frames} | {len(last_detections)} deteccoes"
                cv2.putText(annotated, info, (10, annotated.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(window_name, annotated)
                frame_idx += 1

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = not paused

        cap.release()
        cv2.destroyAllWindows()

    def azure_live_detect(
        self, video_path: str | Path, interval_seconds: float = 1.0,
        window_name: str = "Azure Vision - Deteccao Cirurgica",
    ) -> None:
        """Modo live com Azure Vision API: sobrepõe captions a cada N segundos."""
        from src.services.azure_services import AzureVisionService

        azure = AzureVisionService()
        if not azure.is_available:
            logger.error("Azure Vision não configurado. Configure AZURE_VISION_KEY no .env")
            return

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(w, 960), min(h, 720))

        frame_idx, paused = 0, False
        last_result = ""
        last_update_frame = -999
        delay_ms = max(1, int(1000 / fps))
        azure_interval = max(1, int(fps * interval_seconds))

        import tempfile, time

        logger.info(f"Azure Vision Live: {video_path.name} | atualiza a cada {interval_seconds}s")
        logger.info("ESPAÇO=pausar Q=sair")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx - last_update_frame >= azure_interval:
                    last_update_frame = frame_idx
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            cv2.imwrite(tmp.name, frame)
                            tmp_path = tmp.name

                        result = azure.analyze_image(tmp_path)
                        caption = result.get("caption", "")
                        tags = [str(t) for t in result.get("tags", []) if str(t) != "values"]

                        if caption or tags:
                            last_result = f"Azure [{frame_idx/fps:.1f}s]: {caption}"
                            if tags:
                                last_result += f"\ntags: {', '.join(tags[:5])}"
                            logger.info(last_result.replace('\n', ' | '))
                        else:
                            last_result = f"Frame {frame_idx}/{total_frames}"
                    except Exception as e:
                        last_result = f"Azure erro: {e}"
                    finally:
                        if tmp_path:
                            __import__("pathlib").Path(tmp_path).unlink(missing_ok=True)

                overlay = frame.copy()
                y = 30
                for line in last_result.split("\n"):
                    cv2.putText(overlay, line[:100], (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y += 25

                info = f"Frame {frame_idx}/{total_frames} | {fps:.0f} FPS"
                cv2.putText(overlay, info, (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow(window_name, overlay)
                frame_idx += 1

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = not paused

        cap.release()
        cv2.destroyAllWindows()
        logger.info("Azure Vision Live encerrado")

    def get_keyframes(
        self, video_path: str | Path, num_keyframes: int = 6,
        sample_every_n_frames: int = 10,
    ) -> list[tuple[np.ndarray, float]]:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_dets: list[tuple[float, np.ndarray, list[DetectionResult]]] = []
        frame_idx = 0

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n_frames == 0:
                dets = self.process_frame(frame, frame_idx, fps)
                if dets:
                    all_dets.append((frame_idx / fps, frame.copy(), dets))
            frame_idx += 1

        cap.release()
        all_dets.sort(key=lambda x: len(x[2]), reverse=True)
        keyframes = sorted(all_dets[:num_keyframes], key=lambda x: x[0])

        result = []
        for ts, frame, dets in keyframes:
            annotated = self.annotate_frame(frame, dets)
            cv2.putText(annotated, f"t={ts:.1f}s  {len(dets)} deteccoes",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            result.append((annotated, ts))
        return result

    def generate_annotated_video(
        self, video_path: str | Path, output_path: str | Path,
        sample_every_n_frames: int = 10,
    ) -> Path:
        video_path = Path(video_path)
        output_path = Path(output_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        frame_idx = 0
        last_detections: list[DetectionResult] = []

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n_frames == 0:
                last_detections = self.process_frame(frame, frame_idx, fps)
            annotated = self.annotate_frame(frame, last_detections)
            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()
        logger.info(f"Vídeo anotado: {output_path}")
        return output_path

    def finetune(
        self, data_yaml: str | Path, epochs: int = 50,
        imgsz: int = 640, batch: int = 16,
    ) -> None:
        device = resolve_device(settings.device)
        logger.info(f"Fine-tuning: epochs={epochs}, imgsz={imgsz}, device={device}")
        model = YOLO("yolov8n.pt")
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            hsv_h=0.03,
            hsv_s=0.5,
            hsv_v=0.4,
            degrees=15.0,
            translate=0.15,
            scale=0.6,
            shear=5.0,
            perspective=0.001,
            flipud=0.3,
            fliplr=0.5,
            mosaic=0.8,
            mixup=0.15,
            erasing=0.2,
        )
        metrics = model.val()
        logger.info(f"Validação: mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")

        save_path = Path("models") / "yolov8_custom.pt"
        save_path.parent.mkdir(exist_ok=True)
        model.save(str(save_path))
        self._model = model
        logger.info(f"Modelo salvo: {save_path}")
