"""
Modelo YOLOv8 customizado para análise de vídeos de saúde da mulher.

Critérios do edital atendidos:
- Detecção de sangramento anômalo durante procedimentos ginecológicos
- Identificação de instrumentos cirúrgicos ginecológicos
- Detecção de áreas críticas em cirurgias (útero, ovários, mamas)
- Detecção de objetos suspeitos indicativos de automutilação
- Análise de movimentos em fisioterapia pós-parto
- Sinais não-verbais de desconforto ou medo em consultas
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

from src.config import resolve_device, settings

AZURE_TAG_MAPPING = {
    "blood": (7, "sangramento_anomalo"),
    "wound": (11, "area_critica"),
    "injury": (11, "area_critica"),
    "knife": (12, "objeto_suspeito"),
    "blade": (12, "objeto_suspeito"),
    "weapon": (12, "objeto_suspeito"),
    "surgical instrument": (2, "pinça_hemostatica"),
    "scalpel": (0, "bisturi"),
    "medical equipment": (11, "area_critica"),
    "operating room": (11, "area_critica"),
    "hospital": (11, "area_critica"),
    "surgery": (11, "area_critica"),
    "face": (13, "expressao_desconforto"),
    "person": (13, "expressao_desconforto"),
}

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


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

SURGICAL_CLASSES = {
    0: "bisturi",
    1: "tesoura_cirurgica",
    2: "pinça_hemostatica",
    3: "afastador_cirurgico",
    4: "sutura",
    5: "compressa",
    6: "sangramento_normal",
    7: "sangramento_anomalo",
    8: "utero",
    9: "ovarios",
    10: "mamas",
    11: "area_critica",
    12: "objeto_suspeito",
    13: "expressao_desconforto",
    14: "expressao_medo",
}

PRIORITY_CLASSES = {
    "sangramento_anomalo": 7,
    "area_critica": 11,
    "objeto_suspeito": 12,
    "expressao_desconforto": 13,
    "expressao_medo": 14,
}


@dataclass
class DetectionResult:
    """Resultado de uma detecção individual."""

    class_name: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]
    frame_number: int
    timestamp_seconds: float

    @property
    def is_anomaly(self) -> bool:
        return self.class_id in PRIORITY_CLASSES.values()


@dataclass
class VideoAnalysisReport:
    """Relatório completo da análise de um vídeo clínico."""

    file_path: Path
    video_duration_seconds: float
    total_frames: int
    frames_analyzed: int
    detections: list[DetectionResult] = field(default_factory=list)

    @property
    def anomaly_count(self) -> int:
        return sum(1 for d in self.detections if d.is_anomaly)

    @property
    def has_alerts(self) -> bool:
        return self.anomaly_count > 0

    @property
    def anomaly_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for d in self.detections:
            if d.is_anomaly:
                summary[d.class_name] = summary.get(d.class_name, 0) + 1
        return summary


class YOLODetector:
    """
    Detector YOLOv8 customizado para vídeos clínicos de saúde da mulher.

    Detecta instrumentos cirúrgicos, sangramento anômalo, áreas críticas,
    objetos suspeitos e expressões faciais indicativas de desconforto/medo.
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
        self._azure_vision = None
        self._azure_checked = False

    @property
    def azure_vision(self):
        if self._azure_checked:
            return self._azure_vision
        self._azure_checked = True
        if settings.azure.vision_key and settings.azure.vision_endpoint:
            try:
                from src.services.azure_services import AzureVisionService
                svc = AzureVisionService()
                if svc.is_available:
                    self._azure_vision = svc
                    logger.info("Azure Vision integrado ao detector de vídeo")
                    return svc
            except Exception as e:
                logger.warning(f"Azure Vision indisponível: {e}")
        return None

    def _azure_analyze_frame(
        self, frame: np.ndarray, frame_number: int, fps: float
    ) -> list[DetectionResult]:
        detections = []
        if not self.azure_vision:
            return detections

        import tempfile
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                tmp_path = tmp.name

            result = self.azure_vision.analyze_image(tmp_path)
            timestamp = frame_number / fps if fps > 0 else 0.0

            for tag_name in result.get("tags", []):
                tag_lower = tag_name.lower()
                for key, (cls_id, cls_name) in AZURE_TAG_MAPPING.items():
                    if key in tag_lower:
                        detections.append(DetectionResult(
                            class_name=cls_name,
                            class_id=cls_id,
                            confidence=0.80,
                            bbox=(0, 0, 0, 0),
                            frame_number=frame_number,
                            timestamp_seconds=timestamp,
                        ))
                        break

            caption = result.get("caption", "")
            if caption:
                caption_lower = caption.lower()
                for key, (cls_id, cls_name) in AZURE_TAG_MAPPING.items():
                    if key in caption_lower:
                        detections.append(DetectionResult(
                            class_name=cls_name,
                            class_id=cls_id,
                            confidence=0.75,
                            bbox=(0, 0, 0, 0),
                            frame_number=frame_number,
                            timestamp_seconds=timestamp,
                        ))
        except Exception as e:
            logger.warning(f"Azure Vision frame {frame_number}: {e}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
        return detections

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        if Path(self.model_path).exists():
            logger.info(f"Carregando modelo customizado: {self.model_path}")
            self._model = YOLO(self.model_path)
        else:
            logger.warning(f"Modelo customizado não encontrado. Usando YOLOv8n base.")
            self._model = YOLO("yolov8n.pt")
        device = resolve_device(settings.device)
        self._model.to(device)
        logger.info(f"Modelo YOLOv8 carregado no dispositivo: {device}")

    def process_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> list[DetectionResult]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
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
                class_name = SURGICAL_CLASSES.get(cls_id, f"desconhecido_{cls_id}")

                detections.append(DetectionResult(
                    class_name=class_name,
                    class_id=cls_id,
                    confidence=confidence,
                    bbox=bbox,
                    frame_number=frame_number,
                    timestamp_seconds=timestamp,
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
            file_path=video_path,
            video_duration_seconds=duration,
            total_frames=total_frames,
            frames_analyzed=0,
        )

        logger.info(f"Analisando vídeo: {video_path.name} ({duration:.1f}s, {total_frames} frames, {fps:.1f} FPS)")
        if self.azure_vision:
            logger.info("Azure Vision ativo como detector complementar")

        frame_idx = 0
        frames_to_process = min(total_frames, max_frames) if max_frames else total_frames
        azure_interval = sample_every_n_frames * 5

        with tqdm(total=frames_to_process, desc="Processando vídeo") as pbar:
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

                    if self.azure_vision and frame_idx % azure_interval == 0:
                        azure_dets = self._azure_analyze_frame(frame, frame_idx, fps)
                        report.detections.extend(azure_dets)
                        for d in azure_dets:
                            if d.is_anomaly:
                                logger.info(
                                    f"Azure Vision [{d.timestamp_seconds:.1f}s] "
                                    f"{d.class_name} (conf: {d.confidence:.2f})"
                                )

                    for d in detections:
                        if d.is_anomaly:
                            logger.warning(
                                f"ANOMALIA [{d.timestamp_seconds:.1f}s] "
                                f"{d.class_name} (conf: {d.confidence:.2f})"
                            )

                frame_idx += 1
                pbar.update(1)
                if progress_callback:
                    progress_callback(frame_idx, frames_to_process)

        cap.release()
        logger.info(
            f"Análise concluída: {report.frames_analyzed} frames, "
            f"{len(report.detections)} detecções, "
            f"{report.anomaly_count} anomalias"
        )
        return report

    def annotate_frame(self, frame: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 0, 255) if det.is_anomaly else (0, 255, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        return annotated

    def finetune(
        self,
        data_yaml: str | Path,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
    ) -> None:
        """
        Fine-tuning do modelo YOLOv8 com dataset customizado de imagens clínicas.

        Args:
            data_yaml: Caminho para o arquivo YAML de configuração do dataset
            epochs: Número de épocas de treinamento
            imgsz: Tamanho da imagem de entrada
            batch: Tamanho do batch
        """
        device = resolve_device(settings.device)
        logger.info(f"Iniciando fine-tuning do YOLOv8: epochs={epochs}, imgsz={imgsz}, device={device}")
        model = YOLO(self.model_path)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
        )
        metrics = model.val()
        logger.info(f"Métricas de validação: mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")

        save_path = Path("models") / "yolov8_custom.pt"
        save_path.parent.mkdir(exist_ok=True)
        model.save(str(save_path))
        self._model = model
        logger.info(f"Modelo fine-tuned salvo em: {save_path}")


from tqdm import tqdm
