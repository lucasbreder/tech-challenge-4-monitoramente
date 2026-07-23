"""
Agente de Análise de Vídeo para Saúde da Mulher.

Responsável por processar vídeos clínicos (cirurgias, consultas, fisioterapia,
triagem de violência) e gerar relatórios automatizados com alertas.

Critérios do edital atendidos:
- Análise de vídeos de partos, cirurgias ginecológicas, sessões de fisioterapia
- Identificação de padrões anômalos e sinais de desconforto
- Detecção de sangramento anômalo durante procedimentos
- Identificação de linguagem corporal indicativa de abuso
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from loguru import logger

from src.models.yolo_detector import DetectionResult, VideoAnalysisReport, YOLODetector


class VideoType(str, Enum):
    SURGERY = "cirurgia"
    CONSULTATION = "consulta"
    PHYSIOTHERAPY = "fisioterapia"
    VIOLENCE_SCREENING = "triagem_violencia"


@dataclass
class AnomalyAlert:
    """Alerta gerado a partir de anomalia detectada em vídeo."""

    video_type: VideoType
    anomaly_type: str
    confidence: float
    timestamp_seconds: float
    frame_number: int
    description: str
    severity: str = "baixa"

    def to_dict(self) -> dict:
        return {
            "tipo_video": self.video_type.value,
            "tipo_anomalia": self.anomaly_type,
            "confianca": self.confidence,
            "timestamp_segundos": self.timestamp_seconds,
            "frame": self.frame_number,
            "descricao": self.description,
            "severidade": self.severity,
        }


ANOMALY_DESCRIPTIONS = {
    "sangramento_anomalo": {
        "pt": "Sangramento anômalo detectado durante o procedimento. "
              "Requer avaliação imediata da equipe cirúrgica.",
        "severity": "alta",
    },
    "area_critica": {
        "pt": "Instrumento próximo a área cirúrgica crítica (útero/ovários/mamas). "
              "Verificar posicionamento.",
        "severity": "média",
    },
    "objeto_suspeito": {
        "pt": "Objeto suspeito detectado. Possível indicador de automutilação. "
              "Acionar protocolo de segurança.",
        "severity": "crítica",
    },
    "expressao_desconforto": {
        "pt": "Expressão facial de desconforto detectada na paciente. "
              "Avaliar bem-estar e nível de dor.",
        "severity": "média",
    },
    "expressao_medo": {
        "pt": "Expressão facial de medo detectada na paciente. "
              "Possível sinal de violência doméstica ou abuso. Iniciar protocolo de acolhimento.",
        "severity": "alta",
    },
}


class VideoAnalysisAgent:
    """
    Agente autônomo de análise de vídeo clínico.

    Processa vídeos, detecta anomalias e gera alertas contextualizados
    para a equipe médica especializada em saúde da mulher.
    """

    def __init__(self, detector: YOLODetector | None = None):
        self.detector = detector or YOLODetector()

    def analyze(
        self,
        video_path: str | Path,
        video_type: VideoType = VideoType.CONSULTATION,
        sample_rate: int = 10,
    ) -> tuple[VideoAnalysisReport, list[AnomalyAlert]]:
        video_path = Path(video_path)
        logger.info(f"[VideoAgent] Analisando vídeo tipo '{video_type.value}': {video_path.name}")

        report = self.detector.analyze_video(
            video_path=video_path,
            sample_every_n_frames=sample_rate,
        )

        alerts = self._generate_alerts(report, video_type)

        if alerts:
            logger.warning(
                f"[VideoAgent] {len(alerts)} alertas gerados para {video_path.name}"
            )
        else:
            logger.info(f"[VideoAgent] Nenhuma anomalia crítica detectada em {video_path.name}")

        return report, alerts

    def _generate_alerts(
        self, report: VideoAnalysisReport, video_type: VideoType
    ) -> list[AnomalyAlert]:
        alerts: list[AnomalyAlert] = []

        for det in report.detections:
            if not det.is_anomaly:
                continue

            anomaly_info = ANOMALY_DESCRIPTIONS.get(det.class_name, {})

            alerts.append(AnomalyAlert(
                video_type=video_type,
                anomaly_type=det.class_name,
                confidence=det.confidence,
                timestamp_seconds=det.timestamp_seconds,
                frame_number=det.frame_number,
                description=anomaly_info.get("pt", f"Anomalia detectada: {det.class_name}"),
                severity=anomaly_info.get("severity", "baixa"),
            ))

        alerts.sort(key=lambda a: a.timestamp_seconds)
        return alerts

    def analyze_surgery(self, video_path: str | Path) -> tuple[VideoAnalysisReport, list[AnomalyAlert]]:
        return self.analyze(video_path, VideoType.SURGERY, sample_rate=5)

    def analyze_consultation(self, video_path: str | Path) -> tuple[VideoAnalysisReport, list[AnomalyAlert]]:
        return self.analyze(video_path, VideoType.CONSULTATION, sample_rate=10)

    def analyze_physiotherapy(self, video_path: str | Path) -> tuple[VideoAnalysisReport, list[AnomalyAlert]]:
        return self.analyze(video_path, VideoType.PHYSIOTHERAPY, sample_rate=10)

    def analyze_violence_screening(self, video_path: str | Path) -> tuple[VideoAnalysisReport, list[AnomalyAlert]]:
        return self.analyze(video_path, VideoType.VIOLENCE_SCREENING, sample_rate=5)
