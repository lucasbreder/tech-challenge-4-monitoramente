"""
Pipeline de processamento de vídeo para análise de saúde da mulher.

Orquestra o fluxo completo de análise de vídeo clínico:
carregamento → detecção → geração de relatório → exportação.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.agents.video_agent import (
    AnomalyAlert,
    VideoAnalysisAgent,
    VideoAnalysisReport,
    VideoType,
)
from src.config import VIDEOS_DIR
from src.utils.report_generator import ReportGenerator


class VideoPipeline:
    """
    Pipeline de análise de vídeo clínico.

    Processa vídeos de cirurgias, consultas, fisioterapia e triagem
    de violência, gerando relatórios e alertas.
    """

    def __init__(
        self,
        agent: VideoAnalysisAgent | None = None,
        report_generator: ReportGenerator | None = None,
    ):
        self.agent = agent or VideoAnalysisAgent()
        self.report_gen = report_generator or ReportGenerator()

    def run(
        self,
        video_path: str | Path,
        video_type: VideoType = VideoType.CONSULTATION,
        sample_rate: int = 10,
        export_report: bool = True,
        export_annotated: bool = False,
    ) -> tuple[VideoAnalysisReport, list[AnomalyAlert]]:
        video_path = Path(video_path)
        logger.info(f"=== Pipeline de Vídeo: {video_path.name} ({video_type.value}) ===")

        report, alerts = self.agent.analyze(
            video_path=video_path,
            video_type=video_type,
            sample_rate=sample_rate,
        )

        if export_report:
            self.report_gen.export_video_report(report, alerts, video_type)

        logger.info(
            f"Pipeline de vídeo concluído: {len(report.detections)} detecções, "
            f"{len(alerts)} alertas"
        )
        return report, alerts

    def run_batch(
        self,
        video_dir: str | Path | None = None,
        video_type: VideoType = VideoType.CONSULTATION,
    ) -> list[tuple[VideoAnalysisReport, list[AnomalyAlert]]]:
        video_dir = Path(video_dir) if video_dir else VIDEOS_DIR

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        videos = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]

        if not videos:
            logger.warning(f"Nenhum vídeo encontrado em: {video_dir}")
            return []

        results = []
        for video in videos:
            try:
                result = self.run(video, video_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Erro ao processar {video}: {e}")
                continue

        logger.info(f"Batch concluído: {len(results)}/{len(videos)} vídeos processados")
        return results


def run_video_pipeline(
    video_path: str,
    video_type: str = "consulta",
    sample_rate: int = 10,
    export_report: bool = True,
) -> tuple[VideoAnalysisReport, list[AnomalyAlert]]:
    pipeline = VideoPipeline()
    vt = VideoType(video_type)
    return pipeline.run(video_path, vt, sample_rate, export_report)
