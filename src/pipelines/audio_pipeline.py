"""
Pipeline de processamento de áudio para análise de consultas de saúde da mulher.

Orquestra o fluxo completo de análise de áudio de consulta:
carregamento → transcrição → análise emocional → geração de alerta.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.agents.audio_agent import (
    AudioAlert,
    AudioAnalysisAgent,
    AudioAnalysisReport,
    AudioConsultationType,
)
from src.config import AUDIOS_DIR
from src.utils.report_generator import ReportGenerator


class AudioPipeline:
    """
    Pipeline de análise de áudio de consultas.

    Processa gravações de consultas ginecológicas, pré-natal, pós-parto
    e atendimento a vítimas de violência, gerando relatórios e alertas.
    """

    def __init__(
        self,
        agent: AudioAnalysisAgent | None = None,
        report_generator: ReportGenerator | None = None,
    ):
        self.agent = agent or AudioAnalysisAgent()
        self.report_gen = report_generator or ReportGenerator()

    def run(
        self,
        audio_path: str | Path,
        consultation_type: AudioConsultationType = AudioConsultationType.GYNECOLOGICAL,
        export_report: bool = True,
    ) -> tuple[AudioAnalysisReport, list[AudioAlert]]:
        audio_path = Path(audio_path)
        #logger.info(
        #    f"=== Pipeline de Áudio: {audio_path.name} ({consultation_type.value}) ==="
        #)

        report, alerts = self.agent.analyze(
            audio_path=audio_path,
            consultation_type=consultation_type,
        )

        if export_report:
            self.report_gen.export_audio_report(report, alerts, consultation_type)

        logger.info(
            f"Pipeline de áudio concluído: transcrição={len(report.transcription)} chars, "
            f"{len(alerts)} alertas, risco={report.risk_level}"
        )
        return report, alerts

    def run_batch(
        self,
        audio_dir: str | Path | None = None,
        consultation_type: AudioConsultationType = AudioConsultationType.GYNECOLOGICAL,
    ) -> list[tuple[AudioAnalysisReport, list[AudioAlert]]]:
        audio_dir = Path(audio_dir) if audio_dir else AUDIOS_DIR

        audio_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}
        audios = [f for f in audio_dir.iterdir() if f.suffix.lower() in audio_extensions]

        if not audios:
            logger.warning(f"Nenhum áudio encontrado em: {audio_dir}")
            return []

        results = []
        for audio in audios:
            try:
                result = self.run(audio, consultation_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Erro ao processar {audio}: {e}")
                continue

        logger.info(f"Batch concluído: {len(results)}/{len(audios)} áudios processados")
        return results


def run_audio_pipeline(
    audio_path: str,
    consultation_type: str = "ginecologica",
    export_report: bool = True,
) -> tuple[AudioAnalysisReport, list[AudioAlert]]:
    pipeline = AudioPipeline()
    ct = AudioConsultationType(consultation_type)
    return pipeline.run(audio_path, ct, export_report)
