"""
Pipeline Multimodal para saúde da mulher.

Orquestra o processamento integrado de vídeo, áudio e sinais vitais,
gerando uma avaliação de risco unificada através do agente de fusão.

Critérios do edital atendidos:
- Realizar análise e fusão de diferentes tipos de dados (texto, áudio, vídeo)
- Detectar precocemente riscos em saúde materna e ginecológica
- Identificar sinais de violência doméstica ou abuso
- Monitorar bem-estar psicológico feminino
- Aplicar técnicas de detecção de anomalias em tempo real
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.agents.anomaly_agent import AnomalyDetectionAgent, SignalType, VitalSignRecord
from src.agents.audio_agent import AudioConsultationType
from src.agents.fusion_agent import FusionRiskAssessment, MultimodalFusionAgent
from src.config import AUDIOS_DIR, VIDEOS_DIR
from src.pipelines.audio_pipeline import AudioPipeline
from src.pipelines.video_pipeline import VideoPipeline
from src.services.alert_service import AlertService
from src.utils.report_generator import ReportGenerator


class MultimodalPipeline:
    """
    Pipeline multimodal completo para monitoramento de saúde da mulher.

    Integra análise de vídeo, áudio e sinais vitais, gera score de risco
    unificado e dispara alertas para a equipe médica quando necessário.
    """

    def __init__(
        self,
        video_pipeline: VideoPipeline | None = None,
        audio_pipeline: AudioPipeline | None = None,
        anomaly_agent: AnomalyDetectionAgent | None = None,
        fusion_agent: MultimodalFusionAgent | None = None,
        alert_service: AlertService | None = None,
        report_generator: ReportGenerator | None = None,
    ):
        self.video_pipeline = video_pipeline or VideoPipeline()
        self.audio_pipeline = audio_pipeline or AudioPipeline()
        self.anomaly_agent = anomaly_agent or AnomalyDetectionAgent(
            use_gestational_ranges=False,
        )
        self.fusion_agent = fusion_agent or MultimodalFusionAgent()
        self.alert_service = alert_service or AlertService()
        self.report_gen = report_generator or ReportGenerator()

    def run(
        self,
        patient_id: str,
        video_path: str | Path | None = None,
        audio_path: str | Path | None = None,
        vital_signs: list[dict] | None = None,
        video_type: str = "consulta",
        consultation_type: str = "ginecologica",
        export_report: bool = True,
        progress_callback: callable = None,
        audio_progress_callback: callable = None,
        is_gestational: bool = False,
    ) -> FusionRiskAssessment:
        logger.info(f"=== Pipeline Multimodal: Paciente {patient_id} ===")

        self._log_azure_services()

        video_alerts = []
        audio_alerts = []
        vitals_results = []
        audio_report_score = 0.0

        if video_path:
            from src.agents.video_agent import VideoType

            logger.info(f"Processando vídeo: {video_path}")
            _, video_alerts = self.video_pipeline.run(
                video_path=video_path,
                video_type=VideoType(video_type),
                export_report=export_report,
                progress_callback=progress_callback,
            )

        if audio_path:
            logger.info(f"Processando áudio: {audio_path}")
            audio_report, audio_alerts = self.audio_pipeline.run(
                audio_path=audio_path,
                consultation_type=AudioConsultationType(consultation_type),
                export_report=export_report,
                progress_callback=audio_progress_callback,
            )
            audio_report_score = audio_report.risk_score

        if vital_signs:
            from src.agents.anomaly_agent import GESTATIONAL_REFERENCE_RANGES, DEFAULT_REFERENCE_RANGES

            self.anomaly_agent.reference_ranges = (
                GESTATIONAL_REFERENCE_RANGES if is_gestational else DEFAULT_REFERENCE_RANGES
            )
            logger.info(f"Processando {len(vital_signs)} sinais vitais"
                        f" {'(faixas gestacionais)' if is_gestational else ''}")
            records = []
            for vs in vital_signs:
                record = VitalSignRecord(
                    timestamp=vs.get("timestamp", 0.0),
                    signal_type=SignalType(vs["signal_type"]),
                    value=vs["value"],
                    unit=vs.get("unit", ""),
                    patient_id=patient_id,
                )
                result = self.anomaly_agent.add_record(record)
                if result:
                    vitals_results.append(result)

        assessment = self.fusion_agent.fuse(
            patient_id=patient_id,
            video_alerts=video_alerts,
            audio_alerts=audio_alerts,
            vitals_results=vitals_results,
            audio_report_score=audio_report_score,
        )

        if assessment.overall_risk_level in ("crítico", "alto"):
            self.alert_service.send_alert(assessment)
        else:
            self.alert_service.send_notification(assessment)

        if export_report:
            report_path = self.report_gen.export_fusion_report(assessment)
            self._upload_to_azure(patient_id, video_path, audio_path, report_path)

        logger.info(
            f"Pipeline multimodal concluído: risco={assessment.overall_risk_level} "
            f"(score: {assessment.overall_risk_score:.2f})"
        )
        return assessment

    def _upload_to_azure(
        self, patient_id: str, video_path: str | Path | None,
        audio_path: str | Path | None, report_path: Path,
    ) -> None:
        from src.config import settings

        if not settings.azure.storage_connection_string:
            return

        try:
            from src.services.azure_services import AzureStorageService
            storage = AzureStorageService()
            if not storage.is_available:
                return

            from datetime import datetime
            prefix = f"{patient_id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            storage.upload_file(report_path, container_name="saude-mulher",
                                blob_name=f"{prefix}/{report_path.name}")

            if video_path and Path(video_path).exists():
                vp = Path(video_path)
                storage.upload_file(vp, container_name="saude-mulher",
                                    blob_name=f"{prefix}/video{vp.suffix}")
                logger.info("Vídeo enviado para Azure Storage")

            if audio_path and Path(audio_path).exists():
                ap = Path(audio_path)
                storage.upload_file(ap, container_name="saude-mulher",
                                    blob_name=f"{prefix}/audio{ap.suffix}")
                logger.info("Áudio enviado para Azure Storage")

        except Exception as e:
            logger.warning(f"Não foi possível enviar mídia para Azure Storage: {e}")

    def _log_azure_services(self) -> None:
        from src.config import settings

        services = []
        if settings.azure.speech_key:
            services.append("Speech (transcrição)")
        if settings.azure.vision_key and settings.azure.vision_endpoint:
            services.append("Vision (imagens)")
        if settings.azure.storage_connection_string:
            services.append("Storage (relatórios)")
        if settings.azure.openai_api_key and settings.azure.openai_endpoint:
            services.append("OpenAI (análise de texto)")

        if services:
            logger.info(f"Azure ativo: {', '.join(services)}")
        else:
            logger.info("Azure não configurado — usando apenas modelos locais")


def run_multimodal_pipeline(
    patient_id: str,
    video_path: str | None = None,
    audio_path: str | None = None,
    vital_signs: list[dict] | None = None,
) -> FusionRiskAssessment:
    pipeline = MultimodalPipeline()
    return pipeline.run(
        patient_id=patient_id,
        video_path=video_path,
        audio_path=audio_path,
        vital_signs=vital_signs,
    )
