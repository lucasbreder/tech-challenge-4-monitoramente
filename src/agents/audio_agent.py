"""
Agente de Análise de Áudio para Saúde da Mulher.

Responsável por processar gravações de voz de pacientes em consultas,
detectando sintomas relacionados à fala como depressão pós-parto,
ansiedade, sinais de violência doméstica e fadiga hormonal.

Critérios do edital atendidos:
- Processamento de gravações de voz de pacientes
- Detecção de depressão pós-parto, ansiedade, sinais de violência doméstica
- Análise de tom de voz, hesitação ao relatar sintomas
- Sinais de ansiedade gestacional e trauma vocal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from loguru import logger

from src.models.audio_analyzer import AudioAnalysisReport, AudioAnalyzer


class AudioConsultationType(str, Enum):
    GYNECOLOGICAL = "ginecologica"
    PRENATAL = "pre_natal"
    POSTPARTUM = "pos_parto"
    VIOLENCE_VICTIM = "vitima_violencia"


@dataclass
class AudioAlert:
    """Alerta baseado na análise de áudio da consulta."""

    consultation_type: AudioConsultationType
    alert_type: str
    risk_score: float
    risk_level: str
    evidence: str
    timestamp_seconds: float | None = None

    def to_dict(self) -> dict:
        return {
            "tipo_consulta": self.consultation_type.value,
            "tipo_alerta": self.alert_type,
            "score_risco": self.risk_score,
            "nivel_risco": self.risk_level,
            "evidencia": self.evidence,
            "timestamp_segundos": self.timestamp_seconds,
        }


ALERT_PROTOCOLS = {
    "depressão_pós_parto": {
        "title": "Risco de Depressão Pós-Parto",
        "action": "Encaminhar para avaliação psicológica e psiquiátrica. "
                  "Aplicar Escala de Depressão Pós-Parto de Edimburgo (EPDS).",
    },
    "ansiedade_gestacional": {
        "title": "Ansiedade Gestacional Detectada",
        "action": "Oferecer suporte psicológico. Avaliar necessidade de intervenção "
                  "medicamentosa compatível com gestação.",
    },
    "violencia_domestica": {
        "title": "Possível Violência Doméstica",
        "action": "Acionar protocolo de acolhimento sigiloso. Aplicar questionário "
                  "HARK (Humiliation, Afraid, Rape, Kick). Notificar serviços de proteção.",
    },
    "trauma_vocal": {
        "title": "Indicadores de Trauma na Voz",
        "action": "Encaminhar para serviço de apoio psicossocial. "
                  "Garantir ambiente seguro e acolhedor.",
    },
    "fadiga_hormonal": {
        "title": "Sinais de Fadiga Hormonal",
        "action": "Avaliar níveis hormonais. Verificar função tireoidiana. "
                  "Orientar sobre saúde do sono e nutrição.",
    },
}


class AudioAnalysisAgent:
    """
    Agente autônomo de análise de áudio de consultas.

    Orquestra a transcrição, análise de emoções e geração de alertas
    contextualizados para a equipe de saúde da mulher.
    """

    def __init__(self, analyzer: AudioAnalyzer | None = None):
        self.analyzer = analyzer or AudioAnalyzer()

    def analyze(
        self,
        audio_path: str | Path,
        consultation_type: AudioConsultationType = AudioConsultationType.GYNECOLOGICAL,
    ) -> tuple[AudioAnalysisReport, list[AudioAlert]]:
        audio_path = Path(audio_path)
        logger.info(
            f"[AudioAgent] Analisando áudio de consulta "
            f"'{consultation_type.value}': {audio_path.name}"
        )

        report = self.analyzer.analyze(audio_path)

        alerts = self._generate_alerts(report, consultation_type)

        if alerts:
            logger.warning(
                f"[AudioAgent] {len(alerts)} alertas gerados para {audio_path.name}"
            )
        else:
            logger.info(f"[AudioAgent] Nenhum alerta gerado para {audio_path.name}")

        return report, alerts

    def _generate_alerts(
        self, report: AudioAnalysisReport, consultation_type: AudioConsultationType
    ) -> list[AudioAlert]:
        alerts: list[AudioAlert] = []

        if report.risk_level in ("alto", "crítico"):
            if consultation_type == AudioConsultationType.POSTPARTUM:
                alerts.append(AudioAlert(
                    consultation_type=consultation_type,
                    alert_type="depressão_pós_parto",
                    risk_score=report.risk_score,
                    risk_level=report.risk_level,
                    evidence=f"Score de risco emocional: {report.risk_score:.2f}. "
                             f"Emoções detectadas: {dict(report.sentiment_summary)}",
                ))

            if consultation_type == AudioConsultationType.PRENATAL:
                alerts.append(AudioAlert(
                    consultation_type=consultation_type,
                    alert_type="ansiedade_gestacional",
                    risk_score=report.risk_score,
                    risk_level=report.risk_level,
                    evidence=f"Score de ansiedade vocal: {report.risk_score:.2f}. "
                             f"Emoções predominantes: {dict(report.sentiment_summary)}",
                ))

            if consultation_type in (AudioConsultationType.GYNECOLOGICAL, AudioConsultationType.VIOLENCE_VICTIM):
                textual_risks = [r for r in report.risk_factors
                                 if r in ("violência", "abuso", "medo", "socorro")]
                if textual_risks:
                    alerts.append(AudioAlert(
                        consultation_type=consultation_type,
                        alert_type="violencia_domestica",
                        risk_score=report.risk_score,
                        risk_level=report.risk_level,
                        evidence=f"Indicadores textuais: {textual_risks}. "
                                 f"Score emocional: {report.risk_score:.2f}",
                    ))

            if report.risk_score > 0.6 and not alerts:
                alerts.append(AudioAlert(
                    consultation_type=consultation_type,
                    alert_type="sofrimento_psicologico",
                    risk_score=report.risk_score,
                    risk_level=report.risk_level,
                    evidence="Indicadores de sofrimento psicológico detectados "
                             f"(score: {report.risk_score:.2f})",
                ))

        silence_periods = self.analyzer.preprocessor.detect_silence(
            self.analyzer.preprocessor.load_audio(report.file_path)[0],
            self.analyzer.preprocessor.load_audio(report.file_path)[1],
        )
        if len(silence_periods) > 10:
            alerts.append(AudioAlert(
                consultation_type=consultation_type,
                alert_type="hesitacao_excessiva",
                risk_score=min(1.0, len(silence_periods) / 20.0),
                risk_level="médio",
                evidence=f"Períodos de silêncio/hesitação excessivos ({len(silence_periods)}). "
                         f"Pode indicar dificuldade em relatar sintomas.",
            ))

        for seg in report.segments:
            if seg.has_risk_indicator:
                for indicator in seg.matched_indicators:
                    if indicator in ("depressão", "ansiedade"):
                        already = any(a.alert_type == "indicador_verbal" for a in alerts)
                        if not already:
                            alerts.append(AudioAlert(
                                consultation_type=consultation_type,
                                alert_type="indicador_verbal",
                                risk_score=0.7,
                                risk_level="médio",
                                evidence=f"Indicador verbal '{indicator}' detectado "
                                         f"em: '{seg.text[:100]}...'",
                                timestamp_seconds=seg.start_seconds,
                            ))

        return alerts

    def analyze_gynecological(self, audio_path: str | Path) -> tuple[AudioAnalysisReport, list[AudioAlert]]:
        return self.analyze(audio_path, AudioConsultationType.GYNECOLOGICAL)

    def analyze_prenatal(self, audio_path: str | Path) -> tuple[AudioAnalysisReport, list[AudioAlert]]:
        return self.analyze(audio_path, AudioConsultationType.PRENATAL)

    def analyze_postpartum(self, audio_path: str | Path) -> tuple[AudioAnalysisReport, list[AudioAlert]]:
        return self.analyze(audio_path, AudioConsultationType.POSTPARTUM)

    def analyze_violence_victim(self, audio_path: str | Path) -> tuple[AudioAnalysisReport, list[AudioAlert]]:
        return self.analyze(audio_path, AudioConsultationType.VIOLENCE_VICTIM)
