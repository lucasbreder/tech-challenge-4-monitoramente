"""
Agente de Fusão Multimodal.

Responsável por integrar e correlacionar os dados das três
modalidades (vídeo, áudio e sinais vitais) para gerar uma
visão unificada do estado da paciente e alertas combinados.

Critérios do edital atendidos:
- Fusão de diferentes tipos de dados médicos (texto, áudio, vídeo)
- Detecção precoce de riscos em saúde materna e ginecológica
- Monitoramento de bem-estar psicológico feminino
- Aplicação de detecção de anomalias em tempo real para monitoramento preventivo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from src.agents.anomaly_agent import AnomalyDetectionResult
from src.agents.audio_agent import AudioAlert
from src.agents.video_agent import AnomalyAlert as VideoAnomalyAlert


@dataclass
class FusionRiskAssessment:
    """Avaliação de risco unificada combinando todas as modalidades."""

    patient_id: str
    timestamp: str
    overall_risk_score: float
    overall_risk_level: str

    video_risk_score: float
    audio_risk_score: float
    vitals_risk_score: float

    video_alerts: list[dict] = field(default_factory=list)
    audio_alerts: list[dict] = field(default_factory=list)
    vitals_alerts: list[dict] = field(default_factory=list)

    correlated_risks: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "paciente_id": self.patient_id,
            "timestamp": self.timestamp,
            "score_risco_geral": self.overall_risk_score,
            "nivel_risco_geral": self.overall_risk_level,
            "score_video": self.video_risk_score,
            "score_audio": self.audio_risk_score,
            "score_sinais_vitais": self.vitals_risk_score,
            "alertas_video": self.video_alerts,
            "alertas_audio": self.audio_alerts,
            "alertas_sinais_vitais": self.vitals_alerts,
            "riscos_correlacionados": self.correlated_risks,
            "recomendacoes": self.recommendations,
        }


class MultimodalFusionAgent:
    """
    Agente de fusão multimodal para saúde da mulher.

    Integra resultados dos agentes de vídeo, áudio e sinais vitais,
    calculando um score de risco unificado e gerando recomendações
    contextualizadas para a equipe médica.
    """

    def __init__(self):
        self.video_weight = 0.35
        self.audio_weight = 0.35
        self.vitals_weight = 0.30

    def fuse(
        self,
        patient_id: str,
        video_alerts: list[VideoAnomalyAlert] | None = None,
        audio_alerts: list[AudioAlert] | None = None,
        vitals_results: list[AnomalyDetectionResult] | None = None,
    ) -> FusionRiskAssessment:
        video_alerts = video_alerts or []
        audio_alerts = audio_alerts or []
        vitals_results = vitals_results or []

        video_score = self._compute_video_risk(video_alerts)
        audio_score = self._compute_audio_risk(audio_alerts)
        vitals_score = self._compute_vitals_risk(vitals_results)

        overall_score = (
            video_score * self.video_weight
            + audio_score * self.audio_weight
            + vitals_score * self.vitals_weight
        )

        overall_level = self._classify_risk(overall_score)

        correlated = self._correlate_risks(video_alerts, audio_alerts, vitals_results)
        recommendations = self._generate_recommendations(
            overall_level, video_alerts, audio_alerts, vitals_results, correlated
        )

        return FusionRiskAssessment(
            patient_id=patient_id,
            timestamp=datetime.now().isoformat(),
            overall_risk_score=overall_score,
            overall_risk_level=overall_level,
            video_risk_score=video_score,
            audio_risk_score=audio_score,
            vitals_risk_score=vitals_score,
            video_alerts=[a.to_dict() for a in video_alerts],
            audio_alerts=[a.to_dict() for a in audio_alerts],
            vitals_alerts=[
                {
                    "tipo": r.record.signal_type.value,
                    "valor": r.record.value,
                    "unidade": r.record.unit,
                    "score": r.anomaly_score,
                    "severidade": r.severity,
                }
                for r in vitals_results if r.is_anomaly
            ],
            correlated_risks=correlated,
            recommendations=recommendations,
        )

    def _compute_video_risk(self, alerts: list[VideoAnomalyAlert]) -> float:
        if not alerts:
            return 0.0

        severity_weights = {"crítica": 1.0, "alta": 0.8, "média": 0.5, "baixa": 0.2}
        weighted = sum(severity_weights.get(a.severity, 0.2) * a.confidence for a in alerts)
        return min(1.0, weighted / max(len(alerts), 1))

    def _compute_audio_risk(self, alerts: list[AudioAlert]) -> float:
        if not alerts:
            return 0.0

        return max(a.risk_score for a in alerts) if alerts else 0.0

    def _compute_vitals_risk(self, results: list[AnomalyDetectionResult]) -> float:
        if not results:
            return 0.0

        anomalies = [r for r in results if r.is_anomaly]
        if not anomalies:
            return 0.0

        return sum(a.anomaly_score for a in anomalies) / max(len(anomalies), 1)

    def _classify_risk(self, score: float) -> str:
        if score > 0.8:
            return "crítico"
        elif score > 0.6:
            return "alto"
        elif score > 0.3:
            return "médio"
        return "baixo"

    def _correlate_risks(
        self,
        video_alerts: list[VideoAnomalyAlert],
        audio_alerts: list[AudioAlert],
        vitals_results: list[AnomalyDetectionResult],
    ) -> list[str]:
        correlated = []

        has_violence_video = any(
            a.anomaly_type in ("objeto_suspeito", "expressao_medo")
            for a in video_alerts
        )
        has_violence_audio = any(
            a.alert_type == "violencia_domestica" for a in audio_alerts
        )

        if has_violence_video and has_violence_audio:
            correlated.append(
                "CORRELAÇÃO FORTE: Indicadores visuais E vocais de violência doméstica. "
                "Probabilidade muito elevada. Protocolo de urgência recomendado."
            )
        elif has_violence_video:
            correlated.append(
                "Indicadores visuais de possível violência doméstica. "
                "Sem confirmação por áudio."
            )
        elif has_violence_audio:
            correlated.append(
                "Indicadores vocais de possível violência doméstica. "
                "Sem confirmação visual."
            )

        has_distress_video = any(
            a.anomaly_type == "expressao_desconforto" for a in video_alerts
        )
        has_distress_audio = any(
            a.alert_type in ("depressão_pós_parto", "sofrimento_psicologico")
            for a in audio_alerts
        )
        vital_anomalies = [r for r in vitals_results if r.is_anomaly]

        if has_distress_video and has_distress_audio and vital_anomalies:
            correlated.append(
                "CORRELAÇÃO MULTIMODAL: Sinais de sofrimento detectados em vídeo, áudio "
                "e sinais vitais. Avaliação integrada urgente recomendada."
            )

        return correlated

    def _generate_recommendations(
        self,
        risk_level: str,
        video_alerts: list[VideoAnomalyAlert],
        audio_alerts: list[AudioAlert],
        vitals_results: list[AnomalyDetectionResult],
        correlated: list[str],
    ) -> list[str]:
        recommendations = []

        if risk_level == "crítico":
            recommendations.append(
                "AÇÃO IMEDIATA: Notificar equipe médica plantonista e serviço social. "
                "Ativar protocolo de emergência para saúde da mulher."
            )

        if risk_level in ("crítico", "alto"):
            recommendations.append(
                "Avaliação psicológica e psiquiátrica prioritária nas próximas 24 horas."
            )

            if any(a.alert_type == "violencia_domestica" for a in audio_alerts):
                recommendations.append(
                    "Acionar serviço de proteção à mulher e Delegacia da Mulher "
                    "com consentimento da paciente."
                )

        if risk_level == "médio":
            recommendations.append(
                "Avaliação multidisciplinar (ginecologia, psicologia, serviço social) "
                "agendada para esta semana."
            )
            recommendations.append("Monitoramento intensificado de sinais vitais.")

        if not recommendations:
            recommendations.append("Manter acompanhamento de rotina conforme protocolo.")
            recommendations.append("Reavaliar na próxima consulta.")

        return recommendations
