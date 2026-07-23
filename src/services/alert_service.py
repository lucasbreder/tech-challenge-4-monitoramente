"""
Serviço de alertas para a equipe médica.

Critérios do edital atendidos:
- Alertar a equipe especializada em tempo real
- Fluxo final do alerta à equipe médica
- Notificação de anomalias detectadas
"""

from __future__ import annotations

import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from loguru import logger

from src.config import REPORTS_DIR, settings


class AlertService:
    """
    Serviço de despacho de alertas para equipe médica.

    Suporta múltiplos canais: email, log estruturado e arquivo JSON
    para integração com sistemas hospitalares (HL7/FHIR).
    """

    def __init__(self):
        self.smtp_host = settings.alert.email_smtp_host
        self.smtp_port = settings.alert.email_smtp_port
        self.email_from = settings.alert.email_from
        self.email_to = settings.alert.email_to
        self.email_password = settings.alert.email_password

    def send_alert(self, assessment) -> bool:
        logger.critical(
            f"ALERTA CRÍTICO: Paciente {assessment.patient_id} - "
            f"Risco {assessment.overall_risk_level} (score: {assessment.overall_risk_score:.2f})"
        )
        self._save_alert_log(assessment, level="critical")

        for rec in assessment.recommendations:
            logger.critical(f"  -> {rec}")

        if self.email_password:
            self._send_email(assessment, is_alert=True)

        return True

    def send_notification(self, assessment) -> bool:
        logger.info(
            f"Notificação: Paciente {assessment.patient_id} - "
            f"Risco {assessment.overall_risk_level} (score: {assessment.overall_risk_score:.2f})"
        )
        self._save_alert_log(assessment, level="info")
        return True

    def _save_alert_log(self, assessment, level: str = "info") -> Path:
        log_dir = REPORTS_DIR / "alerts"
        log_dir.mkdir(parents=True, exist_ok=True)

        filename = f"alert_{assessment.patient_id}_{assessment.timestamp.replace(':', '-')}.json"
        filepath = log_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(assessment.to_dict(), f, ensure_ascii=False, indent=2)

        logger.debug(f"Log de alerta salvo: {filepath}")
        return filepath

    def _send_email(self, assessment, is_alert: bool = False) -> bool:
        try:
            msg = MIMEMultipart()
            subject_prefix = "URGENTE" if is_alert else "Notificação"
            msg["Subject"] = (
                f"[{subject_prefix}] Saúde da Mulher - Paciente {assessment.patient_id} "
                f"- Risco {assessment.overall_risk_level.upper()}"
            )
            msg["From"] = self.email_from
            msg["To"] = self.email_to

            body = self._build_email_body(assessment)
            msg.attach(MIMEText(body, "html", "utf-8"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.email_from, self.email_password)
                server.send_message(msg)

            logger.info(f"Email enviado para {self.email_to}")
            return True
        except Exception as e:
            logger.error(f"Falha ao enviar email: {e}")
            return False

    def _build_email_body(self, assessment) -> str:
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: {'#d32f2f' if assessment.overall_risk_level in ('crítico', 'alto') else '#1976d2'};">
                Relatório de Monitoramento Multimodal
            </h2>
            <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <td><strong>Paciente</strong></td>
                    <td>{assessment.patient_id}</td>
                </tr>
                <tr>
                    <td><strong>Data/Hora</strong></td>
                    <td>{assessment.timestamp}</td>
                </tr>
                <tr>
                    <td><strong>Score de Risco</strong></td>
                    <td style="color: {'#d32f2f' if assessment.overall_risk_score > 0.6 else '#388e3c'};">
                        {assessment.overall_risk_score:.2f} ({assessment.overall_risk_level.upper()})
                    </td>
                </tr>
            </table>

            <h3>Scores por Modalidade</h3>
            <ul>
                <li><strong>Vídeo:</strong> {assessment.video_risk_score:.2f}</li>
                <li><strong>Áudio:</strong> {assessment.audio_risk_score:.2f}</li>
                <li><strong>Sinais Vitais:</strong> {assessment.vitals_risk_score:.2f}</li>
            </ul>

            <h3>Riscos Correlacionados</h3>
            <ul>
                {"".join(f"<li>{r}</li>" for r in assessment.correlated_risks) or "<li>Nenhum</li>"}
            </ul>

            <h3>Recomendações</h3>
            <ul>
                {"".join(f"<li>{r}</li>" for r in assessment.recommendations)}
            </ul>

            <hr>
            <p style="font-size: 0.8em; color: #666;">
                Este é um alerta automatizado do Sistema de Monitoramento Multimodal
                para Saúde da Mulher. Em caso de emergência, siga o protocolo hospitalar.
            </p>
        </body>
        </html>
        """
