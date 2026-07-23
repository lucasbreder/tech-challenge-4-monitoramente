"""
Gerador de relatórios automatizados para o sistema multimodal.

Critérios do edital atendidos:
- Relatórios automáticos especializados para:
  - Desvios em procedimentos obstétricos
  - Sinais de complicações em cirurgias ginecológicas
  - Indicadores visuais de desconforto psicológico
  - Alertas para possíveis casos de violência doméstica
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.config import REPORTS_DIR


class ReportGenerator:
    """
    Gerador de relatórios clínicos automatizados.

    Produz relatórios em múltiplos formatos (JSON, Markdown)
    para documentação clínica e auditoria.
    """

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir else REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_video_report(self, report, alerts, video_type) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_report_{video_type.value}_{timestamp}"

        data = {
            "tipo": "relatorio_video",
            "tipo_video": video_type.value,
            "arquivo": str(report.file_path),
            "duracao_segundos": report.video_duration_seconds,
            "total_frames": report.total_frames,
            "frames_analisados": report.frames_analyzed,
            "total_deteccoes": len(report.detections),
            "total_anomalias": report.anomaly_count,
            "resumo_anomalias": report.anomaly_summary,
            "alertas": [a.to_dict() for a in alerts],
            "gerado_em": timestamp,
        }

        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        md_content = self._build_video_markdown_report(data)
        md_path = self.output_dir / f"{filename}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Relatório de vídeo exportado: {json_path}")
        return json_path

    def export_audio_report(self, report, alerts, consultation_type) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_report_{consultation_type.value}_{timestamp}"

        data = {
            "tipo": "relatorio_audio",
            "tipo_consulta": consultation_type.value,
            "arquivo": str(report.file_path),
            "duracao_segundos": report.duration_seconds,
            "idioma": report.language,
            "score_risco": report.risk_score,
            "nivel_risco": report.risk_level,
            "fatores_risco": report.risk_factors,
            "resumo_emocoes": report.sentiment_summary,
            "alertas": [a.to_dict() for a in alerts],
            "transcricao": report.transcription[:500] + "..." if len(report.transcription) > 500 else report.transcription,
            "gerado_em": timestamp,
        }

        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        md_content = self._build_audio_markdown_report(data)
        md_path = self.output_dir / f"{filename}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Relatório de áudio exportado: {json_path}")
        return json_path

    def export_fusion_report(self, assessment) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fusion_report_{assessment.patient_id}_{timestamp}"

        data = assessment.to_dict()
        data["gerado_em"] = timestamp

        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        md_content = self._build_fusion_markdown_report(data)
        md_path = self.output_dir / f"{filename}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Relatório de fusão multimodal exportado: {json_path}")
        return json_path

    def _build_video_markdown_report(self, data: dict) -> str:
        alerts_md = ""
        for a in data.get("alertas", []):
            alerts_md += (
                f"| {a.get('timestamp_segundos', '-')} | {a.get('tipo_anomalia', '-')} | "
                f"{a.get('severidade', '-')} | {a.get('confianca', 0):.2f} | "
                f"{a.get('descricao', '-')[:80]} |\n"
            )

        return f"""# Relatório de Análise de Vídeo

**Tipo:** {data.get('tipo_video', '-')}
**Arquivo:** {data.get('arquivo', '-')}
**Duração:** {data.get('duracao_segundos', 0):.1f}s
**Frames Analisados:** {data.get('frames_analisados', 0)}/{data.get('total_frames', 0)}

## Resumo de Anomalias
```
{json.dumps(data.get('resumo_anomalias', {}), ensure_ascii=False, indent=2)}
```

## Alertas Gerados
| Timestamp | Tipo | Severidade | Confiança | Descrição |
|-----------|------|------------|-----------|-----------|
{alerts_md or '| - | Nenhum alerta | - | - | - |'}

---
*Gerado em: {data.get('gerado_em', '-')}*
"""

    def _build_audio_markdown_report(self, data: dict) -> str:
        alerts_md = ""
        for a in data.get("alertas", []):
            alerts_md += (
                f"| {a.get('tipo_alerta', '-')} | {a.get('nivel_risco', '-')} | "
                f"{a.get('score_risco', 0):.2f} | {a.get('evidencia', '-')[:80]} |\n"
            )

        return f"""# Relatório de Análise de Áudio

**Tipo de Consulta:** {data.get('tipo_consulta', '-')}
**Arquivo:** {data.get('arquivo', '-')}
**Duração:** {data.get('duracao_segundos', 0):.1f}s
**Idioma:** {data.get('idioma', '-')}

## Avaliação de Risco
- **Score:** {data.get('score_risco', 0):.2f}
- **Nível:** {data.get('nivel_risco', '-')}
- **Fatores de Risco:** {', '.join(data.get('fatores_risco', [])) or 'Nenhum'}

## Emoções Detectadas
```
{json.dumps(data.get('resumo_emocoes', {}), ensure_ascii=False, indent=2)}
```

## Alertas Gerados
| Tipo | Nível | Score | Evidência |
|------|-------|-------|-----------|
{alerts_md or '| - | - | - | - |'}

## Transcrição (parcial)
```
{data.get('transcricao', '-')}
```

---
*Gerado em: {data.get('gerado_em', '-')}*
"""

    def _build_fusion_markdown_report(self, data: dict) -> str:
        return f"""# Relatório de Fusão Multimodal

**Paciente:** {data.get('paciente_id', '-')}
**Data/Hora:** {data.get('timestamp', '-')}

## Score de Risco Unificado

| Modalidade | Score |
|------------|-------|
| Vídeo | {data.get('score_video', 0):.2f} |
| Áudio | {data.get('score_audio', 0):.2f} |
| Sinais Vitais | {data.get('score_sinais_vitais', 0):.2f} |
| **Geral** | **{data.get('score_risco_geral', 0):.2f}** |

**Nível de Risco:** {data.get('nivel_risco_geral', '-').upper()}

## Riscos Correlacionados
{chr(10).join(f'- {r}' for r in data.get('riscos_correlacionados', [])) or '- Nenhum'}

## Recomendações
{chr(10).join(f'- {r}' for r in data.get('recomendacoes', []))}

---
*Gerado em: {data.get('gerado_em', '-')}*
"""
