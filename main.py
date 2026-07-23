"""
Orquestrador principal do sistema de monitoramento multimodal para saúde da mulher.

Fase 4 - Tech Challenge
Sistema de análise e fusão de dados multimodais (áudio, vídeo e sinais vitais)
para detecção precoce de riscos em saúde materna e ginecológica.

Uso:
    python main.py                          # Inicia interface interativa
    python main.py video <video.mp4>        # Analisa apenas vídeo
    python main.py audio <audio.wav>        # Analisa apenas áudio
    python main.py multimodal <id> --video <v> --audio <a>  # Pipeline completo
    python main.py demo                     # Inicia demo Streamlit
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agents.audio_agent import AudioConsultationType
from src.agents.video_agent import VideoType
from src.config import settings

app = typer.Typer(
    name="saude-mulher",
    help="Sistema de Monitoramento Multimodal para Saúde da Mulher",
    add_completion=False,
)
console = Console()


def print_banner():
    console.print(
        Panel.fit(
            "[bold cyan]Sistema de Monitoramento Multimodal[/bold cyan]\n"
            "[dim]Saúde da Mulher - Fase 4 Tech Challenge[/dim]\n\n"
            "[green]Modalidades:[/green] Vídeo | Áudio | Sinais Vitais\n"
            "[green]Serviços:[/green] Azure Cognitive Services\n"
            "[green]Modelos:[/green] YOLOv8 | Whisper | Isolation Forest",
            title="[bold]🔬 Bem-vinda[/bold]",
            border_style="cyan",
        )
    )


@app.command()
def video(
    video_path: str = typer.Argument(..., help="Caminho do arquivo de vídeo"),
    video_type: str = typer.Option("consulta", "--type", "-t", help="Tipo de vídeo (cirurgia/consulta/fisioterapia/triagem_violencia)"),
    sample_rate: int = typer.Option(10, "--sample-rate", "-s", help="Analisar 1 frame a cada N"),
    export: bool = typer.Option(True, "--export/--no-export", help="Exportar relatório"),
):
    """Analisa vídeo clínico (cirurgia, consulta, fisioterapia ou triagem de violência)."""
    from src.pipelines.video_pipeline import VideoPipeline

    print_banner()
    console.print(f"\n[bold]Analisando vídeo:[/bold] {video_path}")

    pipeline = VideoPipeline()
    report, alerts = pipeline.run(
        video_path=video_path,
        video_type=VideoType(video_type),
        sample_rate=sample_rate,
        export_report=export,
    )

    _display_video_results(report, alerts)


@app.command()
def audio(
    audio_path: str = typer.Argument(..., help="Caminho do arquivo de áudio"),
    consultation_type: str = typer.Option(
        "ginecologica", "--type", "-t",
        help="Tipo de consulta (ginecologica/pre_natal/pos_parto/vitima_violencia)",
    ),
    export: bool = typer.Option(True, "--export/--no-export", help="Exportar relatório"),
):
    """Analisa áudio de consulta médica (transcrição e emoções)."""
    from src.pipelines.audio_pipeline import AudioPipeline

    print_banner()
    console.print(f"\n[bold]Analisando áudio:[/bold] {audio_path}")

    pipeline = AudioPipeline()
    report, alerts = pipeline.run(
        audio_path=audio_path,
        consultation_type=AudioConsultationType(consultation_type),
        export_report=export,
    )

    _display_audio_results(report, alerts)


@app.command()
def multimodal(
    patient_id: str = typer.Argument(..., help="ID da paciente"),
    video_path: Optional[str] = typer.Option(None, "--video", "-v", help="Caminho do vídeo"),
    audio_path: Optional[str] = typer.Option(None, "--audio", "-a", help="Caminho do áudio"),
    video_type: str = typer.Option("consulta", "--video-type", help="Tipo de vídeo"),
    consultation_type: str = typer.Option("ginecologica", "--audio-type", help="Tipo de consulta"),
    export: bool = typer.Option(True, "--export/--no-export", help="Exportar relatório"),
):
    """Pipeline multimodal completo (vídeo + áudio + sinais vitais)."""
    from src.pipelines.multimodal_pipeline import MultimodalPipeline

    print_banner()
    console.print(f"\n[bold]Pipeline Multimodal - Paciente:[/bold] {patient_id}")

    pipeline = MultimodalPipeline()
    assessment = pipeline.run(
        patient_id=patient_id,
        video_path=video_path,
        audio_path=audio_path,
        video_type=video_type,
        consultation_type=consultation_type,
        export_report=export,
    )

    _display_fusion_results(assessment)


@app.command()
def demo():
    """Inicia a interface interativa Streamlit."""
    import subprocess
    import sys

    streamlit_path = Path(__file__).parent / "src" / "app.py"
    if not streamlit_path.exists():
        console.print("[red]Arquivo app.py não encontrado. Execute 'python main.py demo-ui' primeiro.[/red]")
        return

    console.print("[cyan]Iniciando interface Streamlit...[/cyan]")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(streamlit_path)])


@app.command()
def version():
    """Exibe informações da versão."""
    print_banner()
    console.print("[dim]Versão 1.0.0 - Fase 4 Tech Challenge[/dim]")


def _display_video_results(report, alerts):
    table = Table(title="Resultados da Análise de Vídeo")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="green")

    table.add_row("Duração", f"{report.video_duration_seconds:.1f}s")
    table.add_row("Frames analisados", str(report.frames_analyzed))
    table.add_row("Total detecções", str(len(report.detections)))
    table.add_row("Anomalias", str(report.anomaly_count))

    console.print(table)

    if alerts:
        console.print("\n[bold red]⚠ Alertas Gerados:[/bold red]")
        for alert in alerts:
            console.print(
                f"  [{alert.severity}] {alert.timestamp_seconds:.1f}s - "
                f"{alert.anomaly_type} (conf: {alert.confidence:.2f})"
            )
    else:
        console.print("\n[green]✓ Nenhuma anomalia crítica detectada.[/green]")


def _display_audio_results(report, alerts):
    table = Table(title="Resultados da Análise de Áudio")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="green")

    table.add_row("Duração", f"{report.duration_seconds:.1f}s")
    table.add_row("Idioma", report.language)
    table.add_row("Score de risco", f"{report.risk_score:.2f}")
    table.add_row("Nível de risco", report.risk_level.upper())
    table.add_row("Fatores de risco", ", ".join(report.risk_factors) or "Nenhum")

    risk_color = "red" if report.risk_level in ("alto", "crítico") else "green"
    console.print(table)

    console.print(f"\n[bold {risk_color}]Nível de Risco: {report.risk_level.upper()}[/bold {risk_color}]")

    if report.emotions:
        console.print("\n[bold]Emoções detectadas:[/bold]")
        for emotion, count in report.sentiment_summary.items():
            console.print(f"  {emotion}: {count} segmentos")

    if alerts:
        console.print("\n[bold red]⚠ Alertas Gerados:[/bold red]")
        for alert in alerts:
            console.print(f"  [{alert.risk_level}] {alert.alert_type}: {alert.evidence[:100]}...")
    else:
        console.print("\n[green]✓ Nenhum alerta crítico gerado.[/green]")


def _display_fusion_results(assessment):
    table = Table(title="Resultado da Fusão Multimodal")
    table.add_column("Modalidade", style="cyan")
    table.add_column("Score", style="green")

    table.add_row("Vídeo", f"{assessment.video_risk_score:.2f}")
    table.add_row("Áudio", f"{assessment.audio_risk_score:.2f}")
    table.add_row("Sinais Vitais", f"{assessment.vitals_risk_score:.2f}")
    table.add_row("[bold]Geral[/bold]", f"[bold]{assessment.overall_risk_score:.2f}[/bold]")

    console.print(table)

    risk_color = "red" if assessment.overall_risk_level in ("crítico", "alto") else "green"
    console.print(
        f"\n[bold {risk_color}]Risco Global: {assessment.overall_risk_level.upper()}[/bold {risk_color}]"
    )

    if assessment.correlated_risks:
        console.print("\n[bold yellow]⚠ Riscos Correlacionados:[/bold yellow]")
        for risk in assessment.correlated_risks:
            console.print(f"  • {risk}")

    console.print("\n[bold cyan]Recomendações:[/bold cyan]")
    for rec in assessment.recommendations:
        console.print(f"  • {rec}")


if __name__ == "__main__":
    app()
