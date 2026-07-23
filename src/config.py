"""
Configuração centralizada do sistema de monitoramento multimodal para saúde da mulher.

Critérios atendidos:
- Configuração de serviços Azure (Cognitive Services)
- Parâmetros para todos os modelos (YOLO, Áudio, Anomalias)
- Segurança via variáveis de ambiente
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
AUDIOS_DIR = DATA_DIR / "audios"
REPORTS_DIR = DATA_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


class AzureConfig(BaseSettings):
    """Configurações dos serviços Azure Cognitive Services."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AZURE_", extra="ignore")

    speech_key: str = Field(default="", description="Azure Speech Services key")
    speech_region: str = Field(default="brazilsouth", description="Azure region")
    vision_key: str = Field(default="", description="Azure AI Vision key")
    vision_endpoint: str = Field(default="", description="Azure AI Vision endpoint")
    openai_api_key: str = Field(default="", description="Azure OpenAI key")
    openai_endpoint: str = Field(default="", description="Azure OpenAI endpoint")
    storage_connection_string: str = Field(default="", description="Azure Storage connection string")


class ModelConfig(BaseSettings):
    """Configurações dos modelos de IA."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    yolo_model_path: str = Field(default="yolov8n.pt", description="Caminho do modelo YOLOv8")
    yolo_pretrained: bool = Field(default=True, description="Usar pesos pré-treinados no COCO")
    yolo_confidence_threshold: float = Field(default=0.45, description="Limiar de confiança para detecção")
    yolo_iou_threshold: float = Field(default=0.5, description="Limiar de IOU para NMS")

    whisper_model_size: str = Field(
        default="medium", description="Tamanho do modelo Whisper (tiny/base/small/medium/large)"
    )
    whisper_language: str = Field(default="pt", description="Idioma principal para transcrição")

    audio_sample_rate: int = Field(default=16000, description="Taxa de amostragem do áudio")
    audio_emotion_model: str = Field(
        default="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        description="Modelo HuggingFace para reconhecimento de emoção em fala",
    )

    anomaly_contamination: float = Field(default=0.05, description="Proporção esperada de anomalias")
    anomaly_window_size: int = Field(default=60, description="Janela para detecção de anomalia (segundos)")
    anomaly_zscore_threshold: float = Field(default=3.0, description="Limiar Z-score para outlier")


class AlertConfig(BaseSettings):
    """Configurações do sistema de alertas."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ALERT_", extra="ignore")

    email_smtp_host: str = Field(default="smtp.gmail.com")
    email_smtp_port: int = Field(default=587)
    email_from: str = Field(default="alerts@hospital.com")
    email_to: str = Field(default="equipe_medica@hospital.com")
    email_password: str = Field(default="")

    severity_threshold_high: float = Field(default=0.85, description="Limiar para alerta de alta severidade")
    severity_threshold_medium: float = Field(default=0.65, description="Limiar para alerta de média severidade")


class Settings(BaseSettings):
    """Configuração global agrupada."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    azure: AzureConfig = Field(default_factory=AzureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)

    log_level: str = Field(default="INFO")
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")
    openai_api_key: str = Field(default="")


settings = Settings()
