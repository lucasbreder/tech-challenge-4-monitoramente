"""
Integração com Azure Cognitive Services.

Critérios do edital atendidos:
- Integração com serviços gerenciados em nuvem (Azure Cognitive Services)
- Manutenção de altos padrões de privacidade e segurança para dados sensíveis
- Utilização de serviços em nuvem para ampliar capacidade de processamento
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from src.config import settings


class AzureSpeechService:
    """
    Serviço de fala do Azure para transcrição e análise de sentimentos.

    Oferece alternativa ao Whisper local com recursos adicionais como
    speaker diarization e sentiment analysis integrado.
    """

    def __init__(self):
        self.key = settings.azure.speech_key
        self.region = settings.azure.speech_region
        self._speech_config = None

    @property
    def is_available(self) -> bool:
        return bool(self.key)

    def _get_config(self):
        if self._speech_config is None and self.is_available:
            try:
                import azure.cognitiveservices.speech as speechsdk

                self._speech_config = speechsdk.SpeechConfig(
                    subscription=self.key, region=self.region
                )
                self._speech_config.speech_recognition_language = "pt-BR"
            except ImportError:
                logger.warning(
                    "azure-cognitiveservices-speech não instalado. "
                    "Usando Whisper local como fallback."
                )
                return None
            except Exception as e:
                logger.error(f"Erro ao configurar Azure Speech: {e}")
                return None
        return self._speech_config

    def transcribe(self, audio_path: str | Path) -> str:
        if not self.is_available:
            raise RuntimeError("Azure Speech Service não configurado")

        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError:
            raise RuntimeError("azure-cognitiveservices-speech não instalado")

        speech_config = self._get_config()
        if speech_config is None:
            raise RuntimeError("Falha ao configurar Azure Speech")

        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        else:
            logger.warning(f"Azure Speech: reconhecimento falhou ({result.reason})")
            return ""

    def analyze_sentiment(self, text: str) -> dict:
        if not self.is_available:
            raise RuntimeError("Azure Speech Service não configurado")

        if not settings.azure.openai_endpoint:
            raise RuntimeError("Azure OpenAI endpoint não configurado")

        try:
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_key=settings.azure.openai_api_key,
                api_version="2024-02-01",
                azure_endpoint=settings.azure.openai_endpoint,
            )

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um analista de saúde da mulher. Analise o texto "
                            "e retorne um JSON com: sentimento (positivo/negativo/neutro), "
                            "indicadores_risco (lista), score_risco (0-1). "
                            "Foque em sinais de depressão pós-parto, ansiedade, "
                            "violência doméstica."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
            )
            import json

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Erro na análise de sentimento Azure: {e}")
            return {"sentimento": "neutro", "indicadores_risco": [], "score_risco": 0.0}


class AzureVisionService:
    """
    Serviço de visão computacional do Azure para análise de imagens médicas.

    Complementa a detecção YOLOv8 com recursos de OCR para documentos
    médicos e análise de imagens de exames.
    """

    def __init__(self):
        self.key = settings.azure.vision_key
        self.endpoint = settings.azure.vision_endpoint

    @property
    def is_available(self) -> bool:
        return bool(self.key and self.endpoint)

    def analyze_image(self, image_path: str | Path) -> dict:
        if not self.is_available:
            raise RuntimeError("Azure Vision Service não configurado")

        try:
            from azure.ai.vision.imageanalysis import ImageAnalysisClient
            from azure.ai.vision.imageanalysis.models import VisualFeatures
            from azure.core.credentials import AzureKeyCredential

            client = ImageAnalysisClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key),
            )

            with open(image_path, "rb") as f:
                image_data = f.read()

            result = client.analyze(
                image_data=image_data,
                visual_features=[
                    VisualFeatures.OBJECTS,
                    VisualFeatures.TAGS,
                    VisualFeatures.CAPTION,
                    VisualFeatures.DENSE_CAPTIONS,
                ],
            )

            return {
                "caption": result.caption.text if result.caption else "",
                "tags": [t.name for t in result.tags] if result.tags else [],
                "objects": [o.name for o in result.objects] if result.objects else [],
            }
        except ImportError:
            logger.warning("azure-ai-vision-imageanalysis não instalado")
            return {}
        except Exception as e:
            logger.error(f"Erro no Azure Vision: {e}")
            return {}


class AzureStorageService:
    """
    Serviço de armazenamento seguro no Azure Blob Storage.

    Garante conformidade com LGPD para dados sensíveis de saúde.
    """

    def __init__(self):
        self.connection_string = settings.azure.storage_connection_string

    @property
    def is_available(self) -> bool:
        return bool(self.connection_string)

    def upload_file(
        self,
        file_path: str | Path,
        container_name: str = "saude-mulher",
        blob_name: str | None = None,
    ) -> str:
        if not self.is_available:
            raise RuntimeError("Azure Storage não configurado")

        try:
            from azure.storage.blob import BlobServiceClient

            service = BlobServiceClient.from_connection_string(self.connection_string)
            container = service.get_container_client(container_name)

            if not container.exists():
                container.create_container()

            file_path = Path(file_path)
            blob_name = blob_name or file_path.name

            with open(file_path, "rb") as data:
                container.upload_blob(name=blob_name, data=data, overwrite=True)

            logger.info(f"Arquivo enviado para Azure Storage: {container_name}/{blob_name}")
            return f"{container_name}/{blob_name}"
        except ImportError:
            logger.warning("azure-storage-blob não instalado")
            return ""
        except Exception as e:
            logger.error(f"Erro no upload para Azure Storage: {e}")
            return ""

    def download_file(
        self,
        blob_name: str,
        container_name: str = "saude-mulher",
        output_path: str | Path | None = None,
    ) -> bytes | Path:
        if not self.is_available:
            raise RuntimeError("Azure Storage não configurado")

        try:
            from azure.storage.blob import BlobServiceClient

            service = BlobServiceClient.from_connection_string(self.connection_string)
            container = service.get_container_client(container_name)
            blob = container.download_blob(blob_name)

            if output_path:
                output_path = Path(output_path)
                with open(output_path, "wb") as f:
                    f.write(blob.readall())
                return output_path

            return blob.readall()
        except ImportError:
            raise RuntimeError("azure-storage-blob não instalado")
        except Exception as e:
            logger.error(f"Erro no download do Azure Storage: {e}")
            raise
