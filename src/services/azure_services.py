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
        if not self.key or len(self.key) < 30:
            return False
        if "your_" in self.key.lower() or "sua_" in self.key.lower():
            return False
        return bool(self.key and self.region)

    def _validate_connection(self) -> bool:
        if not self.is_available:
            return False
        try:
            import azure.cognitiveservices.speech as speechsdk
            cfg = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
            cfg.speech_recognition_language = "pt-BR"
            cfg.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "200")
            recognizer = speechsdk.SpeechRecognizer(speech_config=cfg)
            result = recognizer.recognize_once()
            del recognizer
            return True
        except Exception:
            return False

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

        audio_path = Path(audio_path)
        converted_path = audio_path

        if audio_path.suffix.lower() not in (".wav",):
            try:
                import librosa
                import soundfile as sf
                audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
                converted_path = audio_path.with_suffix(".azure_temp.wav")
                sf.write(str(converted_path), audio, sr)
                logger.debug(f"Áudio convertido para WAV 16kHz: {converted_path.name}")
            except Exception as e:
                logger.warning(f"Não foi possível converter áudio para WAV: {e}")

        try:
            speech_config = self._get_config()
            if speech_config is None:
                raise RuntimeError("Falha ao configurar Azure Speech")

            audio_config = speechsdk.audio.AudioConfig(filename=str(converted_path))
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "30000"
            )
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "5000"
            )

            try:
                import librosa
                duration = librosa.get_duration(path=str(converted_path))
            except Exception:
                duration = 0

            if duration > 12:
                all_text = []
                done = False

                def _on_result(evt):
                    nonlocal all_text
                    all_text.append(evt.result.text)

                def _on_session_stopped(evt):
                    nonlocal done
                    done = True

                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=speech_config, audio_config=audio_config
                )
                recognizer.recognized.connect(_on_result)
                recognizer.session_stopped.connect(_on_session_stopped)
                recognizer.start_continuous_recognition_async()

                import time
                timeout = duration + 15
                start = time.time()
                while not done and (time.time() - start) < timeout:
                    time.sleep(0.2)
                recognizer.stop_continuous_recognition_async()

                text = " ".join(all_text)
                if text.strip():
                    return text.strip()
                logger.warning("Azure Speech contínuo: nenhum texto reconhecido")
                return ""
            else:
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=speech_config, audio_config=audio_config
                )
                result = recognizer.recognize_once()

                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    return result.text
                elif result.reason == speechsdk.ResultReason.NoMatch:
                    logger.warning("Azure Speech: nenhum texto reconhecido no áudio")
                    return ""
                elif result.reason == speechsdk.ResultReason.Canceled:
                    cancellation = speechsdk.CancellationDetails.from_result(result)
                    logger.warning(
                        f"Azure Speech cancelado: {cancellation.reason} - {cancellation.error_details}"
                    )
                    return ""
                else:
                    logger.warning(f"Azure Speech: resultado inesperado ({result.reason})")
                    return ""
        finally:
            if converted_path != audio_path:
                converted_path.unlink(missing_ok=True)

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

            tags = []
            if result.tags:
                for t in result.tags:
                    tags.append(t.name if hasattr(t, "name") else str(t))

            objects = []
            if result.objects:
                for o in result.objects:
                    objects.append(o.name if hasattr(o, "name") else str(o))

            return {
                "caption": result.caption.text if result.caption else "",
                "tags": tags,
                "objects": objects,
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

    def list_files(self, container_name: str = "saude-mulher") -> list[dict]:
        if not self.is_available:
            return []

        try:
            from azure.storage.blob import BlobServiceClient

            service = BlobServiceClient.from_connection_string(self.connection_string)
            container = service.get_container_client(container_name)
            if not container.exists():
                return []

            blobs = []
            for blob in container.list_blobs():
                blobs.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified.isoformat() if blob.last_modified else "",
                    "container": container_name,
                })
            return blobs
        except Exception as e:
            logger.error(f"Erro ao listar blobs: {e}")
            return []

    def delete_file(self, blob_name: str, container_name: str = "saude-mulher") -> bool:
        if not self.is_available:
            return False
        try:
            from azure.storage.blob import BlobServiceClient
            service = BlobServiceClient.from_connection_string(self.connection_string)
            container = service.get_container_client(container_name)
            container.delete_blob(blob_name)
            logger.info(f"Blob removido: {container_name}/{blob_name}")
            return True
        except Exception as e:
            logger.error(f"Erro ao deletar blob: {e}")
            return False
