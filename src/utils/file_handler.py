"""
Utilitários de manipulação de arquivos para o sistema multimodal.

Critérios do edital atendidos:
- Processamento de dados multimodais (áudio, vídeo e texto)
- Suporte aos formatos de arquivo especificados
- Conversão entre formatos quando necessário
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class FileHandler:
    """Utilitário para manipulação de arquivos multimodais."""

    @staticmethod
    def validate_video(file_path: str | Path) -> bool:
        file_path = Path(file_path)
        valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        if not file_path.exists():
            logger.error(f"Arquivo não encontrado: {file_path}")
            return False
        if file_path.suffix.lower() not in valid_extensions:
            logger.error(f"Formato de vídeo não suportado: {file_path.suffix}")
            return False
        return True

    @staticmethod
    def validate_audio(file_path: str | Path) -> bool:
        file_path = Path(file_path)
        valid_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}
        if not file_path.exists():
            logger.error(f"Arquivo não encontrado: {file_path}")
            return False
        if file_path.suffix.lower() not in valid_extensions:
            logger.error(f"Formato de áudio não suportado: {file_path.suffix}")
            return False
        return True

    @staticmethod
    def convert_audio_format(
        input_path: str | Path,
        output_format: str = "wav",
        output_dir: str | Path | None = None,
    ) -> Path:
        input_path = Path(input_path)
        output_dir = Path(output_dir) if output_dir else input_path.parent
        output_path = output_dir / f"{input_path.stem}.{output_format}"

        try:
            import librosa
            import soundfile as sf

            audio, sr = librosa.load(str(input_path), sr=16000, mono=True)
            sf.write(str(output_path), audio, sr)
            logger.info(f"Áudio convertido: {input_path.name} -> {output_path.name}")
            return output_path
        except Exception as e:
            logger.error(f"Erro na conversão de áudio: {e}")
            raise

    @staticmethod
    def save_json(data: dict | list, file_path: str | Path) -> Path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return file_path

    @staticmethod
    def load_json(file_path: str | Path) -> dict:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def extract_audio_from_video(
        video_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        video_path = Path(video_path)
        if output_path is None:
            output_path = video_path.with_suffix(".wav")

        try:
            from moviepy.editor import VideoFileClip

            clip = VideoFileClip(str(video_path))
            clip.audio.write_audiofile(str(output_path), verbose=False, logger=None)
            clip.close()
            logger.info(f"Áudio extraído de vídeo: {output_path.name}")
            return Path(output_path)
        except Exception as e:
            logger.error(f"Erro ao extrair áudio do vídeo: {e}")
            raise
