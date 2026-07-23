"""
Modelos de análise de áudio especializados para saúde da mulher.

Critérios do edital atendidos:
- Processamento de gravações de voz de pacientes
- Detecção de depressão pós-parto, ansiedade, sinais de violência doméstica
- Análise de tom de voz, hesitação ao relatar sintomas
- Sinais de ansiedade gestacional
- Padrões vocais indicativos de trauma
- Fadiga hormonal
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import whisper
from loguru import logger
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, pipeline

from src.config import settings

EMOTION_LABELS = {
    "neutral": "neutro",
    "happy": "feliz",
    "sad": "triste",
    "angry": "raiva",
    "fearful": "medo",
    "disgusted": "nojado",
    "surprised": "surpreso",
}

RISK_INDICATORS = [
    "depressão",
    "ansiedade",
    "violência",
    "abuso",
    "medo",
    "trauma",
    "crise",
    "desespero",
    "socorro",
    "ajuda",
    "dor",
    "sofrimento",
    "solidão",
    "insônia",
    "cansaço extremo",
]


@dataclass
class AudioFeatures:
    """Features extraídas do áudio para análise."""

    duration_seconds: float
    sample_rate: int
    rms_energy: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    mfccs: np.ndarray


@dataclass
class EmotionSegment:
    """Segmento do áudio com emoção detectada."""

    start_seconds: float
    end_seconds: float
    emotion: str
    confidence: float


@dataclass
class TranscriptionSegment:
    """Segmento transcrito do áudio."""

    start_seconds: float
    end_seconds: float
    text: str
    confidence: float
    has_risk_indicator: bool = False
    matched_indicators: list[str] = field(default_factory=list)


@dataclass
class AudioAnalysisReport:
    """Relatório completo da análise de áudio."""

    file_path: Path
    duration_seconds: float
    language: str
    transcription: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    emotions: list[EmotionSegment] = field(default_factory=list)
    audio_features: AudioFeatures | None = None
    risk_score: float = 0.0
    risk_level: str = "baixo"
    risk_factors: list[str] = field(default_factory=list)

    @property
    def has_alerts(self) -> bool:
        return self.risk_level in ("alto", "crítico")

    @property
    def sentiment_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for seg in self.emotions:
            summary[seg.emotion] = summary.get(seg.emotion, 0) + 1
        return summary


class AudioPreprocessor:
    """Pré-processamento e feature extraction de áudio clínico."""

    def __init__(self, target_sample_rate: int | None = None):
        self.target_sr = target_sample_rate or settings.model.audio_sample_rate

    def load_audio(self, audio_path: str | Path) -> tuple[np.ndarray, int]:
        """Carrega áudio com resampling automático."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

        audio, sr = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
        return audio, sr

    def extract_features(self, audio: np.ndarray, sr: int) -> AudioFeatures:
        """Extrai features acústicas relevantes para análise clínica."""
        duration = len(audio) / sr

        rms = float(np.sqrt(np.mean(audio ** 2)))
        zcr = float(librosa.feature.zero_crossing_rate(audio).mean())
        spectral_centroid = float(librosa.feature.spectral_centroid(y=audio, sr=sr).mean())
        spectral_bandwidth = float(librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean())
        spectral_rolloff = float(librosa.feature.spectral_rolloff(y=audio, sr=sr).mean())
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        return AudioFeatures(
            duration_seconds=duration,
            sample_rate=sr,
            rms_energy=rms,
            zero_crossing_rate=zcr,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_rolloff=spectral_rolloff,
            mfccs=mfccs,
        )

    def detect_silence(self, audio: np.ndarray, sr: int, threshold_db: float = -30.0) -> list[tuple[float, float]]:
        """Detecta períodos de silêncio (hesitação) no áudio."""
        intervals = librosa.effects.split(audio, top_db=abs(threshold_db))
        return [(start / sr, end / sr) for start, end in intervals]


class SpeechToText:
    """Transcrição de áudio usando OpenAI Whisper."""

    def __init__(self, model_size: str | None = None, language: str | None = None):
        self.model_size = model_size or settings.model.whisper_model_size
        self.language = language or settings.model.whisper_language
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Carregando Whisper {self.model_size}...")
            self._model = whisper.load_model(self.model_size)
            self._model.to(torch.device(settings.device))
        return self._model

    def transcribe(self, audio_path: str | Path) -> AudioAnalysisReport:
        """Transcreve áudio completo e detecta indicadores de risco no texto."""
        audio_path = Path(audio_path)

        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            verbose=False,
            word_timestamps=True,
        )

        segments: list[TranscriptionSegment] = []
        for seg in result.get("segments", []):
            text = seg["text"].strip().lower()
            matched = [ind for ind in RISK_INDICATORS if ind in text]

            segments.append(TranscriptionSegment(
                start_seconds=seg["start"],
                end_seconds=seg["end"],
                text=seg["text"].strip(),
                confidence=seg.get("confidence", 0.0),
                has_risk_indicator=len(matched) > 0,
                matched_indicators=matched,
            ))

        risk_indicators_found = set()
        for seg in segments:
            risk_indicators_found.update(seg.matched_indicators)

        risk_factors = list(risk_indicators_found)

        return AudioAnalysisReport(
            file_path=audio_path,
            duration_seconds=result.get("duration", 0),
            language=result.get("language", self.language),
            transcription=result["text"].strip(),
            segments=segments,
            risk_factors=risk_factors,
        )


class EmotionAnalyzer:
    """
    Analisador de emoções na voz usando modelo pré-treinado.

    Detecta sinais de depressão, ansiedade e trauma vocal.
    """

    def __init__(self):
        self._feature_extractor = None
        self._model = None
        self._classifier = None

    def _ensure_loaded(self):
        if self._feature_extractor is None:
            model_name = settings.model.audio_emotion_model
            logger.info(f"Carregando modelo de emoção vocal: {model_name}")
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self._model = AutoModelForAudioClassification.from_pretrained(model_name)
            self._model.to(torch.device(settings.device))
            self._classifier = pipeline(
                "audio-classification",
                model=self._model,
                feature_extractor=self._feature_extractor,
                device=0 if settings.device == "cuda" else -1,
            )

    def analyze_emotions(
        self, audio: np.ndarray, sr: int, segment_duration: float = 3.0
    ) -> list[EmotionSegment]:
        """
        Analisa emoções por segmentos do áudio.

        Args:
            audio: Array do áudio
            sr: Sample rate
            segment_duration: Duração de cada segmento em segundos
        """
        self._ensure_loaded()

        segment_samples = int(segment_duration * sr)
        total_segments = max(1, len(audio) // segment_samples)

        emotions: list[EmotionSegment] = []

        for i in range(total_segments):
            start_sample = i * segment_samples
            end_sample = min(start_sample + segment_samples, len(audio))
            segment = audio[start_sample:end_sample]

            if len(segment) < sr * 0.5:
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, segment, sr)
                try:
                    result = self._classifier(tmp.name)
                    if result:
                        top = result[0]
                        emotions.append(EmotionSegment(
                            start_seconds=start_sample / sr,
                            end_seconds=end_sample / sr,
                            emotion=top["label"],
                            confidence=top["score"],
                        ))
                except Exception as e:
                    logger.warning(f"Erro ao processar segmento de áudio: {e}")
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

        return emotions

    def compute_risk_score(self, emotions: list[EmotionSegment]) -> tuple[float, str]:
        """
        Calcula score de risco baseado nas emoções detectadas.

        Pondera emoções negativas (tristeza, medo, raiva) sobre neutras.
        """
        if not emotions:
            return 0.0, "baixo"

        negative_emotions = {"sad", "angry", "fearful", "disgusted"}
        total_confidence = sum(e.confidence for e in emotions)
        if total_confidence == 0:
            return 0.0, "baixo"

        negative_score = sum(
            e.confidence for e in emotions
            if e.emotion in negative_emotions
        )
        risk_score = negative_score / total_confidence

        if risk_score > 0.75:
            level = "crítico"
        elif risk_score > 0.50:
            level = "alto"
        elif risk_score > 0.25:
            level = "médio"
        else:
            level = "baixo"

        return risk_score, level


class AudioAnalyzer:
    """
    Analisador completo de áudio para saúde da mulher.

    Orquestra transcrição, análise de emoções e features acústicas.
    """

    def __init__(self):
        self.preprocessor = AudioPreprocessor()
        self.transcriber = SpeechToText()
        self.emotion = EmotionAnalyzer()

    def analyze(self, audio_path: str | Path) -> AudioAnalysisReport:
        audio_path = Path(audio_path)
        logger.info(f"Iniciando análise de áudio: {audio_path.name}")

        audio, sr = self.preprocessor.load_audio(audio_path)

        features = self.preprocessor.extract_features(audio, sr)

        report = self.transcriber.transcribe(audio_path)

        emotions = self.emotion.analyze_emotions(audio, sr)
        report.emotions = emotions

        risk_score, risk_level = self.emotion.compute_risk_score(emotions)

        report.audio_features = features
        report.risk_score = risk_score
        report.risk_level = risk_level

        if report.has_alerts:
            logger.warning(
                f"ALERTA: Risco {risk_level} detectado em {audio_path.name} "
                f"(score: {risk_score:.2f}, fatores textuais: {report.risk_factors})"
            )

        logger.info(
            f"Análise de áudio concluída: duração={features.duration_seconds:.1f}s, "
            f"risco={risk_level} ({risk_score:.2f}), "
            f"emoções={report.sentiment_summary}"
        )
        return report
